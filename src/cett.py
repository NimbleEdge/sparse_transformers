from collections import defaultdict
import logging
import os
import json
from tqdm import tqdm
import argparse

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from src.activation_capture import ActivationCapture, Hook

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# from https://github.com/pytorch/pytorch/issues/64947#issuecomment-2810054982
def quantile(tensor, q, dim=None, keepdim=False):
    """
    Computes the quantile of the input tensor along the specified dimension.
    
    Parameters:
    tensor (torch.Tensor): The input tensor.
    q (float): The quantile to compute, should be a float between 0 and 1.
    dim (int): The dimension to reduce. If None, the tensor is flattened.
    keepdim (bool): Whether to keep the reduced dimension in the output.
    Returns:
    torch.Tensor: The quantile value(s) along the specified dimension.
    """
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    # linear interpolation
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value



def cett_from_threshold(activations, down_weight, threshold, norms=None, tot_norm=None):
    if norms is None:
        col_norms = down_weight.norm(dim=0)
        norms = activations.abs() * col_norms
        tot_norm = activations.matmul(down_weight.t()).norm(dim=-1)
    masked_act = activations * (norms < threshold)
    threshold_norm = masked_act.matmul(down_weight.t()).norm(dim=-1)
    return threshold_norm / tot_norm


def calculate_threshold(activations, down_weight, col_norms, cett_target, n_thresholds=1000):
    norms = activations.abs() * col_norms
    output = activations.matmul(down_weight.t())
    tot_norm = output.norm(dim=-1)

    min_value = norms.min()
    max_value = quantile(norms, 0.99)
    threshold_grid = torch.linspace(min_value, max_value, n_thresholds)
    max_cett = cett_from_threshold(activations, down_weight, max_value, norms=norms, tot_norm=tot_norm)
    outlier_mask = max_cett > cett_target

    left = 0
    right = n_thresholds
    while left < right:
        #print(left,right)
        mid = (left + right) // 2
        cett = cett_from_threshold(activations, down_weight, threshold_grid[mid], norms=norms, tot_norm=tot_norm) # Compute CETT for each token
        cett = cett[outlier_mask].mean()    # Remove outliers and take average
        if cett <= cett_target:
            left = mid + 1
        else:
            right = mid
    return threshold_grid[left]


def find_thresholds(
        model_name: str, 
        dataset_name: str, 
        dataset_config: str,
        save_path: str,
        batch_size: int = 8,
        max_samples: int = 128, 
        max_length: int = 256,
        cett_target: float = 0.2, 
        n_thresholds: int = 10000,
        num_workers: int = 8,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    model.activation_capture = ActivationCapture(model)
    model.activation_capture.register_hooks(hooks=[Hook.UP])

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(
            dataset_name, dataset_config, split="train", streaming=True
        )
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=10000, seed=seed)

    def sample_and_tokenize(examples):
        """Sample text chunks before tokenization for efficiency using vectorized operations."""
        texts = examples["text"]
        tokenized = tokenizer(
            texts, 
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Convert to lists
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    # Tokenize
    dataset = dataset.take(max_samples).map(sample_and_tokenize, batched=False)
    dataset = dataset.with_format("torch")

    dataloader = TorchDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, prefetch_factor=2)  # type: ignore

    # Compute thresholds for each layer across all dataset entries
    logger.info(f"Beginning to compute thresholds using {max_samples} samples")
    thresholds = defaultdict(list)
    with torch.no_grad():
        all_col_norms = {layer_idx: layer.mlp.down_proj.weight.norm(dim=0) \
                         for layer_idx, layer in enumerate(model.activation_capture.get_layers())}
        for batch in tqdm(dataloader, total=max_samples//batch_size):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
        
            _ = model(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))

            for layer_idx, layer in enumerate(model.activation_capture.get_layers()):
                down_weight = layer.mlp.down_proj.weight
                col_norms = all_col_norms[layer_idx]
                activations = model.activation_capture.mlp_activations[Hook.UP][layer_idx]
                threshold = calculate_threshold(activations, down_weight, col_norms, cett_target, n_thresholds)
                thresholds[layer_idx].append(threshold)
            
            model.activation_capture.clear_captures()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for layer_idx, layer_thresholds in thresholds.items():
        thresholds[layer_idx] = sum(layer_thresholds) / len(layer_thresholds)

    # Save layerwise thresholds as record in central json file
    if os.path.exists(save_path):
        with open("save_path", mode="r", encoding="utf-8") as read_file:
            threshold_dict = json.load(read_file)
    else:
        threshold_dict = {}
    threshold_dict[model_name] = thresholds
    with open("save_path", mode="r", encoding="utf-8") as write_file:
        json.dump(threshold_dict, write_file)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training dataset for sparsity predictors"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the base model (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/c4",
        help="Dataset name (default: allenai/c4)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="en",
        help="Dataset configuration (e.g., en for C4)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="thresholds.json",
        help="Path to json file for thresholds",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--cett_target",
        type=float,
        default=0.2,
        help="Optimal CETT value for threshold-finding",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=500,
        help="Number of quantiles to sort neuron outputs into for threshold-finding",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    find_thresholds(
        model_name=args.model_name, 
        dataset_name=args.dataset, 
        dataset_config=args.dataset_config,
        max_samples=args.max_samples, 
        cett_target=args.cett_target, 
        n_quantiles=args.n_quantiles,
        save_path=args.save_path,
        seed=args.seed,
        device=device,
    )
                
