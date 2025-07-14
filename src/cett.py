from collections import defaultdict
import logging
import os
import json
from tqdm import tqdm
import argparse

from datasets import load_dataset
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from src.activation_capture import ActivationCapture, Hook

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_threshold_one_token(neuron_outputs, cett_target, n_quantiles=1000):
    norms = neuron_outputs.norm(dim=0)
    quantiles = norms.quantile(torch.linspace(0,1,n_quantiles))
    tot_norm = neuron_outputs.sum(dim=1).norm()

    def CETT(threshold):
        threshold_norm = ((norms < threshold) * neuron_outputs).sum(dim=1).norm()
        return threshold_norm / tot_norm

    left = 0
    right = quantiles.size(0)
    threshold = 0
    while left < right:
        mid = (left + right) // 2
        cett = CETT(quantiles[mid])
        if cett <= cett_target:
            left = mid + 1
            threshold = quantiles[mid]
        else:
            right = mid - 1
    return threshold


def find_thresholds(
        model_name: str, 
        dataset_name: str, 
        dataset_config: str,
        max_samples: int, 
        cett_target: float, 
        n_quantiles: int,
        save_path: str,
        seed: int,
        device: torch.device,
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
        tokenized = tokenizer(texts, return_tensors="pt")

        # Convert to lists
        return {
            "text": texts,
            "input_ids": tokenized["input_ids"],
        }

    # Tokenize
    dataset = dataset.take(max_samples).map(sample_and_tokenize, batched=False)
    dataset = dataset.with_format("torch")

    dataloader = TorchDataLoader(dataset, batch_size=1, num_workers=8, pin_memory=False, prefetch_factor=2)  # type: ignore

    # Compute thresholds for each layer across all dataset entries
    logger.info(f"Beginning to compute thresholds using {max_samples} samples")
    thresholds = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=max_samples):
            input_ids = batch["input_ids"].to(device)
        
            _ = model(input_ids.squeeze(0))

            for layer_idx, layer in enumerate(model.activation_capture.get_layers()):
                activations = model.activation_capture.mlp_activations[Hook.UP][layer_idx]
                activations = activations.view(-1, activations.size(-1))

                for i in range(activations.size(0)):
                    neuron_outputs = activations[i] * layer.mlp.down_proj.weight
                    threshold = calculate_threshold_one_token(neuron_outputs, cett_target=cett_target, n_quantiles=n_quantiles)
                    thresholds[layer_idx].append(threshold)
            
            model.activation_capture.clear_captures()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for layer_idx, layer_thresholds in thresholds.items():
        thresholds[layer_idx] = sum(layer_thresholds) / len(layer_thresholds)

    # Save layerwise thresholds as record in central json file
    if not os.path.exists(save_path):
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
                
