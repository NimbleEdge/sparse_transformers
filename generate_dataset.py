#!/usr/bin/env python3
"""
Generate training dataset for sparsity predictors.

This script runs a standard LLaMA model on text data and captures:
- Hidden states before each MLP layer (inputs for predictor)
- MLP activations (ground truth for what neurons are important)

The data is saved per-layer to be used for training predictors.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import shutil
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from datasets import load_dataset, Dataset, Features, Value, Array2D
from tqdm import tqdm
from src.activatiion_capture import ActivationCapture

# Suppress the specific warning from HuggingFace datasets about torch.tensor()
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use sourceTensor.detach()", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_layer_dataset(hidden_states_list: List[np.ndarray], 
                        mlp_activations_list: List[np.ndarray],
                        hidden_dim: int,
                        intermediate_dim: int) -> Dataset:
    """Create a HuggingFace dataset from collected activations."""
    
    # Concatenate all batches
    all_hidden_states = np.concatenate(hidden_states_list, axis=0)
    all_mlp_activations = np.concatenate(mlp_activations_list, axis=0)
    
    # Create dataset dict
    dataset_dict = {
        "hidden_states": all_hidden_states,
        "mlp_activations": all_mlp_activations
    }
    
    # Define features
    features = Features({
        "hidden_states": Array2D(shape=(hidden_dim,), dtype="float32"),
        "mlp_activations": Array2D(shape=(intermediate_dim,), dtype="float32")
    })
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    return dataset


def generate_dataset(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    output_dir: str,
    num_samples: int,
    max_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    save_interval: int = 10000
):
    """Generate predictor training dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device.type == "cuda" else None
    )
    
    if device.type != "cuda":
        model = model.to(device)
    
    model.eval()
    
    # Get model dimensions
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_layers = len(model.model.layers)
    
    # Setup activation capture
    capture = ActivationCapture()
    capture.register_hooks(model)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Process dataset
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.with_format("torch")
    
    # Create data loader
    dataloader = DataLoader(
        dataset.take(num_samples),
        batch_size=batch_size,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Initialize storage for each layer
    layer_data = {
        i: {
            "hidden_states": [],
            "mlp_activations": []
        } for i in range(num_layers)
    }
    
    # Process batches
    logger.info(f"Processing {num_samples} samples...")
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating dataset")):
            if samples_processed >= num_samples:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Clear previous captures
            capture.clear_captures()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Collect activations for each layer
            for layer_idx in range(num_layers):
                if layer_idx in capture.hidden_states:
                    hidden_state = capture.hidden_states[layer_idx].detach().clone().cpu()
                    mlp_activation = capture.get_mlp_activations(layer_idx).detach().clone().cpu()
                    
                    if mlp_activation is not None:
                        # Store flattened versions for training
                        # Shape: [batch_size * seq_len, hidden_dim]
                        hidden_flat = hidden_state.view(-1, hidden_state.shape[-1])
                        mlp_flat = mlp_activation.view(-1, mlp_activation.shape[-1])
                        attention_flat = attention_mask.view(-1).cpu()
                        
                        # Only keep non-padded positions
                        valid_mask = attention_flat > 0
                        
                        # Convert to numpy for HuggingFace datasets
                        layer_data[layer_idx]["hidden_states"].append(
                            hidden_flat[valid_mask].numpy().astype(np.float32)
                        )
                        layer_data[layer_idx]["mlp_activations"].append(
                            mlp_flat[valid_mask].numpy().astype(np.float32)
                        )
            
            samples_processed += batch_size
            
            # Save intermediate results periodically
            if samples_processed % save_interval == 0:
                logger.info(f"Saving intermediate results at {samples_processed} samples...")
                save_datasets(layer_data, output_dir, num_layers, hidden_dim, 
                            intermediate_dim, intermediate=True)
    
    # Remove hooks
    capture.remove_hooks()
    
    # Save final datasets
    logger.info("Saving final datasets...")
    save_datasets(layer_data, output_dir, num_layers, hidden_dim, 
                intermediate_dim, intermediate=False)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "num_samples": samples_processed,
        "max_length": max_length,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset generation complete. Processed {samples_processed} samples.")


def save_datasets(layer_data: Dict, output_dir: str, num_layers: int,
                 hidden_dim: int, intermediate_dim: int, intermediate: bool = False):
    """Save layer datasets in HuggingFace format."""
    
    for layer_idx in range(num_layers):
        if layer_data[layer_idx]["hidden_states"]:
            # Create dataset for this layer
            dataset = create_layer_dataset(
                layer_data[layer_idx]["hidden_states"],
                layer_data[layer_idx]["mlp_activations"],
                hidden_dim,
                intermediate_dim
            )
            
            # Save path
            layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
            if intermediate:
                save_path = os.path.join(layer_dir, "intermediate")
            else:
                save_path = layer_dir
            
            # Save dataset in Arrow format (efficient for large datasets)
            dataset.save_to_disk(save_path)
            
            logger.info(f"Saved layer {layer_idx}: {len(dataset)} samples to {save_path}")
            
            # If this is the final save and intermediate exists, remove it
            if not intermediate and os.path.exists(os.path.join(layer_dir, "intermediate")):
                shutil.rmtree(os.path.join(layer_dir, "intermediate"))


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for sparsity predictors")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name or path of the base model (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--dataset", type=str, default="allenai/c4",
                       help="Dataset name (default: allenai/c4)")
    parser.add_argument("--dataset_config", type=str, default="realnewslike",
                       help="Dataset configuration (e.g., realnewslike for C4)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for generated dataset")
    parser.add_argument("--num_samples", type=int, default=50000,
                       help="Number of samples to process")
    parser.add_argument("--max_length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--save_interval", type=int, default=10000,
                       help="Save intermediate results every N samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Generate dataset
    generate_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
