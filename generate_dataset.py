#!/usr/bin/env python3
"""
Generate training dataset for sparsity predictors.

This script runs a standard LLaMA model on text data and captures:
- Input text
- Hidden states for each token before each MLP layer
- MLP activations for each token at each layer

The data is saved with sequence structure preserved for training predictors.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import shutil
import random

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import set_seed
from datasets import load_dataset, Dataset, Features, Value, Array2D, Sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.activation_capture import ActivationCapture

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_layer_dataset(
    texts_list: List[str],
    input_ids_list: List[np.ndarray],
    hidden_states_dict: Dict[int, List[np.ndarray]], 
    mlp_activations_dict: Dict[int, List[np.ndarray]],
    num_layers: int,
    hidden_dim: int,
    intermediate_dim: int,
    max_length: int
) -> Dataset:
    """Create a HuggingFace dataset from collected activations with sequence structure."""
    
    # Prepare data for dataset
    dataset_dict = {
        "text": texts_list,
        "input_ids": input_ids_list,
    }
    
    # Add hidden states and activations for each layer
    for layer_idx in range(num_layers):
        if layer_idx in hidden_states_dict:
            dataset_dict[f"hidden_states_layer_{layer_idx}"] = hidden_states_dict[layer_idx]
            dataset_dict[f"mlp_activations_layer_{layer_idx}"] = mlp_activations_dict[layer_idx]
    
    # Define features with proper shapes
    features_dict = {
        "text": Value("string"),
        "input_ids": Sequence(feature=Value("int32"), length=max_length),
    }
    
    # Add features for each layer
    for layer_idx in range(num_layers):
        if f"hidden_states_layer_{layer_idx}" in dataset_dict:
            features_dict[f"hidden_states_layer_{layer_idx}"] = Array2D(
                shape=(max_length, hidden_dim), dtype="float32"
            )
            features_dict[f"mlp_activations_layer_{layer_idx}"] = Array2D(
                shape=(max_length, intermediate_dim), dtype="float32"
            )
    
    features = Features(features_dict)
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    return dataset


def process_batch(
    tokenized_batch: Dict[str, torch.Tensor],
    model,
    capture: ActivationCapture,
    device: torch.device,
    num_layers: int
) -> Tuple[List[np.ndarray], Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    """Process a batch of texts and return all activations."""
    
    # Move to device
    input_ids = tokenized_batch["input_ids"].to(device)
    attention_mask = tokenized_batch["attention_mask"].to(device)
    
    # Clear previous captures and GPU cache
    capture.clear_captures()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Pre-allocate arrays for efficiency
    batch_size = input_ids.shape[0]
    input_ids_list = []
    hidden_states_dict = {i: [] for i in range(num_layers)}
    mlp_activations_dict = {i: [] for i in range(num_layers)}
    
    # Move attention mask to CPU once
    attention_mask_np = attention_mask.cpu().numpy().astype(np.int8)
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Get sequence length from attention mask
        seq_len = int(attention_mask_np[batch_idx].sum())
        
        # Extract input_ids for this sample
        input_ids_sample = input_ids[batch_idx, :seq_len].cpu().numpy().astype(np.int32)
        input_ids_list.append(input_ids_sample)
        
        # Collect activations for each layer
        for layer_idx in range(num_layers):
            if layer_idx in capture.hidden_states:
                # Move only the needed slice to CPU
                hidden_state = capture.hidden_states[layer_idx][batch_idx, :seq_len, :].cpu().numpy().astype(np.float32)
                hidden_states_dict[layer_idx].append(hidden_state)
                
                # Get MLP activations
                mlp_activation = capture.get_mlp_activations(layer_idx)
                if mlp_activation is not None:
                    mlp_act = mlp_activation[batch_idx, :seq_len, :].cpu().numpy().astype(np.float32)
                    mlp_activations_dict[layer_idx].append(mlp_act)
    
    # Clear GPU tensors from capture to free memory
    capture.clear_captures()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return input_ids_list, hidden_states_dict, mlp_activations_dict


def generate_dataset(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    output_dir: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    save_interval: int = 1000,
    num_workers: int = 4,
    prefetch_batches: int = 2,
    max_samples: int = 100000
):
    """Generate predictor training dataset with optimizations."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
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
        raw_dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=False)
    else:
        raw_dataset = load_dataset(dataset_name, split="train", streaming=False)
    
    # Ensure we have a Dataset object (not DatasetDict)
    if hasattr(raw_dataset, '__getitem__'):
        dataset = raw_dataset  # type: ignore
    else:
        raise ValueError("Expected a Dataset object, got: " + type(raw_dataset).__name__)

    def sample_and_tokenize(examples):
        """Sample text chunks before tokenization for efficiency using vectorized operations."""
        texts = examples["text"]
        chars_per_token = 4
        target_chars = max_length * chars_per_token * 2
        
        # Vectorized length calculation
        text_lengths = np.array([len(text) for text in texts])
        
        # Process all texts
        sampled_texts = []
        for idx in range(len(texts)):
            text = texts[idx]
            text_len = text_lengths[idx]
            
            if text_len > target_chars:
                # Vectorized random sampling
                max_start = text_len - target_chars
                start_idx = np.random.randint(0, max_start + 1)
                
                # Simple word boundary adjustment (simplified for speed)
                # Find space before start_idx
                space_before = text.rfind(' ', 0, start_idx + 1)
                start_idx = space_before + 1 if space_before != -1 else start_idx
                
                # Find space after end_idx
                end_idx = min(int(start_idx + target_chars), int(text_len))
                space_after = text.find(' ', end_idx - 1)
                end_idx = space_after if space_after != -1 else end_idx
                
                sampled_texts.append(text[start_idx:end_idx].strip())
            else:
                sampled_texts.append(text)
        
        # Batch tokenization - much faster than individual tokenization
        if not sampled_texts:
            return {"text": [], "input_ids": [], "attention_mask": []}
            
        tokenized = tokenizer(
            sampled_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"  # Return numpy arrays for faster operations
        )
        
        # Convert to lists
        return {
            "text": sampled_texts,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
        
    num_samples = min(max_samples, len(dataset))
    logger.info(f"Processing {num_samples} samples from dataset")
    
    # Select subset and tokenize
    dataset = dataset.select(range(num_samples))
    dataset = dataset.map(sample_and_tokenize, batched=True)
    dataset = dataset.with_format("torch")
    
    # Create DataLoader with num_workers=0 to avoid shared memory issues
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    # Storage for collected data
    texts_list = []
    input_ids_list = []
    hidden_states_dict = {i: [] for i in range(num_layers)}
    mlp_activations_dict = {i: [] for i in range(num_layers)}
    
    # Process samples
    logger.info(f"Using batch size: {batch_size}")
    
    # Process in larger batches for efficiency
    with torch.no_grad():
        # Process samples in batches
        for batch in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):            
            # Process batch
            input_ids_list_, hidden_states_dict_, mlp_activations_dict_ = process_batch(
                batch, model, capture, device, num_layers
            )
            
            # Extend lists with batch results
            texts_list.extend(batch["text"])
            input_ids_list.extend(input_ids_list_)
            
            # Extend layer dictionaries
            for layer_idx in range(num_layers):
                if layer_idx in hidden_states_dict:
                    hidden_states_dict[layer_idx].extend(hidden_states_dict_[layer_idx])
                    mlp_activations_dict[layer_idx].extend(mlp_activations_dict_[layer_idx])
                        
            # Save intermediate results periodically
            if len(texts_list) % save_interval == 0 and len(texts_list) > 0:
                logger.info(f"Saving intermediate results at {len(texts_list)} samples...")
                save_dataset(
                    texts_list, input_ids_list,
                    hidden_states_dict, mlp_activations_dict,
                    output_dir, num_layers, hidden_dim, intermediate_dim,
                    max_length, intermediate=True
                )
    
    # Remove hooks
    capture.remove_hooks()
    
    # Save final dataset
    logger.info("Saving final dataset...")
    save_dataset(
        texts_list, input_ids_list,
        hidden_states_dict, mlp_activations_dict,
        output_dir, num_layers, hidden_dim, intermediate_dim,
        max_length, intermediate=False
    )
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "num_samples": len(texts_list),
        "max_length": max_length,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "batch_size": batch_size,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset generation complete. Processed {len(texts_list)} samples.")


def save_dataset(
    texts_list: List[str],
    input_ids_list: List[np.ndarray],
    hidden_states_dict: Dict[int, List[np.ndarray]],
    mlp_activations_dict: Dict[int, List[np.ndarray]],
    output_dir: str,
    num_layers: int,
    hidden_dim: int,
    intermediate_dim: int,
    max_length: int,
    intermediate: bool = False
):
    """Save dataset in HuggingFace format."""
    
    if not texts_list:
        logger.warning("No data to save")
        return
    
    # Create dataset
    dataset = create_layer_dataset(
        texts_list, input_ids_list,
        hidden_states_dict, mlp_activations_dict,
        num_layers, hidden_dim, intermediate_dim, max_length
    )
    
    # Save path
    if intermediate:
        save_path = os.path.join(output_dir, "intermediate")
    else:
        save_path = output_dir
    
    # Save dataset in Arrow format with optimal settings
    dataset.save_to_disk(
        save_path,
        num_shards=4,  # Split into multiple files for faster I/O
        num_proc=4     # Use multiple processes for saving
    )
    
    logger.info(f"Saved dataset: {len(dataset)} samples to {save_path}")
    
    # If this is the final save and intermediate exists, remove it
    if not intermediate and os.path.exists(os.path.join(output_dir, "intermediate")):
        shutil.rmtree(os.path.join(output_dir, "intermediate"))


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
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Save intermediate results every N samples")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--prefetch_batches", type=int, default=2,
                       help="Number of batches to prefetch")
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
    
    # Set number of threads for CPU operations
    if device.type == "cpu":
        torch.set_num_threads(args.num_workers)
    
    # Generate dataset
    generate_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
