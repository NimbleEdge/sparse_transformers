#!/usr/bin/env python3
"""
Generate training dataset for sparsity predictors.

This script runs a standard LLaMA model on text data and captures:
- Input text
- Hidden states for the last token before each MLP layer
- MLP activations for the last token at each layer

The data is saved incrementally using:
- .npz files for numpy arrays (compressed, one file per save batch)
- Single CSV file for metadata (text, batch references)

This approach avoids loading full datasets into memory and allows for:
- Resumable processing
- Memory-efficient storage with optimal compression
- Lazy loading of arrays when needed

Note: Only the last token's representations are saved to reduce storage requirements
and focus on the final contextual representations for each sequence.

Usage examples:
  # Generate dataset
  python generate_dataset.py --model_name meta-llama/Llama-3.2-3B-Instruct --output_dir ./data/c4 --max_samples 100000 --device cuda --save_interval 500
  
  # Show dataset statistics without loading arrays
  python generate_dataset.py --show_stats --output_dir data/c4
  
  # Inspect a specific sample
  python generate_dataset.py --inspect_sample 0 --output_dir data/c4
  
  # Create unified HuggingFace dataset from CSV file after processing (optional)
  python generate_dataset.py --create_unified_dataset --output_dir data/c4
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from src.activation_capture import ACTIVATION_CAPTURE, ActivationCapture
import csv
import glob
from src.trainer import get_sample_by_index
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset_stats(output_dir: str) -> Optional[Dict]:
    """Get dataset statistics without loading arrays into memory."""
    try:
        csv_file = os.path.join(output_dir, "dataset.csv")
        if not os.path.exists(csv_file):
            return None
        
        arrays_dir = os.path.join(output_dir, "arrays")
        batch_files = glob.glob(os.path.join(arrays_dir, "batch_*.npz"))
        
        total_samples = 0
        
        # Count samples in the single CSV file
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                total_samples = sum(1 for _ in reader)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
        
        # Estimate storage sizes
        metadata_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
        arrays_size = sum(os.path.getsize(bf) for bf in batch_files if os.path.exists(bf))
        
        # Calculate average samples per batch
        avg_samples_per_batch = 0
        if batch_files and total_samples > 0:
            avg_samples_per_batch = total_samples / len(batch_files)
        
        return {
            "total_samples": total_samples,
            "total_batches": len(batch_files),
            "avg_samples_per_batch": int(avg_samples_per_batch),
            "metadata_size_mb": metadata_size / (1024 * 1024),
            "arrays_size_mb": arrays_size / (1024 * 1024),
            "total_size_mb": (metadata_size + arrays_size) / (1024 * 1024),
            "compression_ratio": f"{arrays_size / max(1, metadata_size):.1f}x"
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        return None


def process_batch(
    tokenized_batch: Dict[str, torch.Tensor],
    model,
    capture: ActivationCapture,
    device: torch.device,
    num_layers: int
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    """Process a batch of texts and return last token activations for each sample."""
    
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
    attention_mask_list = []
    # Move attention mask to CPU once
    attention_mask_np = attention_mask.cpu().numpy().astype(np.int8)
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Extract input_ids for this sample
        input_ids_sample = input_ids[batch_idx].cpu().numpy().astype(np.int32)
        input_ids_list.append(input_ids_sample)
        attention_mask_list.append(attention_mask_np[batch_idx])
        
        # Collect last token activations for each layer
        for layer_idx in range(num_layers):
            if layer_idx in capture.hidden_states:
                # Extract only the last token's hidden state [-1, :]
                hidden_state = capture.hidden_states[layer_idx][batch_idx,-1,:].cpu().numpy().astype(np.float32)
                hidden_states_dict[layer_idx].append(hidden_state)
                
                # Get last token's MLP activations
                mlp_activation = capture.get_gate_activations(layer_idx)
                if mlp_activation is not None:
                    mlp_act = mlp_activation[batch_idx,-1,:].cpu().numpy().astype(np.float32)
                    mlp_activations_dict[layer_idx].append(mlp_act)
    
    # Clear GPU tensors from capture to free memory
    capture.clear_captures()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return input_ids_list, attention_mask_list, hidden_states_dict, mlp_activations_dict


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
    max_samples: int = 100000,
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
    
    # Setup activation capture
    capture_cls = ACTIVATION_CAPTURE[model.config.model_type]
    capture = capture_cls()
    capture.register_hooks(model)

    # Get model dimensions
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_layers = len(capture.get_layers(model))
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        raw_dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=False)
    else:
        raw_dataset = load_dataset(dataset_name, split="train", streaming=False)
    
    # Ensure we have a Dataset object (not DatasetDict)
    from datasets import Dataset as HFDataset
    if isinstance(raw_dataset, HFDataset):
        dataset = raw_dataset
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
    dataloader = TorchDataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)  # type: ignore
    # Storage for collected data
    texts_list = []
    input_ids_list = []
    hidden_states_dict = {i: [] for i in range(num_layers)}
    mlp_activations_dict = {i: [] for i in range(num_layers)}
    attention_mask_list = []
    # Process samples
    logger.info(f"Using batch size: {batch_size}")
    
    # Process in larger batches for efficiency
    with torch.no_grad():
        # Process samples in batches
        for batch in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):            
            # Process batch
            input_ids_list_, attention_mask_list_, hidden_states_dict_, mlp_activations_dict_ = process_batch(
                batch, model, capture, device, num_layers
            )
            
            # Extend lists with batch results
            texts_list.extend(batch["text"])
            input_ids_list.extend(input_ids_list_)
            attention_mask_list.extend(attention_mask_list_)
            # Extend layer dictionaries
            for layer_idx in range(num_layers):
                if layer_idx in hidden_states_dict:
                    hidden_states_dict[layer_idx].extend(hidden_states_dict_[layer_idx])
                    mlp_activations_dict[layer_idx].extend(mlp_activations_dict_[layer_idx])
                        
            # Save intermediate results periodically
            if len(texts_list) % save_interval == 0 and len(texts_list) > 0:
                logger.info(f"Saving intermediate results at {len(texts_list)} samples...")
                save_dataset(
                    texts_list, input_ids_list, attention_mask_list,
                    hidden_states_dict, mlp_activations_dict,
                    output_dir, num_layers
                )
                
                # Clear accumulated data after saving to avoid re-processing
                texts_list.clear()
                input_ids_list.clear()
                attention_mask_list.clear()
                for layer_idx in range(num_layers):
                    if layer_idx in hidden_states_dict:
                        hidden_states_dict[layer_idx].clear()
                        mlp_activations_dict[layer_idx].clear()
                logger.info("Cleared accumulated data after save")
    
    # Remove hooks
    capture.remove_hooks()
    
    # Save any remaining data as final dataset
    if texts_list:  # Only save if there's remaining data
        logger.info("Saving final batch...")
        save_dataset(
            texts_list, input_ids_list, attention_mask_list,
            hidden_states_dict, mlp_activations_dict,
            output_dir, num_layers
        )
    else:
        logger.info("No remaining data to save.")
    
    # Get final dataset size for metadata by counting the single CSV file
    try:
        csv_file = os.path.join(output_dir, "dataset.csv")
        if os.path.exists(csv_file):
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                total_samples = sum(1 for _ in reader)
        else:
            total_samples = 0
    except Exception as e:
        logger.warning(f"Error counting samples for metadata: {e}")
        total_samples = 0
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "num_samples": total_samples,
        "max_length": max_length,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "batch_size": batch_size,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset generation complete. Total samples in dataset: {total_samples}")


def save_dataset(
    texts_list: List[str],
    input_ids_list: List[np.ndarray],
    attention_mask_list: List[np.ndarray],
    hidden_states_dict: Dict[int, List[np.ndarray]],
    mlp_activations_dict: Dict[int, List[np.ndarray]],
    output_dir: str,
    num_layers: int
):
    """Save dataset using single .npz file for arrays and append to single CSV for metadata."""
    
    if not texts_list:
        logger.warning("No data to save")
        return
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    arrays_dir = os.path.join(output_dir, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)
    
    # Generate unique timestamp for this batch
    timestamp = int(time.time() * 1000000)  # microsecond precision
    
    # Prepare single batch data
    batch_data = {
        'input_ids': np.stack(input_ids_list),
        'attention_mask': np.stack(attention_mask_list),
    }
    logger.info(f"Input IDs shape: {batch_data['input_ids'].shape}")
    # Add layer data to batch
    for layer_idx in range(num_layers):
        if layer_idx in hidden_states_dict:
            batch_data[f'hidden_states_layer_{layer_idx}'] = np.stack(hidden_states_dict[layer_idx])
            batch_data[f'mlp_activations_layer_{layer_idx}'] = np.stack(mlp_activations_dict[layer_idx])
    
    # Save single batch as .npz file
    batch_filename = f"batch_{timestamp:016d}.npz"
    batch_path = os.path.join(arrays_dir, batch_filename)
    np.savez_compressed(batch_path, **batch_data)
    logger.info(f"Saved batch with {len(texts_list)} samples to {batch_filename}")
    # Create CSV rows for all samples in this batch
    csv_rows = []
    for sample_idx in range(len(texts_list)):
        row = {
            "text": texts_list[sample_idx],
            "batch_file": batch_filename,
            "batch_index": sample_idx,
        }
        
        # Add layer columns
        for layer_idx in range(num_layers):
            if layer_idx in hidden_states_dict:
                row[f"layer_{layer_idx}_available"] = True
            else:
                row[f"layer_{layer_idx}_available"] = False
        
        csv_rows.append(row)
    
    # Append to single CSV file
    csv_file = os.path.join(output_dir, "dataset.csv")
    file_exists = os.path.exists(csv_file)
    
    # Determine fieldnames
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
    else:
        return
    
    # Append to CSV file
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist or is empty
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
            logger.info(f"Created new CSV file: {csv_file}")
        
        # Write all rows
        writer.writerows(csv_rows)
    
    logger.info(f"Appended {len(csv_rows)} samples to {csv_file}")

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
    parser.add_argument("--inspect_sample", type=int, default=None,
                       help="Inspect a specific sample by index (useful for debugging)")
    parser.add_argument("--show_stats", action="store_true",
                       help="Show dataset statistics without loading arrays")
    
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
    
    # Handle dataset statistics
    if args.show_stats:
        logger.info(f"Getting dataset statistics from {args.output_dir}")
        stats = get_dataset_stats(args.output_dir)
        if stats:
            logger.info("Dataset Statistics:")
            logger.info(f"  Total samples: {stats['total_samples']:,}")
            logger.info(f"  Total batches: {stats['total_batches']:,}")
            logger.info(f"  Avg samples per batch: {stats['avg_samples_per_batch']:,}")
            logger.info(f"  Metadata size: {stats['metadata_size_mb']:.1f} MB")
            logger.info(f"  Arrays size: {stats['arrays_size_mb']:.1f} MB")
            logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")
            logger.info(f"  Arrays/Metadata ratio: {stats['compression_ratio']}")
        else:
            logger.error("Could not get dataset statistics")
        return
    
    # Handle sample inspection
    if args.inspect_sample is not None:
        logger.info(f"Inspecting sample {args.inspect_sample} from {args.output_dir}")
        sample = get_sample_by_index(args.output_dir, args.inspect_sample)
        if sample:
            logger.info(f"Sample {args.inspect_sample}:")
            logger.info(f"  Text: {sample['text'][:100]}...")
            logger.info(f"  Input IDs shape: {sample['input_ids'].shape}")
            logger.info(f"  Attention mask shape: {sample['attention_mask'].shape}")
            for key in sample.keys():
                if 'hidden_states' in key:
                    logger.info(f"  {key} shape (last token): {sample[key].shape}")
                elif 'mlp_activations' in key:
                    logger.info(f"  {key} shape (last token): {sample[key].shape}")
        else:
            logger.error(f"Could not load sample {args.inspect_sample}")
        return
    
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
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
