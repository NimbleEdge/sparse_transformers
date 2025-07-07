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
"""

import argparse
import csv
import glob
import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

from src.activation_capture import ActivationCaptureTraining

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
            with open(csv_file, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                total_samples = sum(1 for _ in reader)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")

        # Estimate storage sizes
        metadata_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
        arrays_size = sum(
            os.path.getsize(bf) for bf in batch_files if os.path.exists(bf)
        )

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
            "compression_ratio": f"{arrays_size / max(1, metadata_size):.1f}x",
        }

    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        return None


def process_batch(
    tokenized_batch: Dict[str, torch.Tensor],
    model,
    device: torch.device,
    num_layers: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Process a batch of texts and return last token activations for each sample."""

    # Move to device
    input_ids = tokenized_batch["input_ids"].to(device)
    # Clear previous captures and GPU cache
    model.activation_capture.clear_captures()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids=input_ids.squeeze(0))

    # Pre-allocate arrays for efficiency
    hidden_states_dict = {}
    mlp_activations_dict = {}
    for layer_idx in range(num_layers):
        hidden_state = model.activation_capture.get_hidden_states(layer_idx)[0]
        hidden_states_dict[layer_idx] = (
            hidden_state.view(-1, hidden_state.shape[-1])
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        mlp_activation = model.activation_capture.get_gate_activations(layer_idx)
        mlp_activations_dict[layer_idx] = (
            mlp_activation[0]
            .view(-1, mlp_activation.shape[-1])
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # Clear GPU tensors from capture to free memory
    model.activation_capture.clear_captures()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return hidden_states_dict, mlp_activations_dict


def generate_dataset(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    output_dir: str,
    device: torch.device,
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
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    model.activation_capture = ActivationCaptureTraining(model)
    model.activation_capture.register_hooks()

    # Get model dimensions
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_layers = len(model.activation_capture.get_layers())

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(
            dataset_name, dataset_config, split="train", streaming=True
        )
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=10000, seed=42)

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

    # Process in larger batches for efficiency
    with torch.no_grad():
        # Process samples in batches
        for idx, element in enumerate(
            tqdm(dataloader, desc="Processing batches", total=max_samples)
        ):
            # Process batch
            hidden_states_dict, mlp_activations_dict = process_batch(
                element, model, device, num_layers
            )

            save_dataset(
                idx, hidden_states_dict, mlp_activations_dict, output_dir, num_layers
            )

            # Clear accumulated data after saving to avoid re-processing
            hidden_states_dict.clear()
            mlp_activations_dict.clear()
            logger.info("Cleared accumulated data after save")

    # Remove hooks
    model.activation_capture.remove_hooks()

    # Get final dataset size for metadata by counting the single CSV file
    try:
        csv_file = os.path.join(output_dir, "dataset.csv")
        if os.path.exists(csv_file):
            with open(csv_file, "r", newline="") as f:
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
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Dataset generation complete. Total samples in dataset: {total_samples}"
    )


def save_dataset(
    idx: int,
    hidden_states_dict: Dict[int, np.ndarray],
    mlp_activations_dict: Dict[int, np.ndarray],
    output_dir: str,
    num_layers: int,
):
    """Save dataset using single .npz file for arrays and append to single CSV for metadata."""

    if not hidden_states_dict:
        logger.warning("No data to save")
        return

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    arrays_dir = os.path.join(output_dir, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)

    # Prepare single batch data
    batch_data = {}
    csv_rows = []
    chunk_size = 500
    # Add layer data to batch
    for start_idx in range(0, hidden_states_dict[0].shape[0], chunk_size):
        for layer_idx in range(num_layers):
            batch_data[f"hidden_states_layer_{layer_idx}"] = hidden_states_dict[
                layer_idx
            ][start_idx : start_idx + chunk_size]
            batch_data[f"mlp_activations_layer_{layer_idx}"] = mlp_activations_dict[
                layer_idx
            ][start_idx : start_idx + chunk_size]
        # Save single batch as .npz file
        batch_filename = f"batch_{idx}_{start_idx}.npz"
        batch_path = os.path.join(arrays_dir, batch_filename)
        np.savez_compressed(batch_path, **batch_data)
        print(
            f"Saved {batch_data[f'hidden_states_layer_{0}'].shape[0]} samples with {start_idx} start index to {batch_filename}"
        )
        # Create CSV rows for all samples in this batch
        for sample_idx in range(batch_data[f"hidden_states_layer_{0}"].shape[0]):
            row = {
                "batch_file": batch_filename,
                "batch_index": sample_idx,
            }
            csv_rows.append(row)

    # Append to single CSV file
    csv_file = os.path.join(output_dir, "dataset.csv")
    file_exists = os.path.exists(csv_file)

    # Append to CSV file
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())

        # Write header only if file doesn't exist or is empty
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
            logger.info(f"Created new CSV file: {csv_file}")

        # Write all rows
        writer.writerows(csv_rows)

    logger.info(f"Appended {len(csv_rows)} samples to {csv_file}")


def main():
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
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum number of samples to process",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show dataset statistics without loading arrays",
    )

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

    # Generate dataset
    generate_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        device=device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
