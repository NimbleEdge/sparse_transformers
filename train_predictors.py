#!/usr/bin/env python3
"""
Training script for sparsity predictors using pre-generated datasets.

This script trains the LoRA-based sparsity predictors to predict which neurons
will be most important based on pre-generated ground truth data.

Usage:
    python train_predictors_from_dataset.py \
        --config configs/llama_skip_causal_3b_predictor_training.json \
        --dataset_dir ./predictor_training_data \
        --output_dir ./trained_predictors \
        --batch_size 256 \
        --num_epochs 3 \
        --use_wandb
"""

import argparse
import logging
import time

import torch

import wandb

from transformers import AutoConfig
from transformers.trainer_utils import set_seed
from src.predictor_trainer import LayerwisePredictorTrainer, PredictorDataset


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(description="Train sparsity predictors from pre-generated datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing layer datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained models")
    parser.add_argument("--layers", type=str, default="all", 
                       help="Which layers to train (all, or comma-separated indices)")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--k", type=int, default=None, help="Top-k neurons to predict (default: use config)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples per layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-skip-predictors", help="W&B project name")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load config
    config = AutoConfig.from_pretrained(args.config)
    
    # Determine k
    k = args.k if args.k is not None else int(config.intermediate_size * config.top_k_fraction)
    logger.info(f"Using k={k} (top-k neurons to predict)")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"predictor-training-{int(time.time())}"
        )
    
    # Initialize trainer
    trainer = LayerwisePredictorTrainer(config, device)
    
    # Determine which layers to train
    if args.layers == "all":
        layer_indices = list(range(config.num_hidden_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    
    # Train each layer
    for layer_idx in layer_indices:
        try:
            # Load datasets
            train_dataset = PredictorDataset(args.dataset_dir, layer_idx, "train", args.max_samples)
            val_dataset = PredictorDataset(args.dataset_dir, layer_idx, "val", args.max_samples)
            
            # Train predictor
            trainer.train_layer(
                layer_idx=layer_idx,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                k=k,
                use_wandb=args.use_wandb
            )
            
        except Exception as e:
            logger.error(f"Failed to train layer {layer_idx}: {e}")
            continue
    
    # Save all predictors
    trainer.save_predictors(args.output_dir)
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 