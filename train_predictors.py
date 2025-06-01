#!/usr/bin/env python3
"""
Training script for sparsity predictors in LlamaSkipConnection models.

This script trains the LoRA-based sparsity predictors to predict which neurons
will be most important based on ground truth activations from standard LLaMA.

The script uses the C4 (Colossal Clean Crawled Corpus) dataset by default, which
is the same dataset used to train LLaMA models. C4 provides high-quality English
text data that matches the training distribution of the original LLaMA models,
making it ideal for training sparsity predictors.

C4 Dataset:
- 750GB of English-language text from Common Crawl
- Cleaned and filtered web content 
- Used to train Google's T5/LaMDA and Meta's LLaMA models
- Available via HuggingFace datasets as 'allenai/c4'

Usage:
    python train_predictors.py \
        --config configs/llama_skip_causal_3b_predictor_training.json \
        --output_dir ./trained_predictors \
        --num_samples 50000 \
        --batch_size 8 \
        --num_epochs 5 \
        --use_wandb
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
from datasets import load_dataset
import wandb
from tqdm import tqdm

from src.models.llama.modelling_llama_skip import LlamaSkipConnectionForCausalLM
from src.models.llama.configuration_llama_skip import LlamaSkipConnectionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for training sparsity predictors."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def load_training_data(dataset_name: str = "allenai/c4", 
                      dataset_config: str = "realnewslike",
                      num_samples: int = 10000,
                      max_length: int = 512) -> Tuple[List[str], List[str]]:
    """Load and prepare training data."""
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
    
    if dataset_name == "allenai/c4":
        # Load C4 dataset with streaming for efficiency
        dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        
        # Extract text samples from streaming dataset
        train_texts = []
        val_texts = []
        
        logger.info("Extracting samples from C4 dataset...")
        for i, sample in enumerate(dataset):
            if i >= num_samples + 1000:  # Extra samples for validation
                break
            
            text = sample['text']
            if len(text.strip()) > 50:  # Filter out very short texts
                if i < num_samples:
                    train_texts.append(text)
                else:
                    val_texts.append(text)
        
        # Ensure we have validation samples
        if not val_texts:
            val_texts = train_texts[-1000:]  # Use last 1000 as validation
            train_texts = train_texts[:-1000]
            
    else:
        # Original logic for other datasets like WikiText
        dataset = load_dataset(dataset_name, dataset_config)
        
        # Filter out empty texts and combine train/validation
        train_texts = [text for text in dataset['train']['text'] if text.strip()]
        val_texts = [text for text in dataset['validation']['text'] if text.strip()]
        
        # Limit number of samples
        train_texts = train_texts[:num_samples]
        val_texts = val_texts[:min(1000, len(val_texts))]
    
    logger.info(f"Loaded {len(train_texts)} training samples, {len(val_texts)} validation samples")
    return train_texts, val_texts


def evaluate_predictor_accuracy(model: LlamaSkipConnectionForCausalLM,
                               dataloader: DataLoader,
                               device: torch.device,
                               max_batches: int = 50) -> Dict[str, float]:
    """Evaluate predictor accuracy against ground truth."""
    model.eval()  # Set to evaluation mode
    
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass to collect predictor scores and ground truth
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate metrics for each layer
            layer_accuracies = []
            layer_precisions = []
            layer_recalls = []
            layer_f1s = []
            
            for layer in model.model.layers:
                if hasattr(layer, 'training_mode') and layer.training_mode:
                    # Get last hidden states for this evaluation
                    hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
                    if hidden_states is not None:
                        # Get predictor scores
                        hidden_reshaped = hidden_states.view(-1, hidden_states.shape[-1])
                        pred_scores = layer.mlp_lora_proj(hidden_reshaped)
                        
                        # Get ground truth activations
                        gt_activations = layer.get_ground_truth_activations(hidden_reshaped)
                        
                        # Create binary masks
                        k = int(layer.sparsity * pred_scores.shape[-1])
                        _, gt_indices = torch.topk(torch.abs(gt_activations), k, dim=-1)
                        _, pred_indices = torch.topk(pred_scores, k, dim=-1)
                        
                        # Calculate metrics
                        gt_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                        pred_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                        gt_mask.scatter_(1, gt_indices, True)
                        pred_mask.scatter_(1, pred_indices, True)
                        
                        # Accuracy
                        correct = (gt_mask == pred_mask).float().mean()
                        layer_accuracies.append(correct.item())
                        
                        # Precision, Recall, F1
                        tp = (gt_mask & pred_mask).sum().float()
                        fp = (~gt_mask & pred_mask).sum().float()
                        fn = (gt_mask & ~pred_mask).sum().float()
                        
                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)
                        
                        layer_precisions.append(precision.item())
                        layer_recalls.append(recall.item())
                        layer_f1s.append(f1.item())
            
            if layer_accuracies:
                total_accuracy += sum(layer_accuracies) / len(layer_accuracies)
                total_precision += sum(layer_precisions) / len(layer_precisions)
                total_recall += sum(layer_recalls) / len(layer_recalls)
                total_f1 += sum(layer_f1s) / len(layer_f1s)
                num_batches += 1
    
    model.train()  # Switch back to training mode
    
    if num_batches == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return {
        "accuracy": total_accuracy / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches
    }


def train_predictors(
    model: LlamaSkipConnectionForCausalLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: str,
    eval_steps: int = 500,
    save_steps: int = 1000,
    use_wandb: bool = False
) -> None:
    """Train the sparsity predictors using PyTorch's built-in training mode."""
    
    # Setup training mode using PyTorch's built-in methods
    model.train()  # Enable training mode
    model.freeze_non_predictor_parameters()
    
    # Setup optimizer
    predictor_params = model.get_predictor_parameters()
    optimizer = torch.optim.AdamW(predictor_params, lr=learning_rate, weight_decay=0.01)
    
    # Setup scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    best_f1 = 0.0
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Number of predictor parameters: {sum(p.numel() for p in predictor_params)}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_predictor_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_predictor_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluation
            if global_step % eval_steps == 0:
                logger.info(f"Evaluating at step {global_step}")
                try:
                    eval_metrics = evaluate_predictor_accuracy(model, val_dataloader, device)
                    logger.info(f"Evaluation metrics: {eval_metrics}")
                    
                    if use_wandb:
                        wandb.log({
                            "eval/accuracy": eval_metrics["accuracy"],
                            "eval/precision": eval_metrics["precision"],
                            "eval/recall": eval_metrics["recall"],
                            "eval/f1": eval_metrics["f1"],
                            "step": global_step
                        })
                    
                    # Save best model
                    if eval_metrics["f1"] > best_f1:
                        best_f1 = eval_metrics["f1"]
                        best_model_path = os.path.join(save_dir, "best_predictors")
                        try:
                            model.save_pretrained(best_model_path)
                            logger.info(f"Saved best model with F1: {best_f1:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to save best model: {e}")
                    
                    # Model is already back in training mode from evaluate_predictor_accuracy
                except Exception as e:
                    logger.warning(f"Evaluation failed at step {global_step}: {e}")
                    model.train()  # Ensure we're back in training mode
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint at step {global_step}")
            
            # Log training metrics
            if use_wandb and global_step % 100 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "step": global_step
                })
        
        # End of epoch logging
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "epoch": epoch + 1
            })
    
    # Final save
    final_model_path = os.path.join(save_dir, "final_predictors")
    model.save_pretrained(final_model_path)
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train sparsity predictors for LlamaSkipConnection")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained models")
    parser.add_argument("--dataset", type=str, default="allenai/c4", help="Dataset name (default: allenai/c4)")
    parser.add_argument("--dataset_config", type=str, default="realnewslike", 
                       help="Dataset configuration (default: realnewslike for C4)")
    parser.add_argument("--num_samples", type=int, default=10000, 
                       help="Number of training samples (default: 10000)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-skip-predictors", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="llama-skip-predictors", help="W&B entity name")
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
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"predictor-training-{int(time.time())}"
        )
    
    # Load model configuration
    config = LlamaSkipConnectionConfig.from_json_file(args.config)
    checkpoint = config._name_or_path
    
    # Register custom models
    AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
    AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    model = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=config)
    model = model.to(device)
    
    # Load training data
    train_texts, val_texts = load_training_data(
        args.dataset, args.dataset_config, args.num_samples, args.max_length
    )
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, args.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, args.max_length)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train predictors
    train_predictors(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb
    )
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 