import logging
import os
from typing import Dict,Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers.optimization import get_linear_schedule_with_warmup

from datasets import load_from_disk
import wandb
from tqdm import tqdm

from src.modeling_skip import FastLoRAProjection
from src.configuration_skip import SkipConnectionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorDataset(TorchDataset):
    """Dataset for training predictors from pre-generated data."""
    
    def __init__(self, dataset_dir: str, layer_idx: int, split: str = "train", 
                 max_samples: Optional[int] = None):
        """
        Args:
            dataset_dir: Directory containing layer datasets
            layer_idx: Which layer's predictor to train
            split: 'train' or 'val'
            max_samples: Maximum number of samples to use
        """
        self.layer_idx = layer_idx
        self.layer_dir = os.path.join(dataset_dir, f"layer_{layer_idx}")
        
        # Load the HuggingFace dataset
        if not os.path.exists(self.layer_dir):
            raise ValueError(f"Layer dataset not found at {self.layer_dir}")
            
        self.dataset = load_from_disk(self.layer_dir)
        
        # Split into train/val if needed
        if "train" not in self.dataset:
            # Create train/val split (90/10)
            split_dataset = self.dataset.train_test_split(test_size=0.1, seed=42)
            self.dataset = split_dataset["train"] if split == "train" else split_dataset["test"]
        else:
            self.dataset = self.dataset[split]
        
        # Limit samples if requested
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
        
        logger.info(f"Loaded {len(self.dataset)} samples for layer {layer_idx} ({split})")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "hidden_states": torch.tensor(item["hidden_states"], dtype=torch.float32),
            "mlp_activations": torch.tensor(item["mlp_activations"], dtype=torch.float32)
        }


class LayerwisePredictorTrainer:
    """Trainer for layer-wise predictors."""
    
    def __init__(self, config: SkipConnectionConfig, device: torch.device):
        self.config = config
        self.device = device
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Initialize predictors for each layer
        self.predictors = nn.ModuleList([
            FastLoRAProjection(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                lora_size=config.lora_size
            ).to(device)
            for _ in range(config.num_hidden_layers)
        ])
    
    def compute_loss(self, 
                    layer_idx: int,
                    hidden_states: torch.Tensor,
                    mlp_activations: torch.Tensor,
                    k: int) -> torch.Tensor:
        """Compute predictor loss."""
        # Get predictor scores
        pred_scores = self.predictors[layer_idx](hidden_states)  # [batch_size, intermediate_size]
        
        # Get top-k indices from ground truth
        gt_mask = (mlp_activations > 0).long()
        
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(pred_scores, gt_mask)
        
        return loss
    
    def evaluate_predictor(self, 
                          layer_idx: int,
                          dataloader: DataLoader,
                          max_batches: int = 50) -> Dict[str, float]:
        """Evaluate predictor performance."""
        self.predictors[layer_idx].eval()
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                hidden_states = batch["hidden_states"].to(self.device)
                mlp_activations = batch["mlp_activations"].to(self.device)
                
                # Get predictions
                pred_scores = self.predictors[layer_idx](hidden_states)
                pred_mask = (F.sigmoid(pred_scores) > 0.5)
                
                # Get ground truth
                gt_mask = (mlp_activations > 0)
                tp = (pred_mask * gt_mask).sum().item()
                fp = (pred_mask * (1 - gt_mask)).sum().item()
                fn = ((1 - pred_mask) * gt_mask).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                
                num_batches += hidden_states.shape[0]
        
        self.predictors[layer_idx].train()
        
        if num_batches == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        return {
            "precision": total_precision / num_batches,
            "recall": total_recall / num_batches,
            "f1": total_f1 / num_batches
        }
    
    def train_layer(self, layer_idx: int, train_dataset: PredictorDataset,
                   val_dataset: PredictorDataset, num_epochs: int,
                   batch_size: int, learning_rate: float,
                   k: int, use_wandb: bool = False) -> FastLoRAProjection:
        """Train a single layer's predictor."""
        
        logger.info(f"Training predictor for layer {layer_idx}")
        
        predictor = self.predictors[layer_idx]
        predictor.train()
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps
        )
        
        best_f1 = 0.0
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Layer {layer_idx} - Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                hidden_states = batch["hidden_states"].to(self.device)
                mlp_activations = batch["mlp_activations"].to(self.device)
                
                # Compute loss
                loss = self.compute_loss(layer_idx, hidden_states, mlp_activations, k)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log to wandb
                if use_wandb and global_step % 50 == 0:
                    wandb.log({
                        f"layer_{layer_idx}/train_loss": loss.item(),
                        f"layer_{layer_idx}/learning_rate": scheduler.get_last_lr()[0],
                        "step": global_step
                    })
            
            # Evaluation
            eval_metrics = self.evaluate_predictor(layer_idx, val_loader, k)
            logger.info(f"Layer {layer_idx} - Epoch {epoch+1} - Eval metrics: {eval_metrics}")
            
            if use_wandb:
                wandb.log({
                    f"layer_{layer_idx}/eval_precision": eval_metrics["precision"],
                    f"layer_{layer_idx}/eval_recall": eval_metrics["recall"],
                    f"layer_{layer_idx}/eval_f1": eval_metrics["f1"],
                    "epoch": epoch + 1
                })
            
            # Save best model
            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]
                logger.info(f"Layer {layer_idx} - New best F1: {best_f1:.4f}")
        
        return self.predictors[layer_idx]  # type: ignore
    
    def save_predictors(self, save_dir: str):
        """Save all trained predictors."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each predictor
        for i, predictor in enumerate(self.predictors):
            torch.save(predictor.state_dict(), 
                      os.path.join(save_dir, f"predictor_layer_{i}.pt"))
        
        # Save config
        self.config.save_pretrained(save_dir)
        
        logger.info(f"Saved predictors to {save_dir}")
