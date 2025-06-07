import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictorTrainingLoss(nn.Module):
    """Loss function for training sparsity predictors based on ground truth activations."""
    
    def __init__(self, loss_type: str = "bce", temperature: float = 1.0, alpha: float = 1.0, 
                 confidence_penalty: float = 0.1, focal_gamma: float = 2.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.alpha = alpha
        self.confidence_penalty = confidence_penalty  # Weight for confidence regularization
        self.focal_gamma = focal_gamma  # Gamma parameter for focal loss
        
    def forward(self, predicted_scores: torch.Tensor, ground_truth_activations: torch.Tensor, 
                sparsity_ratio: float) -> torch.Tensor:
        """
        Compute predictor training loss with enhanced binary classification.
        
        Args:
            predicted_scores: [batch_size, intermediate_size] - predictor output scores
            ground_truth_activations: [batch_size, intermediate_size] - actual activations from standard LLaMA
            sparsity_ratio: Target sparsity ratio
        """
        batch_size, intermediate_size = predicted_scores.shape
        
        # Create ground truth binary mask based on top-k activations
        k = max(1, int(intermediate_size * sparsity_ratio))
        
        # Get top-k indices for each batch item
        _, top_k_indices = torch.topk(torch.abs(ground_truth_activations), k, dim=-1)
        
        # Create binary ground truth mask
        ground_truth_mask = torch.zeros_like(ground_truth_activations, dtype=torch.bool)
        ground_truth_mask.scatter_(1, top_k_indices, True)
        ground_truth_target = ground_truth_mask.float()  # Convert to float for loss computation
        
        # Apply temperature scaling and sigmoid to get probabilities
        predicted_probs = torch.sigmoid(predicted_scores / self.temperature)
        
        if self.loss_type == "bce":
            # Enhanced Binary Cross-Entropy with confidence penalty
            bce_loss = F.binary_cross_entropy(predicted_probs, ground_truth_target, reduction='none')
            
            # Confidence penalty: penalize predictions close to 0.5 (uncertain)
            # This encourages predictions to be close to 0 or 1
            confidence_loss = self.confidence_penalty * torch.mean(
                4 * predicted_probs * (1 - predicted_probs)  # Maximum at 0.5, minimum at 0 and 1
            )
            
            loss = torch.mean(bce_loss) + confidence_loss
            
        elif self.loss_type == "focal":
            # Focal loss for hard example mining and confident predictions
            ce_loss = F.binary_cross_entropy(predicted_probs, ground_truth_target, reduction='none')
            
            # Calculate focal weight: (1 - p_t)^gamma where p_t is the prob of correct class
            p_t = predicted_probs * ground_truth_target + (1 - predicted_probs) * (1 - ground_truth_target)
            focal_weight = (1 - p_t) ** self.focal_gamma
            
            focal_loss = focal_weight * ce_loss
            
            # Add confidence penalty
            confidence_loss = self.confidence_penalty * torch.mean(
                4 * predicted_probs * (1 - predicted_probs)
            )
            
            loss = torch.mean(focal_loss) + confidence_loss
            
        elif self.loss_type == "ranking":
            # Enhanced ranking loss with confidence penalty
            active_scores = predicted_scores[ground_truth_mask]
            inactive_scores = predicted_scores[~ground_truth_mask]
            
            if len(active_scores) > 0 and len(inactive_scores) > 0:
                # Sample pairs for efficiency
                n_pairs = min(1000, len(active_scores) * len(inactive_scores))
                active_sample = active_scores[torch.randint(0, len(active_scores), (n_pairs,))]
                inactive_sample = inactive_scores[torch.randint(0, len(inactive_scores), (n_pairs,))]
                
                # Margin ranking loss with larger margin for clearer separation
                ranking_loss = F.margin_ranking_loss(
                    active_sample, inactive_sample, 
                    torch.ones_like(active_sample), margin=2.0  # Increased margin
                )
                
                # Add confidence penalty on probabilities
                confidence_loss = self.confidence_penalty * torch.mean(
                    4 * predicted_probs * (1 - predicted_probs)
                )
                
                loss = ranking_loss + confidence_loss
            else:
                # Fallback to BCE if no active/inactive samples
                loss = F.binary_cross_entropy(predicted_probs, ground_truth_target)
            
        elif self.loss_type == "mse":
            # MSE loss with normalized activations and confidence penalty
            normalized_activations = torch.abs(ground_truth_activations) / (torch.abs(ground_truth_activations).max(dim=-1, keepdim=True)[0] + 1e-8)
            mse_loss = F.mse_loss(predicted_probs, normalized_activations)
            
            # Add confidence penalty
            confidence_loss = self.confidence_penalty * torch.mean(
                4 * predicted_probs * (1 - predicted_probs)
            )
            
            loss = mse_loss + confidence_loss
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return self.alpha * loss