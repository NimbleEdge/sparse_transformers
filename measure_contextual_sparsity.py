#!/usr/bin/env python3
"""
Contextual Sparsity Measurement for Standard LLaMA Models

This script measures contextual sparsity patterns in LLaMA models following the 
Deja Vu approach. It analyzes activation patterns for every token prediction
to understand which neurons are consistently important across different contexts.

Reference: Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextualSparsityAnalyzer:
    """Analyzer for measuring contextual sparsity patterns in LLaMA models."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Sparsity tracking
        self.layer_activations = defaultdict(list)  # Layer -> [activations]
        self.token_sparsity = defaultdict(list)     # Layer -> [sparsity_ratios]
        self.heavy_hitters = defaultdict(Counter)   # Layer -> {neuron_id: frequency}
        self.contextual_patterns = defaultdict(list) # Context -> [active_neurons]
        self.sequence_patterns = []                 # [(context_tokens, active_neurons)]
        
        # Analysis parameters
        self.sparsity_thresholds = [0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
        self.context_window = config.get('context_window', 32)
        self.max_sequences = config.get('max_sequences', 1000)
        
        # Hook handles for activation collection
        self.hook_handles = []
        self.current_activations = {}
        
    def register_hooks(self):
        """Register forward hooks to collect MLP activations."""
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                # Store intermediate activations (after gate * up, before down projection)
                if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj'):
                    gate_out = module.act_fn(module.gate_proj(input[0]))
                    up_out = module.up_proj(input[0])
                    intermediate = gate_out * up_out  # Element-wise multiplication
                    self.current_activations[layer_name] = intermediate.detach()
                    
            return hook_fn
        
        # Register hooks for all MLP layers
        for i, layer in enumerate(self.model.model.layers):
            layer_name = f"layer_{i}"
            handle = layer.mlp.register_forward_hook(create_hook(layer_name))
            self.hook_handles.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def compute_token_sparsity(self, activations: torch.Tensor, thresholds: List[float]) -> Dict[float, float]:
        """Compute sparsity ratios at different thresholds for a single token."""
        # activations: [batch_size, seq_len, intermediate_size]
        abs_activations = torch.abs(activations)
        
        # Convert to float32 for quantile computation (quantile requires float32/float64)
        if abs_activations.dtype == torch.float16:
            abs_activations = abs_activations.float()
        
        sparsity_ratios = {}
        for threshold in thresholds:
            # Compute threshold value (percentile of activation magnitudes)
            threshold_value = torch.quantile(abs_activations, threshold, dim=-1, keepdim=True)
            active_mask = abs_activations > threshold_value
            sparsity_ratio = active_mask.float().mean().item()
            sparsity_ratios[threshold] = 1.0 - sparsity_ratio  # Convert to sparsity
            
        return sparsity_ratios
    
    def get_top_k_neurons(self, activations: torch.Tensor, k: int) -> Set[int]:
        """Get top-k most active neurons for a token."""
        # activations: [intermediate_size]
        abs_activations = torch.abs(activations)
        
        # Ensure we can compute topk (usually works with float16, but ensure compatibility)
        if abs_activations.dtype == torch.float16 and not abs_activations.is_cuda:
            abs_activations = abs_activations.float()
            
        _, top_indices = torch.topk(abs_activations, k)
        return set(top_indices.cpu().numpy())
    
    def analyze_contextual_similarity(self, context_tokens: List[int], active_neurons: Set[int]):
        """
        Analyze similarity between current context and previous contexts.
        
        Note: Uses FULL context (all preceding tokens) instead of a sliding window
        to capture complete contextual patterns and long-range dependencies.
        This provides more accurate contextual sparsity analysis at the cost of
        more memory usage and potentially fewer exact context matches.
        """
        # Use full context instead of limiting to context_window
        # This captures complete contextual information for better sparsity analysis
        context_key = tuple(context_tokens)  # Use all available tokens as context
        self.contextual_patterns[context_key].append(active_neurons)
        
        # Store for sequence-level analysis
        self.sequence_patterns.append((context_tokens.copy(), active_neurons.copy()))
    
    def compute_heavy_hitters(self, layer_name: str, active_neurons: Set[int]):
        """Track neurons that are frequently active (heavy hitters)."""
        for neuron_id in active_neurons:
            self.heavy_hitters[layer_name][neuron_id] += 1
    
    def process_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """Process a single sequence and collect sparsity measurements."""
        batch_size, seq_len = input_ids.shape
        results = {
            'sequence_length': seq_len,
            'layer_sparsity': defaultdict(list),
            'token_patterns': [],
            'heavy_hitters_per_layer': defaultdict(set)
        }
        
        self.model.eval()
        with torch.no_grad():
            # Process sequence token by token for autoregressive analysis
            context_tokens = []
            
            for pos in range(1, seq_len):  # Start from position 1 (predicting token 1 from token 0)
                # Current context (up to position pos-1)
                current_input = input_ids[:, :pos]
                current_mask = attention_mask[:, :pos] if attention_mask is not None else None
                
                # Clear previous activations
                self.current_activations.clear()
                
                # Forward pass to collect activations
                if input_ids.is_cuda:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=current_input,
                            attention_mask=current_mask,
                            output_hidden_states=False,
                            use_cache=False
                        )
                else:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=current_input,
                            attention_mask=current_mask,
                            output_hidden_states=False,
                            use_cache=False
                        )
                
                # Analyze activations for each layer
                current_token = input_ids[0, pos].item()
                context_tokens.append(current_token)
                
                token_analysis = {
                    'position': pos,
                    'token_id': current_token,
                    'layer_analysis': {}
                }
                
                for layer_name, activations in self.current_activations.items():
                    # Get activations for the last token position
                    last_token_activations = activations[0, -1, :]  # [intermediate_size]
                    
                    # Compute sparsity ratios
                    sparsity_ratios = self.compute_token_sparsity(
                        last_token_activations.unsqueeze(0).unsqueeze(0), 
                        self.sparsity_thresholds
                    )
                    
                    # Get top-k active neurons (using 10% as active threshold)
                    k = max(1, int(0.1 * last_token_activations.shape[0]))
                    active_neurons = self.get_top_k_neurons(last_token_activations, k)
                    
                    # Track heavy hitters
                    self.compute_heavy_hitters(layer_name, active_neurons)
                    
                    # Store analysis
                    token_analysis['layer_analysis'][layer_name] = {
                        'sparsity_ratios': sparsity_ratios,
                        'active_neurons': list(active_neurons),
                        'activation_stats': {
                            'mean': last_token_activations.mean().item(),
                            'std': last_token_activations.std().item(),
                            'max': last_token_activations.max().item(),
                            'min': last_token_activations.min().item()
                        }
                    }
                    
                    # Store for layer-wise analysis
                    results['layer_sparsity'][layer_name].append(sparsity_ratios)
                    results['heavy_hitters_per_layer'][layer_name].update(active_neurons)
                
                # Analyze contextual patterns
                if len(context_tokens) >= 4:  # Need some context
                    for layer_name, activations in self.current_activations.items():
                        last_token_activations = activations[0, -1, :]
                        k = max(1, int(0.1 * last_token_activations.shape[0]))
                        active_neurons = self.get_top_k_neurons(last_token_activations, k)
                        self.analyze_contextual_similarity(context_tokens, active_neurons)
                
                results['token_patterns'].append(token_analysis)
        
        return results
    
    def compute_contextual_consistency(self) -> Dict[str, float]:
        """Compute consistency metrics for contextual patterns."""
        consistency_metrics = {}
        
        for layer_name in self.heavy_hitters.keys():
            # Heavy hitter concentration
            total_activations = sum(self.heavy_hitters[layer_name].values())
            if total_activations > 0:
                # Top 10% neurons concentration
                sorted_neurons = sorted(self.heavy_hitters[layer_name].items(), 
                                      key=lambda x: x[1], reverse=True)
                top_10_percent = int(0.1 * len(sorted_neurons)) or 1
                top_concentration = sum(count for _, count in sorted_neurons[:top_10_percent])
                consistency_metrics[f'{layer_name}_heavy_hitter_concentration'] = top_concentration / total_activations
        
        # Context similarity metrics
        context_overlap_scores = []
        for context_key, neuron_lists in self.contextual_patterns.items():
            if len(neuron_lists) > 1:
                # Compute pairwise Jaccard similarity
                similarities = []
                for i in range(len(neuron_lists)):
                    for j in range(i + 1, len(neuron_lists)):
                        set1, set2 = neuron_lists[i], neuron_lists[j]
                        if len(set1) > 0 and len(set2) > 0:
                            intersection = len(set1.intersection(set2))
                            union = len(set1.union(set2))
                            similarities.append(intersection / union if union > 0 else 0.0)
                
                if similarities:
                    context_overlap_scores.extend(similarities)
        
        if context_overlap_scores:
            consistency_metrics['mean_context_overlap'] = np.mean(context_overlap_scores)
            consistency_metrics['std_context_overlap'] = np.std(context_overlap_scores)
        
        return consistency_metrics
    
    def save_results(self, save_dir: str):
        """Save analysis results to files."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw data
        results = {
            'heavy_hitters': dict(self.heavy_hitters),
            'contextual_patterns': {str(k): [list(s) for s in v] 
                                  for k, v in self.contextual_patterns.items()},
            'sequence_patterns': [(tokens, list(neurons)) 
                                for tokens, neurons in self.sequence_patterns],
            'consistency_metrics': self.compute_contextual_consistency()
        }
        
        with open(os.path.join(save_dir, 'sparsity_analysis.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary statistics
        summary = {
            'total_sequences_analyzed': len(self.sequence_patterns),
            'total_layers': len(self.heavy_hitters),
            'sparsity_thresholds': self.sparsity_thresholds,
            'context_window': self.context_window,
            'consistency_metrics': results['consistency_metrics']
        }
        
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {save_dir}")
    
    def plot_sparsity_analysis(self, save_dir: str):
        """Generate visualization plots for sparsity analysis."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Sparsity ratios across layers and thresholds
        plt.figure(figsize=(12, 8))
        
        layer_names = sorted(self.heavy_hitters.keys())
        threshold_data = defaultdict(list)
        
        # Aggregate sparsity data across all processed sequences
        for layer_name in layer_names:
            layer_sparsity_data = []
            # This would need to be collected during processing
            # For now, we'll use heavy hitter data as proxy
            total_neurons = 8192  # Typical intermediate size for LLaMA
            active_neurons = len(self.heavy_hitters[layer_name])
            sparsity_proxy = 1.0 - (active_neurons / total_neurons)
            layer_sparsity_data.append(sparsity_proxy)
        
        # Plot heavy hitter distribution
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Heavy hitter concentration per layer
        plt.subplot(2, 2, 1)
        layer_indices = range(len(layer_names))
        heavy_hitter_counts = [len(self.heavy_hitters[layer]) for layer in layer_names]
        
        plt.bar(layer_indices, heavy_hitter_counts)
        plt.xlabel('Layer Index')
        plt.ylabel('Number of Heavy Hitter Neurons')
        plt.title('Heavy Hitter Distribution Across Layers')
        plt.xticks(layer_indices[::4], [f'L{i}' for i in layer_indices[::4]])
        
        # Subplot 2: Top neurons frequency distribution
        plt.subplot(2, 2, 2)
        all_frequencies = []
        for layer_name in layer_names[:5]:  # Show first 5 layers
            frequencies = list(self.heavy_hitters[layer_name].values())
            if frequencies:
                all_frequencies.extend(frequencies)
        
        if all_frequencies:
            plt.hist(all_frequencies, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Activation Frequency')
            plt.ylabel('Number of Neurons')
            plt.title('Neuron Activation Frequency Distribution')
            plt.yscale('log')
        
        # Subplot 3: Context similarity distribution
        plt.subplot(2, 2, 3)
        consistency_metrics = self.compute_contextual_consistency()
        if 'mean_context_overlap' in consistency_metrics:
            overlap_scores = []
            for neuron_lists in self.contextual_patterns.values():
                if len(neuron_lists) > 1:
                    for i in range(len(neuron_lists)):
                        for j in range(i + 1, len(neuron_lists)):
                            set1, set2 = neuron_lists[i], neuron_lists[j]
                            if len(set1) > 0 and len(set2) > 0:
                                intersection = len(set1.intersection(set2))
                                union = len(set1.union(set2))
                                overlap_scores.append(intersection / union if union > 0 else 0.0)
            
            if overlap_scores:
                plt.hist(overlap_scores, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Jaccard Similarity')
                plt.ylabel('Frequency')
                plt.title('Contextual Pattern Similarity')
        
        # Subplot 4: Layer-wise heavy hitter concentration
        plt.subplot(2, 2, 4)
        concentrations = []
        for layer_name in layer_names:
            total_activations = sum(self.heavy_hitters[layer_name].values())
            if total_activations > 0:
                sorted_neurons = sorted(self.heavy_hitters[layer_name].items(), 
                                      key=lambda x: x[1], reverse=True)
                top_10_percent = int(0.1 * len(sorted_neurons)) or 1
                top_concentration = sum(count for _, count in sorted_neurons[:top_10_percent])
                concentrations.append(top_concentration / total_activations)
            else:
                concentrations.append(0.0)
        
        plt.plot(layer_indices, concentrations, 'o-')
        plt.xlabel('Layer Index')
        plt.ylabel('Top 10% Neuron Concentration')
        plt.title('Heavy Hitter Concentration by Layer')
        plt.xticks(layer_indices[::4], [f'L{i}' for i in layer_indices[::4]])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}")


class C4Dataset(Dataset):
    """C4 dataset for contextual sparsity analysis."""
    
    def __init__(self, tokenizer, max_length: int = 512, num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load C4 dataset
        logger.info("Loading C4 dataset...")
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        
        # Process samples
        self.samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            text = sample['text']
            if len(text.strip()) > 50:  # Filter out very short texts
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                if encoding['input_ids'].shape[1] > 10:  # Ensure minimum sequence length
                    self.samples.append({
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                        'text': text[:200] + "..." if len(text) > 200 else text
                    })
        
        logger.info(f"Loaded {len(self.samples)} C4 samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(description="Measure contextual sparsity in LLaMA models")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="HuggingFace model name or path")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of C4 samples to analyze")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (recommend 1 for token-by-token analysis)")
    parser.add_argument("--context_window", type=int, default=32,
                       help="Context window size for pattern analysis (UNUSED: now using full context)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_plots", action="store_true",
                       help="Generate and save analysis plots")
    
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
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True
    )
    
    if device.type != "cuda":
        model = model.to(device)
    
    # Load C4 dataset
    dataset = C4Dataset(tokenizer, args.max_length, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize sparsity analyzer
    config = {
        'context_window': args.context_window,
        'max_sequences': args.num_samples
    }
    analyzer = ContextualSparsityAnalyzer(model, tokenizer, config)
    
    # Register hooks for activation collection
    analyzer.register_hooks()
    
    try:
        # Process dataset
        logger.info("Starting contextual sparsity analysis...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing sequences")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Process sequence
            results = analyzer.process_sequence(input_ids, attention_mask)
            
            # Log progress
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} sequences")
                
                # Intermediate consistency metrics
                consistency = analyzer.compute_contextual_consistency()
                if 'mean_context_overlap' in consistency:
                    logger.info(f"Current mean context overlap: {consistency['mean_context_overlap']:.4f}")
        
        # Save results
        logger.info("Saving analysis results...")
        analyzer.save_results(args.output_dir)
        
        # Generate plots if requested
        if args.save_plots:
            logger.info("Generating visualization plots...")
            analyzer.plot_sparsity_analysis(args.output_dir)
        
        # Print final summary
        consistency_metrics = analyzer.compute_contextual_consistency()
        
        print(f"\n{'='*60}")
        print(f"🎯 CONTEXTUAL SPARSITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Model: {args.model_name}")
        print(f"📝 Sequences analyzed: {len(analyzer.sequence_patterns)}")
        print(f"🧠 Layers analyzed: {len(analyzer.heavy_hitters)}")
        print(f"🔍 Context patterns found: {len(analyzer.contextual_patterns)}")
        
        print(f"\n📈 Consistency Metrics:")
        for metric, value in consistency_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
        
        # Heavy hitter summary
        print(f"\n🎯 Heavy Hitter Analysis:")
        for layer_name in sorted(list(analyzer.heavy_hitters.keys())[:5]):  # Show first 5 layers
            heavy_count = len(analyzer.heavy_hitters[layer_name])
            total_activations = sum(analyzer.heavy_hitters[layer_name].values())
            print(f"   {layer_name}: {heavy_count} heavy hitters, {total_activations} total activations")
        
        print(f"\n✅ Analysis completed! Results saved to: {args.output_dir}")
        print(f"📁 Files generated:")
        print(f"   - sparsity_analysis.pkl (raw data)")
        print(f"   - summary.json (summary statistics)")
        if args.save_plots:
            print(f"   - sparsity_analysis.png (visualization)")
        
    finally:
        # Clean up hooks
        analyzer.remove_hooks()


if __name__ == "__main__":
    main() 