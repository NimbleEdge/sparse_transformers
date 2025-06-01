from transformers import PreTrainedModel
from dataclasses import dataclass
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    ModelOutput
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    add_start_docstrings,
    replace_return_docstrings,
    logging
)
from transformers.models.llama.modeling_llama import(
     LlamaRMSNorm, LlamaRotaryEmbedding, LlamaDecoderLayer,FlashAttentionKwargs, KwargsForCausalLM,
     LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, LLAMA_START_DOCSTRING, LlamaMLP
)
from typing import List, Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.processing_utils import Unpack

from torch import nn
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
import torch.utils.cpp_extension

# Import C++ extensions
import torch
from sparse_transformers import (
    sparse_mlp_forward,
    WeightCache,
    approx_topk_threshold
)

from src.models.llama.configuration_llama_skip import LlamaSkipConnectionConfig

logger = logging.get_logger(__name__)

@dataclass
class BaseModelOutputWithPastAndPredictorLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None    


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

class FastLoRAProjection(nn.Module):
    def __init__(self, hidden_size, intermediate_size, lora_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.lora_size = lora_size
        self.down = nn.Linear(hidden_size, lora_size, bias=False)
        self.up = nn.Linear(lora_size, intermediate_size, bias=False)
        
        # Pre-allocate buffers on CPU initially
        self.register_buffer('intermediate', torch.empty(1, lora_size))
        self.register_buffer('output', torch.empty(1, intermediate_size))
    
    def to(self, *args, **kwargs):
        # Move mask to same device as model when .to() is called
        device = args[0] if args else kwargs.get('device')
        if device:
            self.intermediate = self.intermediate.to(device)
            self.output = self.output.to(device)
        return super().to(*args, **kwargs)
    
    def _resize_buffers(self, batch_size: int, dtype: torch.dtype):
        if self.intermediate.size(0) != batch_size:
            self.intermediate.resize_(batch_size, self.lora_size)
            self.intermediate = self.intermediate.to(dtype=dtype)
            self.output.resize_(batch_size, self.intermediate_size)
            self.output = self.output.to(dtype=dtype)
   
    def forward(self, x):
        batch_size = x.size(0)
        
        # Check if gradients are required (training mode)
        if x.requires_grad or any(p.requires_grad for p in self.parameters()):
            # Use regular matrix multiplication for gradient computation
            intermediate = torch.mm(x, self.down.weight.t())
            output = torch.mm(intermediate, self.up.weight.t())
            return output
        else:
            # Use optimized in-place operations for inference
            self._resize_buffers(batch_size, x.dtype)
            torch.mm(x, self.down.weight.t(), out=self.intermediate)
            torch.mm(self.intermediate, self.up.weight.t(), out=self.output)
            return self.output

class LlamaSkipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, sparsity: float, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.sparsity = sparsity
        self.init_mask = torch.ones(intermediate_size, dtype=torch.bool, device=self.gate_proj.weight.device)
        self.init_mask[int(intermediate_size * sparsity):] = 0
        
        # Create and initialize weight cache (always use optimized version)
        self.weight_cache = WeightCache(   
            self.init_mask,
            hidden_size,
            self.gate_proj.weight.detach(),
            self.up_proj.weight.detach(), 
            self.down_proj.weight.detach()
        )

        # Register buffers - start with reasonable size and ensure they can be resized
        self.register_buffer('down_proj_buffer', torch.zeros(1, hidden_size, requires_grad=False))
        self.register_buffer('combined_proj_buffer', torch.zeros(1, 2 * int(intermediate_size * sparsity), requires_grad=False))

    def to(self, *args, **kwargs):
        # Move buffers to same device as model when .to() is called
        device = args[0] if args else kwargs.get('device')
        if device:
            self.down_proj_buffer = self.down_proj_buffer.to(device)
            self.combined_proj_buffer = self.combined_proj_buffer.to(device)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sparse_mlp_forward(
            x.detach(), 
            self.weight_cache.get_concat_weight(),
            self.weight_cache.get_active_down_weight(),
            self.down_proj_buffer,
            self.combined_proj_buffer,
            "silu"
        )

class LlamaSkipDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaSkipConnectionConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.sparsity = config.sparsity
        
        # Initialize mask with proper dtype
        self.register_buffer('mlp_mask', torch.zeros(
            config.intermediate_size,
            dtype=torch.bool
        ).contiguous())

        # Create LoRA projection for sparsity prediction
        self.lora_size = int(config.intermediate_size * 0.04)
        self.mlp_lora_proj = FastLoRAProjection(
            config.hidden_size, 
            config.intermediate_size,
            self.lora_size
        )
        
        # Check if this is a training configuration
        self.is_training_config = getattr(config, 'training', False)
        
        # Only initialize predictor training components if explicitly enabled
        if self.is_training_config:
            # Standard MLP for ground truth collection during training
            self.mlp = LlamaMLP(config)
            
            # Loss function for predictor training
            self.predictor_loss_fn = PredictorTrainingLoss(
                loss_type=getattr(config, 'predictor_loss_type', 'bce'),
                temperature=getattr(config, 'predictor_temperature', 1.0),
                alpha=getattr(config, 'predictor_loss_alpha', 1.0),
                confidence_penalty=getattr(config, 'predictor_confidence_penalty', 0.1),
                focal_gamma=getattr(config, 'predictor_focal_gamma', 2.0)
            )
        else:
            self.mlp = LlamaSkipMLP(
                config.hidden_size,
                config.intermediate_size,
                config.sparsity,
                config.mlp_bias,
            )
            # Simply use the weight cache from the MLP
            self.weight_cache = self.mlp.weight_cache

    def get_ground_truth_activations(self, hidden_states: torch.Tensor) -> torch.Tensor:            
        # Compute standard MLP intermediate activations
        gate_proj = self.mlp.gate_proj(hidden_states)
        
        # Apply SiLU activation to gate projection
        gate_activated = F.silu(gate_proj)
        
        return gate_activated

    def compute_predictor_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute loss for training the sparsity predictor."""
        # Get predictor scores
        hidden_states_reshaped = hidden_states.view(-1, hidden_states.shape[-1])
        predicted_scores = self.mlp_lora_proj(hidden_states_reshaped)
        
        # Get ground truth activations
        ground_truth_activations = self.get_ground_truth_activations(hidden_states_reshaped)
        
        # Compute predictor loss
        loss = self.predictor_loss_fn(predicted_scores, ground_truth_activations, self.sparsity)
        
        return loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Reshape hidden states for batch processing
        hidden_states_reshaped = hidden_states.view(-1, hidden_states.shape[-1])

        if not self.training:  # Use PyTorch's built-in training flag
                # 1. LoRA projection to get importance scores
            lora_proj_scores = self.mlp_lora_proj(hidden_states_reshaped)

            # 2. Ultra-fast sparsity-based threshold using C++ Count-Min Sketch operator
            batch_size, intermediate_size = lora_proj_scores.shape
            k = max(1, int(self.sparsity * intermediate_size))  # Number of neurons to activate
            
            # Use optimized C++ Count-Min Sketch operator for threshold computation
            threshold = approx_topk_threshold(lora_proj_scores, k)
            
            # 3. Binary mask creation
            binary_mask = (lora_proj_scores >= threshold).bool()
            
            # Normalize 2D mask to 1D by taking union across batch dimension
            self.weight_cache.update_active_weights(binary_mask.any(dim=0))  # [batch_size, intermediate_size] → [intermediate_size]
          
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_reshaped)

        if self.training:
            predictor_loss = self.compute_predictor_loss(hidden_states_reshaped)
        else:
            predictor_loss = None
        
        hidden_states = hidden_states.view(residual.shape)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs, predictor_loss

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaSkipPreTrainedModel(PreTrainedModel):
    config_class = LlamaSkipConnectionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaSkipDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



class LlamaSkipConnectionModel(LlamaSkipPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaSkipDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_predictor_parameters(self):
        """Get parameters of all predictor networks for optimization."""
        predictor_params = []
        for layer in self.layers:
            predictor_params.extend(layer.mlp_lora_proj.parameters())
        return predictor_params
    
    def freeze_non_predictor_parameters(self):
        """Freeze all parameters except predictor networks."""
        # Freeze main model parameters
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.rotary_emb.parameters():
            param.requires_grad = False
            
        # Freeze layer parameters except predictors
        for layer in self.layers:
            # Freeze attention parameters
            for param in layer.self_attn.parameters():
                param.requires_grad = False
            for param in layer.input_layernorm.parameters():
                param.requires_grad = False
            for param in layer.post_attention_layernorm.parameters():
                param.requires_grad = False
                
            # Freeze standard MLP parameters (used for ground truth) - only if it exists
            if hasattr(layer, 'standard_mlp'):
                for param in layer.standard_mlp.parameters():
                    param.requires_grad = False
                
            # Freeze sparse MLP parameters
            for param in layer.mlp.parameters():
                param.requires_grad = False
                
            # Keep predictor parameters trainable
            for param in layer.mlp_lora_proj.parameters():
                param.requires_grad = True

    def unfreeze_all_parameters(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPastAndPredictorLoss]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_predictor_losses = []  # Collect predictor losses

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs, predictor_loss = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:

                layer_outputs, predictor_loss = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Collect predictor loss if available
            if predictor_loss is not None:
                all_predictor_losses.append(predictor_loss)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute total predictor loss
        total_predictor_loss = None
        if all_predictor_losses:
            total_predictor_loss = torch.stack(all_predictor_losses).mean()

        return BaseModelOutputWithPastAndPredictorLoss(
            loss=total_predictor_loss,
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

class LlamaSkipConnectionForCausalLM(LlamaSkipPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [
        "model.layers.*.mlp.combined_proj_buffer",
        "model.layers.*.mlp.down_proj_buffer",
        "model.layers.*.mlp_lora_proj.down.weight",
        "model.layers.*.mlp_lora_proj.intermediate",
        "model.layers.*.mlp_lora_proj.output", 
        "model.layers.*.mlp_lora_proj.up.weight",
        "model.layers.*.mlp_mask",
        "model.layers.*.standard_mlp.gate_proj.weight",
        "model.layers.*.standard_mlp.up_proj.weight",
        "model.layers.*.standard_mlp.down_proj.weight"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaSkipConnectionModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model
    
    def get_predictor_parameters(self):
        """Get parameters of all predictor networks for optimization."""
        return self.model.get_predictor_parameters()

    def freeze_non_predictor_parameters(self):
        """Freeze all parameters except predictor networks."""
        # Freeze LM head
        for param in self.lm_head.parameters():
            param.requires_grad = False
        
        # Freeze model parameters except predictors
        self.model.freeze_non_predictor_parameters()
    
    def unfreeze_all_parameters(self):
        """Unfreeze all model parameters."""
        self.model.unfreeze_all_parameters()
        for param in self.lm_head.parameters():
            param.requires_grad = True

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        total_loss = None        
        if labels is not None:
            # Compute language modeling loss
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            
            # Combine with predictor loss if in training mode
            if outputs.loss is not None:
                # Weight the predictor loss (can be configured)
                predictor_weight = getattr(self.config, 'predictor_loss_weight', 0.1)
                total_loss = loss + predictor_weight * outputs.loss
            else:
                total_loss = loss
        elif outputs.loss is not None:
            # If we're in training mode with predictor loss but no labels, use predictor loss as main loss
            total_loss = outputs.loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if total_loss is not None:
                return (total_loss,) + output
            elif loss is not None:
                return (loss,) + output
            elif outputs.loss is not None:
                return (outputs.loss,) + output
            else:
                return output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [LlamaSkipConnectionForCausalLM, LlamaSkipMLP, FastLoRAProjection]