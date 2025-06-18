# limitations under the License.
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from dataclasses import dataclass
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    ModelOutput
)
from transformers.models.llama.modeling_llama import(
     LlamaRotaryEmbedding,
     LlamaMLP, LlamaAttention, KwargsForCausalLM, FlashAttentionKwargs
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.processing_utils import Unpack
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import logging, is_torch_flex_attn_available
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

# Import C++ extensions
from sparse_transformers import (
    sparse_mlp_forward,
    WeightCache,
    approx_topk_threshold
)

from src.models.llama.configuration_llama_skip import LlamaSkipConnectionConfig
from src.modeling_utils import (
    FastLoRAProjection, BaseModelOutputWithPastAndPredictorLoss
)
from src.modeling_skip import SkipMLP, SkipDecoderLayer, build_skip_connection_model, build_skip_connection_model_for_causal_lm

logger = logging.get_logger(__name__)

                     
class LlamaSkipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, sparsity: float, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.sparsity = sparsity
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Initialize mask but defer WeightCache creation until post_init
        self.init_mask = torch.ones(intermediate_size, dtype=torch.bool)
        self.init_mask[int(intermediate_size * sparsity):] = 0
        
        self.weight_cache = None

        # Register buffers - start with reasonable size and ensure they can be resized
        self.register_buffer('down_proj_buffer', torch.zeros(1, hidden_size, requires_grad=False))
        self.register_buffer('combined_proj_buffer', torch.zeros(1, 2 * int(intermediate_size * sparsity), requires_grad=False))

    def initialize_weight_cache(self):
        """Tie weights after weights are loaded (called from post_init)."""
        if self.weight_cache is None:
            # Create and initialize weight cache
            self.weight_cache = WeightCache(   
                self.init_mask,
                self.hidden_size,
                self.gate_proj.weight,
                self.up_proj.weight, 
                self.down_proj.weight
            )

    def to(self, *args, **kwargs):
        # Move buffers to same device as model when .to() is called
        result = super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device')
        if device:
            self.down_proj_buffer = self.down_proj_buffer.to(device)
            self.combined_proj_buffer = self.combined_proj_buffer.to(device)
            if hasattr(self, 'init_mask'):
                self.init_mask = self.init_mask.to(device)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sparse_mlp_forward(
            x.detach(), 
            self.weight_cache.get_concat_weight(),
            self.weight_cache.get_active_down_weight(),
            self.down_proj_buffer,
            self.combined_proj_buffer,
            "silu"
        )
        return out

class LlamaSkipRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaSkipRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
        
class LlamaSkipDecoderLayer(SkipDecoderLayer):
    def _init_components(self, config, layer_idx):
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = LlamaSkipRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaSkipRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _set_mlp_train(self, config: LlamaSkipConnectionConfig):
        self.mlp = LlamaMLP(config)

    def _set_mlp_inference(self, config: LlamaSkipConnectionConfig):
        self.mlp = SkipMLP(
            config.hidden_size,
            config.intermediate_size,
            config.sparsity,
            config.mlp_bias,
        )



class LlamaSkipPreTrainedModel(PreTrainedModel):
    config_class = LlamaSkipConnectionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaSkipDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

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
        elif isinstance(module, LlamaSkipRMSNorm):
            module.weight.data.fill_(1.0)


LlamaSkipConnectionModelBase: type[LlamaSkipPreTrainedModel] = build_skip_connection_model(LlamaSkipPreTrainedModel)

class LlamaSkipConnectionModel(LlamaSkipConnectionModelBase):
    def _init_components(self, config: LlamaSkipConnectionConfig):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaSkipDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaSkipRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
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
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
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
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    
LlamaSkipConnectionForCausalLM: type[LlamaSkipPreTrainedModel] = \
    build_skip_connection_model_for_causal_lm(LlamaSkipPreTrainedModel, LlamaSkipConnectionModel)


__all__ = [LlamaSkipConnectionForCausalLM]