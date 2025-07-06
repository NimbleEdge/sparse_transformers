# limitations under the License.
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_flex_attn_available

from sparse_transformers import WeightCache, sparse_mlp_forward
from src.activation_capture import ActivationCaptureDefault

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

logger = logging.get_logger(__name__)


class FastLoRAProjection(nn.Module):
    def __init__(self, hidden_size, intermediate_size, lora_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.lora_size = lora_size
        # Force creation of linear layers with actual tensors (not meta tensors)
        self.down = nn.Linear(hidden_size, lora_size, bias=False)
        self.up = nn.Linear(lora_size, intermediate_size, bias=False)

    def _fix_unloaded_weights(self):
        out = self.to_empty(device="cpu")
        self._init_weights()
        return out

    def _init_weights(self):
        with torch.no_grad():
            torch.nn.init.xavier_normal_(self.down.weight)
            torch.nn.init.xavier_normal_(self.up.weight)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))
                     
class SkipMLP(nn.Module):
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
        
        self.weight_cache : Optional[WeightCache] = None

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
            self.weight_cache.get_concat_weight(),  # type: ignore
            self.weight_cache.get_active_down_weight(),  # type: ignore
            self.down_proj_buffer,
            self.combined_proj_buffer,
            "silu"
        )
        return out



class SkipDecoderLayer(ABC, GradientCheckpointingLayer):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.sparsity = config.sparsity

        self._init_components(config, layer_idx)

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
            self._set_mlp_train(config)
        else:
            self._set_mlp_inference(config)

    @abstractmethod
    def _init_components(self, config, layer_idx):
        pass

    @abstractmethod
    def _set_mlp_train(self, config):
        pass

    @abstractmethod
    def _set_mlp_inference(self, config):
        pass

    @property
    def weight_cache(self):
        """Dynamically access the weight cache from the MLP."""
        if hasattr(self.mlp, 'weight_cache') and isinstance(self.mlp, SkipMLP):
            return self.mlp.weight_cache

    
    def _compute_binary_mask(self, hidden_states):
        lora_proj_scores = self.mlp_lora_proj(hidden_states.view(-1, hidden_states.shape[-1]))
        binary_mask = (lora_proj_scores >= lora_proj_scores.mean() + 2 * lora_proj_scores.std()).bool()
        self.weight_cache.update_active_weights(binary_mask.any(dim=0))  # type: ignore
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # type: ignore
        if not self.training:  # Use PyTorch's built-in training flag
            self._compute_binary_mask(hidden_states)
          
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
        )  # type: ignore
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # type: ignore
        hidden_states = self.mlp(hidden_states)  # type: ignore
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    

def build_skip_connection_model(pretrained_model_class: type[PreTrainedModel]) -> type[PreTrainedModel]:
    class SkipConnectionModel(ABC, pretrained_model_class):
        def __init__(self, config: PretrainedConfig):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            self._init_components(config)
            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()
        
        @abstractmethod
        def _init_components(self, config: PretrainedConfig):
            pass

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
            for param in self.parameters():
                param.requires_grad = False

            for layer in self.layers:
                # Keep predictor parameters trainable
                for param in layer.mlp_lora_proj.parameters():
                    param.requires_grad = True

        def unfreeze_all_parameters(self):
            """Unfreeze all model parameters."""
            for param in self.parameters():
                param.requires_grad = True

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
        ) -> BaseModelOutputWithPast:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
            if not isinstance(past_key_values, (type(None), Cache)):
                raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )  # type: ignore

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0) # type: ignore

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions # type: ignore
            )  # type: ignore

            hidden_states = inputs_embeds
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)  # type: ignore

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            for decoder_layer in self.layers[: self.config.num_hidden_layers]: # type: ignore
                if output_hidden_states:
                    all_hidden_states += (hidden_states,) # type: ignore

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )  # type: ignore

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)  # type: ignore

            hidden_states = self.norm(hidden_states)  # type: ignore

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,  # type: ignore
                attentions=all_self_attns,
            )
        
        @abstractmethod
        def _update_causal_mask(
            self,
            attention_mask: Union[torch.Tensor, "BlockMask"], # type: ignore    
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool = False,
        ):
            pass
    return SkipConnectionModel


def build_skip_connection_model_for_causal_lm(pretrained_model_class: type[PreTrainedModel], base_model_class: type[PreTrainedModel]):
    ACTIVATION_CAPTURE = ActivationCaptureDefault

    class SkipConnectionModelForCausalLM(pretrained_model_class, GenerationMixin):
        _tied_weights_keys = ["lm_head.weight"]
        _tp_plan = {"lm_head": "colwise_rep"}
        _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
        _keys_to_ignore_on_load_missing = [
            "model.layers.*.mlp.combined_proj_buffer",
            "model.layers.*.mlp.down_proj_buffer",
            "model.layers.*.mlp.init_mask",
            "model.layers.*.mlp.weight_cache",
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
            self.model = base_model_class(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if config.capture_activations is not None:
                self.activation_capture = ACTIVATION_CAPTURE(self)

            # Initialize weights and apply final processing
            self.post_init()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            out = super(SkipConnectionModelForCausalLM, cls).from_pretrained(*args, **kwargs)
            for module in out.modules():
                if any(hasattr(p, 'is_meta') and p.is_meta for p in module.parameters()) and \
                        hasattr(module, '_fix_unloaded_weights'):
                    module = module._fix_unloaded_weights()  # type: ignore
            return out
            
        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        def set_decoder(self, decoder):
            self.model = decoder

        def get_decoder(self):
            return self.model

        def get_predictor_parameters(self):
            """Get parameters of all predictor networks for optimization."""
            return self.model.get_predictor_parameters()  # type: ignore

        def freeze_non_predictor_parameters(self):
            """Freeze all parameters except predictor networks."""
            # Freeze LM head
            for param in self.lm_head.parameters():
                param.requires_grad = False
            
            # Freeze model parameters except predictors
            self.model.freeze_non_predictor_parameters()  # type: ignore

        def reset_cache(self):
            """Reset cache of all layers."""
            for layer in self.model.layers:  # type: ignore
                layer.mlp.weight_cache = None  # type: ignore
                layer.mlp.initialize_weight_cache()  # type: ignore

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
        ) -> CausalLMOutputWithPast:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

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

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs: BaseModelOutputWithPast = self.model(
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
            )  # type: ignore

            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :]) # type: ignore

            loss = None
            if labels is not None:
                # Compute language modeling loss
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    return SkipConnectionModelForCausalLM