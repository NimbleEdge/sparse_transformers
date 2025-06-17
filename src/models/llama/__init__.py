from . import configuration_llama_skip
from . import modelling_llama_skip

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_llama_skip import LlamaSkipConnectionConfig
from .modelling_llama_skip import LlamaSkipConnectionForCausalLM
AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

__all__ = [configuration_llama_skip, modelling_llama_skip]