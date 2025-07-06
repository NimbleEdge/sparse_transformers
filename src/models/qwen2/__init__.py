from . import configuration_qwen_skip
from . import modelling_qwen_skip

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_qwen_skip import Qwen2SkipConnectionConfig
from .modelling_qwen_skip import Qwen2SkipConnectionForCausalLM
AutoConfig.register("qwen2-skip", Qwen2SkipConnectionConfig)
AutoModelForCausalLM.register(Qwen2SkipConnectionConfig, Qwen2SkipConnectionForCausalLM)

__all__ = [configuration_qwen_skip, modelling_qwen_skip]