from . import configuration_gemma_skip
from . import modelling_gemma_skip

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_gemma_skip import Gemma3nSkipConnectionConfig
from .modelling_gemma_skip import Gemma3nSkipConnectionForCausalLM
AutoConfig.register("gemma3n-skip", Gemma3nSkipConnectionConfig)
AutoModelForCausalLM.register(Gemma3nSkipConnectionConfig, Gemma3nSkipConnectionForCausalLM)

from src.activation_capture import register_activation_capture, ActivationCaptureDefault
register_activation_capture('gemma3n', ActivationCaptureDefault)

__all__ = [configuration_gemma_skip, modelling_gemma_skip]