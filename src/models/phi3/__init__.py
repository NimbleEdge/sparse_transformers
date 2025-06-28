from . import configuration_phi_skip
from . import modelling_phi_skip

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_phi_skip import Phi3SkipConnectionConfig
from .modelling_phi_skip import Phi3SkipConnectionForCausalLM
AutoConfig.register("phi3-skip", Phi3SkipConnectionConfig)
AutoModelForCausalLM.register(Phi3SkipConnectionConfig, Phi3SkipConnectionForCausalLM)

from .activation_capture_phi import ActivationCapturePhi3
from src.activation_capture import register_activation_capture
register_activation_capture('phi3', ActivationCapturePhi3)

__all__ = [configuration_phi_skip, modelling_phi_skip]