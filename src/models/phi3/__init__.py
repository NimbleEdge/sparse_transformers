from . import configuration_phi_skip
from . import modelling_phi_skip

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_phi_skip import Phi3SkipConnectionConfig
from .modelling_phi_skip import Phi3SkipConnectionForCausalLM
AutoConfig.register("phi3-skip", Phi3SkipConnectionConfig)
AutoModelForCausalLM.register(Phi3SkipConnectionConfig, Phi3SkipConnectionForCausalLM)

__all__ = [configuration_phi_skip, modelling_phi_skip]