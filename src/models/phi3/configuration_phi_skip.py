from transformers import Phi3Config,  PretrainedConfig
import os
from typing import Union, Any
from src.configuration_skip import build_skip_config

Phi3SkipConnectionConfig: type[Phi3Config] = build_skip_config(Phi3Config, "phi3-skip")
