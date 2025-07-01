from transformers import OPTConfig,  PretrainedConfig
import os
from typing import Union, Any
from src.configuration_skip import build_skip_config

OPTSkipConnectionConfig: type[OPTConfig] = build_skip_config(OPTConfig, "opt-skip")