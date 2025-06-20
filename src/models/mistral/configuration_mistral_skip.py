from transformers import MistralConfig,  PretrainedConfig
import os
from typing import Union, Any
from src.configuration_skip import build_skip_config

MistralSkipConnectionConfig: type[MistralConfig] = build_skip_config(MistralConfig, "mistral-skip")