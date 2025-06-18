from transformers import Qwen2Config,  PretrainedConfig
import os
from typing import Union, Any
from src.configuration_skip import build_skip_config

Qwen2SkipConnectionConfig = build_skip_config(Qwen2Config, "llama-skip")
