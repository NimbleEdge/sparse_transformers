from transformers import MistralConfig
from src.configuration_skip import build_skip_config

MistralSkipConnectionConfig = build_skip_config(MistralConfig, "mistral-skip")