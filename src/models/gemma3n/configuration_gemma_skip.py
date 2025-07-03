from transformers import Gemma3nTextConfig
from src.configuration_skip import build_skip_config

Gemma3nSkipConnectionConfig = build_skip_config(Gemma3nTextConfig, "gemma3n-skip")