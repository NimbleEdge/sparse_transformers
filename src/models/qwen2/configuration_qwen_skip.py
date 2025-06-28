from transformers import Qwen2Config
from src.configuration_skip import build_skip_config

Qwen2SkipConnectionConfig = build_skip_config(Qwen2Config, "qwen2-skip")
