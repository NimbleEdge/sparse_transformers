from transformers import LlamaConfig
from optimum.utils import NormalizedTextConfig, MistralDummyPastKeyValuesGenerator, DummyTextInputGenerator
from optimum.exporters.onnx.config import TextDecoderWithPositionIdsOnnxConfig


class LlamaSkipConnectionConfig(LlamaConfig):
    model_type = "llama-skip"

    def __init__(self, sparsity=0.2, **kwargs):
        self.sparsity = sparsity
        super().__init__(**kwargs)


class LlamaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # Llama now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig