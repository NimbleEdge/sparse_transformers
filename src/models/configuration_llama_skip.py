from transformers import LlamaConfig
from optimum.utils import NormalizedTextConfig, MistralDummyPastKeyValuesGenerator, DummyTextInputGenerator
import os
from typing import Union
from optimum.exporters.onnx.config import TextDecoderWithPositionIdsOnnxConfig


class LlamaSkipConnectionConfig(LlamaConfig):
    model_type = "llama-skip"

    def __init__(self, sparsity : float, **kwargs):
        self._sparsity = sparsity
        super().__init__(**kwargs)
    
    @property
    def sparsity(self):
        return self._sparsity
    
    @sparsity.setter
    def sparsity(self, value):
        self._sparsity = value

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)


class LlamaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14  # Llama now uses F.scaled_dot_product_attention by default for torch>=2.1.1.

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig