from transformers import  PretrainedConfig
import os
from typing import Union, Any, Type



def build_skip_config(base_config_class: type[PretrainedConfig], model_type_name: str) -> type[PretrainedConfig]:
    class SkipConnectionConfig(base_config_class):
        model_type: str = model_type_name
        has_no_defaults_at_init: bool = True

        def __init__(self, 
                    sparsity: float,
                    predictor_loss_type: str = "bce",
                    predictor_temperature: float = 1.0,
                    predictor_loss_alpha: float = 1.0,
                    predictor_loss_weight: float = 0.1,
                    use_optimized_weight_cache: bool = True,
                    capture_activations: str = None,
                    **kwargs):
            self._sparsity = sparsity
            self.predictor_loss_type = predictor_loss_type
            self.predictor_temperature = predictor_temperature
            self.predictor_loss_alpha = predictor_loss_alpha
            self.predictor_loss_weight = predictor_loss_weight
            self.use_optimized_weight_cache = use_optimized_weight_cache
            self.capture_activations = capture_activations
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

        @classmethod
        def from_dict(cls, config_dict: dict[str, Any], **kwargs):
            if "name_or_path" in kwargs and ("name_or_path" in config_dict or "_name_or_path" in config_dict):
                del kwargs["name_or_path"]
            return super().from_dict(config_dict, **kwargs)
    return SkipConnectionConfig
