from typing import List, Tuple, Union, Literal, Optional
import os

import torch
import transformers

from lm_eval.models.huggingface import HFLM

# Can potentially move model initialization and weight loading into this class as well 
class SparseLM(HFLM):

    # May add things to this later
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained,
            *args,
            **kwargs
        )
        
    # May need to override _model_generate in a similar way to this - are those the only two endpoints for task requests?
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # Reset cache between model calls for independent queries
        self.model.reset_cache()
        
        return super()._loglikelihood_tokens(
            requests=requests,
            disable_tqdm=disable_tqdm,
            override_bs=override_bs
        )