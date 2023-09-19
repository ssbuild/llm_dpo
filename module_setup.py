# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 16:22

import os
from deep_training.utils.hf import register_transformer_model,register_transformer_config
from config import *

_model_card = (train_info_args["model_name_or_path"] or train_info_args["config_name"])
_model_card = _model_card.split('/')[-1].lower() if os.path.isdir(_model_card) else str(os.path.dirname(_model_card)).split('/')[-1].lower()

#支持自定义hf中文模型载入
if "baichuan2" in _model_card:
    if "7b" in _model_card:
        from aigc_zoo.model_zoo.baichuan.baichuan2_7b.llm_model import (MyTransformer,PetlArguments, # noqa
                                                                        LoraConfig,PetlModel,
                                                                        PromptArguments,
                                                                        BaichuanConfig,
                                                                        BaichuanTokenizer)
    else:
        from aigc_zoo.model_zoo.baichuan.baichuan2_13b.llm_model import (MyTransformer, PetlArguments, # noqa
                                                                        LoraConfig,PetlModel,
                                                                        PromptArguments,
                                                                        BaichuanConfig,
                                                                        BaichuanTokenizer)

elif "baichuan" in _model_card:
    if "7b" in _model_card:
        from aigc_zoo.model_zoo.baichuan.baichuan_7b.llm_model import (MyTransformer, PetlArguments,  # noqa
                                                                        LoraConfig, PetlModel,
                                                                        PromptArguments,
                                                                        BaichuanConfig,
                                                                        BaichuanTokenizer)
    else:
        from aigc_zoo.model_zoo.baichuan.baichuan_13b.llm_model import (MyTransformer, PetlArguments,  # noqa
                                                                         LoraConfig, PetlModel,
                                                                         PromptArguments,
                                                                         BaichuanConfig,
                                                                         BaichuanTokenizer)

del _model_card

__all__ = [
    "module_setup"
]

def module_setup():
    pass
    # 导入模型
    #register_transformer_config(XverseConfig)
    # register_transformer_model(LlamaForCausalLM, AutoModelForCausalLM)