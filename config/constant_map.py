# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]


MODELS_MAP = {
    'bloom-560m': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    },
    'bloom-1b7': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    },
    'opt-125m': {
        'model_type': 'opt',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-125m',
    },

    'opt-350m': {
        'model_type': 'opt',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
    },

    'llama-7b-hf': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    },

    'Qwen-7B-Chat': {
        'model_type': 'qwen',
        'model_name_or_path': '/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat',
    },

    'Baichuan2-7B-Chat': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-7B-Chat',
    },

    'Baichuan2-13B-Chat': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan2/Baichuan2-13B-Chat',
    },

    'Baichuan-13B-Chat': {
        'model_type': 'baichuan',
        'model_name_or_path': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat',
    },

    'XVERSE-13B-Chat': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat',
    },

    'internlm-chat-7b': {
        'model_type': 'internlm',
        'model_name_or_path': '/data/nlp/pre_models/torch/internlm/internlm-chat-7b',
    },
    'internlm-chat-7b-8k': {
        'model_type': 'internlm',
        'model_name_or_path': '/data/nlp/pre_models/torch/internlm/internlm-chat-7b-8k',
    },
}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

train_model_config = train_info_models['opt-350m']