# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

train_info_models = {
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


# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}

train_model_config = train_info_models['opt-350m']