# @Time    : 2023/4/19 23:02
# @Author  : tk
# @FileName: data_utils
from collections import OrderedDict

from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from deep_training.nlp.models.petl import PetlArguments
from deep_training.nlp.models.petl.prompt import PromptArguments
from transformers import HfArgumentParser
from data_factory.data_helper_loader import (NN_DataHelper_Base,
                                             NN_DataHelper_baichuan,
                                             NN_DataHelper_chatglm,
                                             NN_DataHelper_chatglm2,
                                             NN_DataHelper_bloom,
                                             NN_DataHelper_internlm,
                                             NN_DataHelper_gpt2,
                                             NN_DataHelper_llama,
                                             NN_DataHelper_moss,
                                             NN_DataHelper_moss_plugin,
                                             NN_DataHelper_opt,
                                             NN_DataHelper_qwen,
                                             NN_DataHelper_tiger,
                                             NN_DataHelper_xverse,
                                             NN_DataHelper_rwkv,
                                             NN_DataHelper_openbuddy)

from config import *
from module_setup import global_model_card


def _find_data_helper():
    data_helper_mapper = OrderedDict({
        "baichuan": NN_DataHelper_baichuan,
        "chatglm2": NN_DataHelper_chatglm2,
        "chatglm": NN_DataHelper_chatglm,
        "xverse": NN_DataHelper_xverse,
        "qwen": NN_DataHelper_qwen,
        "gpt2": NN_DataHelper_gpt2,
        "llama": NN_DataHelper_llama,
        "internlm": NN_DataHelper_internlm,
        "opt": NN_DataHelper_opt,
        "bloom": NN_DataHelper_bloom,
        "tiger": NN_DataHelper_tiger,
        "moss": NN_DataHelper_moss,
        "moss_plugin": NN_DataHelper_moss_plugin,
        "rwkv": NN_DataHelper_rwkv,
        "openbuddy": NN_DataHelper_openbuddy,
        "default": NN_DataHelper_Base,
    })
    NN_DataHelper = None
    for k in data_helper_mapper:
        if k in global_model_card:
            if k == "moss" and "plugin" in global_model_card:
                k = "moss_plugin"
            NN_DataHelper = data_helper_mapper[ k ]
            break
    return NN_DataHelper


NN_DataHelper = _find_data_helper()
if NN_DataHelper is None:
    NN_DataHelper = NN_DataHelper_Base
    raise ValueError(f"{global_model_card} for data_helper is not implemented ")

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
        model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)
    elif global_args["trainer_backend"] == "ac":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args, allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args, allow_extra_keys=True, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})

    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
