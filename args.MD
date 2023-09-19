### 多机多卡训练
```text
 例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


### lora finetuning
```text
       global_args = {
          "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
          "num_layers": -1, # 是否使用骨干网络的全部层数 ， -1 表示全层, 否则只用只用N层
          "num_layers_key":  "num_layers",
      }
         lora_info_args = {
               'with_lora': True,  # 是否启用lora模块
               'r': 8,
               'target_modules': ['query_key_value'],
               'target_dtype': None,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
               'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        }
```


### int高效训练方式
   lora int8   
```text
       global_args = {
          "load_in_8bit": True, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
          "num_layers": -1, # 是否使用骨干网络的全部层数 ， -1 表示全层, 否则只用只用N层
          "num_layers_key":  "num_layers",
      }
         lora_info_args = {
               'with_lora': True,  # 是否启用lora模块
               'r': 8,
               'target_modules': ['query_key_value'],
               'target_dtype': None,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
               'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        }
```

## 是否开启deepspeed
    修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed


## 精度训练
  precision='16'  # 自行修改 "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
