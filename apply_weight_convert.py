import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    LlavaConfig,
)

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
from lite_llama.executor.weight_convert import (
    convert_llavallama_hf_to_litellama,
    convert_llama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

checkpoints_dir = "/path/llm_weights/llava-v1.5-7b"

if "llava" in checkpoints_dir.lower():
    model = (
        LlavaForConditionalGeneration.from_pretrained(  # LlavaForConditionalGeneration
            checkpoints_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoints_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")

hf_sd = model.state_dict()

# for name, parameters in hf_sd.items():
#     print(name, parameters.shape)

if "qwen2" in checkpoints_dir.lower():
    llm_config = AutoConfig.from_pretrained(checkpoints_dir)
    num_layers = llm_config.num_hidden_layers
    print("num_layers: ", num_layers)
    convert_qwen2_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

elif "llama" in checkpoints_dir.lower():
    llm_config = AutoConfig.from_pretrained(checkpoints_dir)
    num_layers = llm_config.num_hidden_layers
    print("num_layers: ", num_layers)
    convert_llama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)

elif "llava" in checkpoints_dir.lower():
    llava_config = LlavaConfig.from_pretrained(checkpoints_dir)
    num_layers = llava_config.text_config.num_hidden_layers
    print("num_layers: ", num_layers)
    convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers)
else:
    print("Error! Unsupported model type!")