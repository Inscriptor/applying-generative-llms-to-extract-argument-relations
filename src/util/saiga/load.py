import sys

import torch

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import PeftConfig, PeftModel


def load_saiga(
    model_name: str,
    load_in_8bit: bool = True,
    torch_dtype: str = None,
    use_flash_attention_2: bool = False
):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)

    config = PeftConfig.from_pretrained(model_name)
    base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)

    if torch_dtype is not None:
        torch_dtype = getattr(torch, torch_dtype)
    else:
        torch_dtype = base_model_config.torch_dtype

    if device == "cuda":
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                quantization_config=quantization_config,
                use_flash_attention_2=use_flash_attention_2,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                use_flash_attention_2=use_flash_attention_2,
            )
        model = PeftModel.from_pretrained(model, model_name, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            device_map={"": device}
        )

    model.eval()
    return model, tokenizer, generation_config
