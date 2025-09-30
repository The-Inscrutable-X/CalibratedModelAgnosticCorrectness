from dataclasses import dataclass
import json
import os
import sys
from typing import TypedDict
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel
import torch

model_loadstring_dict = {
    #Huggingface Models

    "Qwen2.5-72B-Instruct": "Qwen",
    "Qwen2.5-32B-Instruct": "Qwen", 
    "Qwen2.5-7B": "Qwen", 
    "Qwen2.5-7B-Instruct": "Qwen", 
    "gemma-2b": "google", 
    "gpt2-large": "openai-community", 
    "Llama-2-7b-hf": "meta-llama", 
    "Llama-2-7b-chat-hf": "meta-llama",
    "Llama-2-7b-chat-hf-ct-oe": "calibration-tuning",
    "Llama-2-7b-hf-ct-oe": "calibration-tuning",
    "Llama-2-7b-chat-hf-ct-choice": "calibration-tuning",
    "Llama-2-7b-hf-ct-choice": "calibration-tuning",
    "Meta-Llama-3-8B-Instruct": "meta-llama", 
    "Meta-Llama-3-8B": "meta-llama", 
    "Mistral-7B-v0.3": "mistralai", 
    "Meta-Llama-3-70B-Instruct": "meta-llama",
    "gemma-3-12b-it": "google",
    "gemma-3-4b-it": "google",
    "gemma-3-1b-it": "google",
    "gemma-3-27b-it": "google",

    # Qwen3 models
    "Qwen3-0.6B": "Qwen",
    "Qwen3-0.6B-Base": "Qwen",
    "Qwen3-1.7B": "Qwen", 
    "Qwen3-1.7B-Base": "Qwen",
    "Qwen3-4B": "Qwen",
    "Qwen3-4B-Base": "Qwen", 
    "Qwen3-8B": "Qwen",
    "Qwen3-8B-Base": "Qwen",
    "Qwen3-14B": "Qwen",
    "Qwen3-14B-Base": "Qwen",
    "Qwen3-32B": "Qwen",
    "Qwen3-30B-A3B": "Qwen",
    "Qwen3-30B-A3B-Base": "Qwen",
    "Qwen3-235B-A22B": "Qwen",

    # Added per request (sub-70B without [X])
    "Llama-3.1-8B-Instruct": "meta-llama",       # include; under 70B
    "Qwen2.5-3B-Instruct": "Qwen",               # include; under 70B
    "gemma-3-4b": "google",                      # base variants to complement -it
    "gemma-3-12b": "google",
    "gemma-3-27b": "google",

    # Phi family (canonical HF IDs for <70B text LMs)
    "Phi-3-mini-4k-instruct": "microsoft",       # ~3.8B
    "Phi-3-small-8k-instruct": "microsoft",      # ~7B
    "Phi-3-medium-4k-instruct": "microsoft",      # ~14B
    "Phi-3-medium-128k-instruct": "microsoft",      # ~14B
    "phi-4": "microsoft",                        # ~14.7B (text LM)

    # Additional models
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai",
    "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": "hugging-quants",
}

class ModelInfo(TypedDict):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

def get_basemodel_loadstring(engine):
    if "+" in engine:
        base_model_name = engine.split("+")[0].split("_")[0]
    else:
        base_model_name = engine.split("_")[0]
    loadstring = model_loadstring_dict[base_model_name] + "/" +  base_model_name
    return loadstring

def load_model(engine, checkpoints_dir = None, device_map = "auto", full_32_precision=False, \
               brainfloat=False, verbose=False, load_in_8bit=False, load_in_4bit=False, lora_model=False, trust_remote_code=True, manual_precision=True) -> ModelInfo:
    """Can handle many types of models."""

    loadstring = get_basemodel_loadstring(engine)

    # Determine torch_dtype based on manual_precision logic
    if manual_precision:
        if full_32_precision:
            torch_dtype = torch.float32
        elif brainfloat:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = "auto"
    print(f"Using {torch_dtype=} during model loading")
    
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:  # This should usually not be used for small model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  
        )

    if engine.endswith("quantized_model"):  # FULLLY SAVED MODEL
        tokenizer = AutoTokenizer.from_pretrained(loadstring, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(loadstring, device_map="auto", torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        print("Base model loaded, now replacing with saved state dict.")
        devices_mapper = {}
        for name, module in model.named_parameters():
          devices_mapper[name] = module.dtype
        loaded_state_dict = torch.load(os.path.join(checkpoints_dir, engine+".pt"))
        model.load_state_dict(loaded_state_dict)
        for key, param in model.named_parameters():
            param.data = param.data.to(devices_mapper[key])
    elif engine.endswith("lora_model") or lora_model:  
        tokenizer = AutoTokenizer.from_pretrained(loadstring, trust_remote_code=trust_remote_code)
        model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(checkpoints_dir, engine), device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        print(f"Model loaded: {type(model)}")
    elif engine.endswith("fullft_model"):  
        with open(os.path.join(checkpoints_dir, engine, "config.json"), "r") as f:
            info = f.read()
            configs = json.loads(info)
            tokenizer_loadstring = configs.get("_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_loadstring, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(os.path.join(checkpoints_dir, engine), device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    else:
        print(engine)
        loadstring = model_loadstring_dict[engine] + "/" +  engine
        model = AutoModelForCausalLM.from_pretrained(loadstring, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(loadstring, trust_remote_code=trust_remote_code)

    print("Model loaded of type:", type(model))
    unique_dtypes = set()
    for name, param in model.named_parameters():
        unique_dtypes.add(param.dtype)
        verbose and print(f"{name}: {param.device}")
        if device_map != None or not ("cpu" in device_map):
            assert param.device != "cpu"
    print("Activation Dtypes for model:", unique_dtypes)
    print(f"Model loaded from {checkpoints_dir=} {engine=}")

    return {"model": model, "tokenizer": tokenizer}

def load_tokenizer(engine, verbose=False):
    loadstring = get_basemodel_loadstring(engine)
    tokenizer = AutoTokenizer.from_pretrained(loadstring)
    return tokenizer