import argparse
import os 
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from transformers.models.instructblip.modeling_instructblip import InstructBlipAttention

# from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers, prune_gradient, prune_gblm, wanda_pp
from lib.eval import eval_ppl



def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="cuda:1"
    )
    print("printing gpu allocation for all the layers")
    print(model.hf_device_map)
    model.seqlen = 2048

    return model



model = get_llm("/ssd1/haozeng/models/meta-llama/Llama-3.2-1B-Instruct")   # 只加载，不做任何操作
for idx, block in enumerate(model.model.layers):
    weight = block.self_attn.q_proj.weight          # Tensor
    # 如果想确认是否含 NaN/Inf，可加 isfinite 检测
    status = torch.isfinite(weight).all().item()
    print(f"layer {idx} | shape {weight.shape} | finite={status} | first5:", 
          weight.flatten()[:5])