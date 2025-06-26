import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import tracemalloc
import psutil
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict
import random
from typing import List, Dict
from copy import deepcopy

# 获得每个block的输入

# CausalResnetBlock1D 块列表
CAUSAL_RESNET_BLOCKS = (
    # up + mid + down = 1 + 12 + 1 =14
	["decoder.estimator.up_blocks.0.0.0"] + 
	[f"decoder.estimator.mid_blocks.{i}.0" for i in range(12)] +
    ["decoder.estimator.down_blocks.0.0"] 
)
features_by_layer_resnet = defaultdict(list)

def get_hook_resnet(layer_name):
    def hook(module, input, output):
        # 对于 ResnetBlock1D 的 forward，input 是 (x, mask, time_emb)
        x, mask, time_emb = input
        features_by_layer_resnet[layer_name].append({
            "x": x.detach().cpu(),
            "mask": mask.detach().cpu(),
            "time_emb": time_emb.detach().cpu()
        })
    return hook

def register_hooks_for_resnet_blocks(model):
    handles = []
    for name, module in model.flow.named_modules():
        if name in CAUSAL_RESNET_BLOCKS:
            handle = module.register_forward_hook(get_hook_resnet(name))
            handles.append(handle)
    return handles

def get_input_sets_resnet(model, inputs):
    features_by_layer_resnet.clear()  # 清空之前缓存
    
    handles = register_hooks_for_resnet_blocks(model)

    for _ in cosyvoice.inference_zero_shot(*inputs, stream=False):
        break  # 只跑一个step收集特征

    for h in handles:
        h.remove()
    
    collected = {}
    for name in CAUSAL_RESNET_BLOCKS:
        # print("name: ",name)
        # print("features_by_layer_resnet:",features_by_layer_resnet)
        if name in features_by_layer_resnet:
            collected[name] = features_by_layer_resnet[name][0]

    
    activated = [name for name in CAUSAL_RESNET_BLOCKS if name in features_by_layer_resnet]
    missed = [name for name in CAUSAL_RESNET_BLOCKS if name not in features_by_layer_resnet]
    # print(f"CAUSAL_RESNET_BLOCKS Total layers: {len(CAUSAL_RESNET_BLOCKS)}")
    # print(f"CAUSAL_RESNET_BLOCKS Activated layers: {len(activated)}")
    # print(f"CAUSAL_RESNET_BLOCKS Missed layers: {len(missed)}")

    
    return collected 


# BasicTransformerBlock 块列表
BASIC_TRANSFORMER_BLOCKS = (
    # up + mid + down = 4 + 48 + 4 =56
	[f"decoder.estimator.up_blocks.0.1.{i}" for i in range(4)] +
	[f"decoder.estimator.mid_blocks.{i}.1.{j}" for i in range(12) for j in range(4)] +
    [f"decoder.estimator.down_blocks.0.1.{i}" for i in range(4)]
)
features_by_layer = defaultdict(list)

def get_hook(layer_name):
    def hook(module, input, output):
        if input and isinstance(input[0], torch.Tensor):
            features_by_layer[layer_name].append(input[0].detach().cpu())
    return hook

def register_flow_hooks(model):
    handles = []

    for name, module in model.flow.named_modules():
        if name in BASIC_TRANSFORMER_BLOCKS:       
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)

    return handles

def get_input_sets(model, inputs):
    features_by_layer.clear()  # 清空之前缓存
    
    handles = register_flow_hooks(model)

    for _ in cosyvoice.inference_zero_shot(*inputs, stream=False):
        break  # 只跑一个step收集特征

    for h in handles:
        h.remove()
    
    collected = {}
    for name in BASIC_TRANSFORMER_BLOCKS:
        if name in features_by_layer:
            collected[name] = features_by_layer[name][0]

    
    activated = [name for name in BASIC_TRANSFORMER_BLOCKS if name in features_by_layer]
    missed = [name for name in BASIC_TRANSFORMER_BLOCKS if name not in features_by_layer]
    print(f"BASIC_TRANSFORMER_BLOCKS Total layers: {len(BASIC_TRANSFORMER_BLOCKS)}")
    print(f"BASIC_TRANSFORMER_BLOCKS Activated layers: {len(activated)}")
    print(f"BASIC_TRANSFORMER_BLOCKS Missed layers: {len(missed)}")

    
    return collected 



# 获得每个block内部每层的激活值
activation_cache = {}
def register_input_hooks(block: nn.Module):
    """
    Register hooks to capture input activations for all Linear layers in the block.
    """
    for name, submodule in block.named_modules():
        if isinstance(submodule, nn.Linear):  # 你可以扩展到其他层
            def hook_fn(mod, inp, out, module_name=name):
                # 缓存当前层的输入 (inp 是 tuple)
                if module_name not in activation_cache:
                    activation_cache[module_name] = []
                activation_cache[module_name].append(inp[0].detach())
            submodule.register_forward_hook(hook_fn)

def get_signal(name: str) -> List[torch.Tensor]:
    """
    Get recorded input activations for a given layer name.
    """
    return activation_cache.get(name, [])

def zero_ratio(tensor):
    num_zero = (tensor == 0).sum().item()
    total_elements = tensor.numel()
    return num_zero / total_elements

# ========== Algorithm 2: RGS Score Computation ==========
def compute_rgs_score(block, inps, alpha) -> Dict[str, torch.Tensor]:
    sq_grad = {name: torch.zeros_like(param) for name, param in block.named_parameters()}

    activation_cache.clear()
    register_input_hooks(block)  # 注册 hook

    for inp in inps:
        device = next(block.parameters()).device
        x = inp['x'].to(device)
        mask = inp['mask'].to(device)
        time_emb = inp['time_emb'].to(device)
        loss = torch.norm(block(x, mask, time_emb))

        loss.backward(retain_graph=True)

        with torch.no_grad():
            for name, param in block.named_parameters():
                if param.grad is not None:
                    sq_grad[name] += param.grad.detach() ** 2
        block.zero_grad()

    inps_len = len(inps)
    for key in sq_grad:
        sq_grad[key] = torch.sqrt(sq_grad[key] / inps_len)

    score_dict = {}
    # 遍历当前块中所有可学习的参数
    #print("activation_cache: ",activation_cache)
    for name, param in block.named_parameters():
        cache_name = name.rsplit(".", 1)[0]
        if param.requires_grad and name.endswith('weight') and cache_name in activation_cache:
            
            layer_inps = torch.stack(get_signal(cache_name))
            norm_term = layer_inps.norm(p=2, dim=[0,1])
            norm_term = norm_term.unsqueeze(0)
            score = param.data.abs() * (alpha * sq_grad[name] + norm_term)
            # 把权重的剪枝打分保存起来，后续用于排序和剪枝
            score_dict[name] = score

    return score_dict

# ========== Algorithm 1: Wanda++ Pruning ==========
def wanda_plus_plus_pruning(model, input_sets, alpha, K):
    """
    Args:
        model: Transformer with decoder blocks
        input_sets: List[List[Tensor]] — X^l for each decoder block l
        alpha: scaling factor for score
        K: RO optimization rounds
    Returns:
        pruned model
    """
 
    for name, block in model.named_modules():
        if name not in CAUSAL_RESNET_BLOCKS:
            continue
    
        X_l = input_sets[name] 

        # Step 1: RGS loss & gradient collection
        rgs_score = compute_rgs_score(block, X_l, alpha)
        for k in range(K):
            # 复制未剪枝的 block 作为 teacher（冻结参数）
            block_orig = deepcopy(block)
            # Step 2: Select RO samples (you can do random sampling here) ，eg select 2 samples
            X_hat_l = random.sample(X_l, min(len(X_l), 2))  

            # Step 3: Pruning by RGS
            for name, param in block.named_parameters():
                if name in rgs_score:
                    # Prune lowest X% scores (e.g. 30%)
                    score = rgs_score[name]
                    threshold = torch.quantile(score, 0.3)
                    mask = (score > threshold).float()
                    param.data *= mask  # Zero out pruned weights

            # Step 4: Regional Optimization (fine-tune with small batch)            
            block_orig.eval()
            for p in block_orig.parameters():
                p.requires_grad = False

            # 启用当前剪枝后的 block（作为 student）
            block.train()
            optimizer = torch.optim.RMSprop(block.parameters(), lr=3e-7)

            for x_m in X_hat_l:
                optimizer.zero_grad()
                
                device = next(block.parameters()).device
                x = x_m['x'].to(device)
                mask = x_m['mask'].to(device)
                time_emb = x_m['time_emb'].to(device)
                
                with torch.no_grad():
                    y_teacher = block_orig(x, mask, time_emb)
                y_student = block(x, mask, time_emb)
                loss = torch.nn.functional.mse_loss(y_student, y_teacher)
                loss.backward()
                optimizer.step() 

        # Final pruning after RO
        final_score = compute_rgs_score(block, X_l, alpha)
        #print("final_score: ",final_score)
        for name, param in block.named_parameters():
            if name in final_score:
                score = final_score[name]
                threshold = torch.quantile(score, 0.3)
                mask = (score > threshold).float()
                #print("origin param.data: ", param.data)
                print("mask ratio: ", zero_ratio(mask))
                param.data *= mask
                #print("param.data:", param.data)

    return model


def list_all_layer_names(model):
    layer_names = []
    for name, module in model.named_modules():
        layer_names.append(name)
    return layer_names

def print_block_inputs_shape(block_input_sets):
    print(f"Total blocks: {len(block_input_sets)}")
    for block_name, inputs in block_input_sets.items():
        print(f"\nBlock: {block_name}")
        print(f"  Number of samples: {len(inputs)}")  
        
    
        sample_input = inputs[0]
        
        if isinstance(sample_input, (torch.Tensor, np.ndarray)):
            print(f"  Input shape per sample: {sample_input.shape}")
        elif isinstance(sample_input, dict):
            print("  Input is a dictionary with keys:", sample_input.keys())
            # 可以进一步检查字典中的 tensor 形状
            for k, v in sample_input.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    print(f"    {k}: shape={v.shape}")
        else:
            print(f"  Input type: {type(sample_input)}")


def count_nonzero_parameters(model):
    return sum((p != 0).sum().item() for p in model.parameters())


# 加载模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
model = cosyvoice.model
flow_blocks = model.flow



#准备输入数据
texts = [
    "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    "在他讲述那个荒诞故事的过程中，他突然停下来，因为他自己也被逗笑了",
    "今天的会议让我受益匪浅。",
    "你听说过那个奇怪的传闻吗？",
    "她的声音就像春天的阳光一样温暖。"
]  # 至少 128 条文本
contexts = [
    "希望你以后能够做的比我还好呦。",
    "用四川话说这句话",
    "我觉得他有点激动。",
    "轻声细语的感觉。",
    "像在朗诵一样平稳。"
]
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)


# 初始化一个字典，键是 block 名，值是该 block 的所有样本输入
block_input_sets = {}
for i in range(len(texts)):
    text = texts[i]
    context = contexts[i]
    collected = get_input_sets_resnet(model, (text, context, prompt_speech_16k))  # collected 是字典
    
    # 遍历每个 block 的输入，并存储到 block_input_sets
    for block_name, block_input in collected.items():
        if block_name not in block_input_sets:
            block_input_sets[block_name] = []  # 初始化该 block 的输入列表
        block_input_sets[block_name].append(block_input)  


print("原模型非零参数数量：", count_nonzero_parameters(flow_blocks))

pruned_model = wanda_plus_plus_pruning(flow_blocks, block_input_sets, 0.5, 5)

print("剪枝后非零参数数量：", count_nonzero_parameters(pruned_model))# print(pruned_model)
print("pruned_model is flow_blocks:", pruned_model is flow_blocks) 
