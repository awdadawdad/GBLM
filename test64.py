import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.flow.decoder import CausalResnetBlock1D
from matcha.models.components.transformer import BasicTransformerBlock
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



# CausalResnetBlock1D 块列表
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
        if isinstance(module, CausalResnetBlock1D):
            handle = module.register_forward_hook(get_hook_resnet(name))
            handles.append(handle)
    return handles

def collect_transformer_inputs(model):
    inputs_dict = {}
    for name, module in model.flow.named_modules():
        if isinstance(module, BasicTransformerBlock):
            if hasattr(module, "_input_cache") and module._input_cache:
                inputs_dict[name] = module._input_cache[0]
    return inputs_dict

# 获得CausalResnetBlock1D快的输入
def get_input_sets(model, inputs):
    # 清空之前缓存
    features_by_layer_resnet.clear()  
    
    handles_res = register_hooks_for_resnet_blocks(model)    

    for _ in cosyvoice.inference_zero_shot(*inputs, stream=False):
        break  # 只跑一个step收集特征

    for h in handles_res:
        h.remove()   

    
    collected_res = {}
    for name in features_by_layer_resnet:
        collected_res[name] = features_by_layer_resnet[name][0]
   
    # trans_basic
    inputs_by_block = collect_transformer_inputs(model)

    return collected_res, inputs_by_block


# 获得trans快的输入
def find_transformer_inputs(model, input):
    for _ in cosyvoice.inference_zero_shot(*input, stream=False):
        break
    
    inputs_by_block = collect_transformer_inputs(model)

    return inputs_by_block

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
def compute_rgs_score(block, inps, alpha, block_type) -> Dict[str, torch.Tensor]:
    sq_grad = {name: torch.zeros_like(param) for name, param in block.named_parameters()}

    activation_cache.clear()
    register_input_hooks(block)  # 注册 hook
    #print("current inps:", inps)
    for inp in inps:
        if block_type == 'res':
            device = next(block.parameters()).device
            x = inp['x'].to(device)
            mask = inp['mask'].to(device)
            time_emb = inp['time_emb'].to(device)

            loss = torch.norm(block(x, mask, time_emb))
        elif block_type == 'trans':
            device = next(block.parameters()).device
            inp = inp.to(device)
            loss = torch.norm(block(inp))

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
    #print("block_type is:", block_type)
    for name, param in block.named_parameters():
        cache_name = name.rsplit(".", 1)[0]
        if param.requires_grad and name.endswith('weight') and cache_name in activation_cache:
            layer_inps = torch.stack(get_signal(cache_name))
            #print("layer_inps: ",layer_inps.shape)
            norm_term = layer_inps.norm(p=2, dim=tuple(range(layer_inps.ndim - 1)))  # shape: [in_features]
            norm_term = norm_term.unsqueeze(0)  # shape: [1, in_features]
            #print("param shape", param.data.shape,'norm_term shape: ',norm_term.shape)
            score = param.data.abs() * (alpha * sq_grad[name] + norm_term)
            # 把权重的剪枝打分保存起来，后续用于排序和剪枝
            score_dict[name] = score

    return score_dict

# ========== Algorithm 1: Wanda++ Pruning ==========
TARGET_BLOCKS = (CausalResnetBlock1D, BasicTransformerBlock)
def wanda_plus_plus_pruning(model, input_sets, input_sets_trans, alpha, K, sparsity):
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
        if not isinstance(block, TARGET_BLOCKS):
            continue
        X_l = []
        block_type = ''
        if isinstance(block, CausalResnetBlock1D):
            X_l = input_sets[name] 
            block_type = 'res'
        elif isinstance(block, BasicTransformerBlock):
            X_l = input_sets_trans[name]
            block_type = 'trans'

        rgs_score = compute_rgs_score(block, X_l, alpha, block_type)
        #print("******************* rgs_score:",rgs_score)
        
        group_size = 64
        #sparsity = 0.75
        RO_batch = 32
        # 复制未剪枝的 block 作为 teacher（冻结参数）
        block_orig = deepcopy(block)
        for k in range(K):
            
            # Step 2: Select RO samples (you can do random sampling here) ，eg select 2 samples
            X_hat_l = random.sample(X_l, min(len(X_l), RO_batch))  

            # Step 3: Pruning by RGS，input_feature上64个一组
            
            # for name, param in block.named_parameters():
            #     if name in rgs_score:
            #         # Prune lowest X% scores (e.g. 30%)
            #         score = rgs_score[name]
            #         threshold = torch.quantile(score, 0.3)
            #         mask = (score > threshold).float()
            #         print("param.data.shape: ", param.data.shape)
            #         print("param.data:", param.data)
            #         param.data *= mask  # Zero out pruned weights

            for name, param in block.named_parameters():
                if name in rgs_score:
                    score = rgs_score[name]  # shape: [out_features, in_features]
                    mask = torch.zeros_like(score)
                    in_features = score.shape[1]
                    num_groups = (in_features + group_size - 1) // group_size
                    for g in range(num_groups):
                        start = g * group_size
                        end = min((g + 1) * group_size, in_features)
                        group_score = score[:, start:end]  # shape: [out_features, group_size]
                        # 计算阈值
                        threshold = torch.quantile(group_score, sparsity)
                        group_mask = (group_score > threshold).float()
                        mask[:, start:end] = group_mask
                    param.data *= mask  

            # Step 4: Regional Optimization (fine-tune with small batch)            
            block_orig.eval()
            for p in block_orig.parameters():
                p.requires_grad = False

            # 启用当前剪枝后的 block（作为 student）
            block.train()
            optimizer = torch.optim.RMSprop(block.parameters(), lr=3e-7)

            if block_type == 'res':
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
            elif block_type == 'trans':
                for x_m in X_hat_l:
                    optimizer.zero_grad()
                    
                    device = next(block.parameters()).device
                    x_m = x_m.to(device)
                    
                    with torch.no_grad():
                        y_teacher = block_orig(x_m)
                    y_student = block(x_m)
                    loss = torch.nn.functional.mse_loss(y_student, y_teacher)
                    loss.backward()
                    optimizer.step() 

        # Final pruning after RO
        final_score = compute_rgs_score(block, X_l, alpha, block_type)
        #print("final_score: ",final_score)

        # for name, param in block.named_parameters():
        #     if name in final_score:
        #         score = final_score[name]
        #         threshold = torch.quantile(score, 0.3)
        #         mask = (score > threshold).float()
        #         #print("origin param.data: ", param.data)
        #         #print("mask ratio: ", zero_ratio(mask))
        #         param.data *= mask
        #         #print("param.data:", param.data)
        
        for name, param in block.named_parameters():
            if name in final_score:
                score = final_score[name]  # shape: [out_features, in_features]
                mask = torch.zeros_like(score)
                in_features = score.shape[1]
                num_groups = (in_features + group_size - 1) // group_size
                for g in range(num_groups):
                    start = g * group_size
                    end = min((g + 1) * group_size, in_features)
                    group_score = score[:, start:end]  # shape: [out_features, group_size]
                    # 计算阈值
                    threshold = torch.quantile(group_score, sparsity)
                    group_mask = (group_score > threshold).float()
                    mask[:, start:end] = group_mask
                param.data *= mask  

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

def create_audio_tuples(dataset_path, output_size=128):
    """
    生成(text, context, audio_array)元组，其中audio_array是对应text的16kHz音频
    
    参数:
        dataset_path: train-clean-100的路径
        output_size: 需要生成的样本数量
    
    返回:
        [(text_str, context_str, audio_array), ...]
    """
    # 1. 首先收集所有可用的(text, audio_path)对
    id2files = defaultdict(list)
    text_audio_pairs = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.normalized.txt'):
            # 拼接完整路径
                full_path = os.path.join(root, f)
                # 去掉后缀
                no_ext = os.path.splitext(full_path)[0][:-len('.normalized')]
                # 拿到 test-clean 后的第一级ID
                rel_path = os.path.relpath(full_path, dataset_path)
                id_key = rel_path.split(os.sep)[0]                
                id2files[id_key].append(no_ext)
    

    pairs = []
    for id_key, file_list in id2files.items():
        if len(file_list) < 2:
            continue  
        context_path = file_list[0] + '.normalized.txt'
        with open(context_path, 'r', encoding='utf-8') as f:
            context_str = f.read().strip()
        prompt_speech = file_list[0] + '.wav'
        for text_path in file_list[1:]:
            with open(text_path + '.normalized.txt', 'r', encoding='utf-8') as f:
                text_str = f.read().strip()
            pairs.append((text_str, context_str, prompt_speech))
            if len(pairs) >= output_size:
                break
        if len(pairs) >= output_size:
            break

    
    # for i, (text, context, prompt) in enumerate(pairs[:3]):
    #     print(text, "\n" ,context, "\n" ,prompt)
    return pairs

if __name__ == "__main__":

    # 加载模型
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    model = cosyvoice.model
    flow_blocks = model.flow

    train_clean_100_path = "/home/linlinsong/CosyVoice/LibriTTS/test-clean"  # 替换为实际路径    
    # 生成数据
    pairs = create_audio_tuples(train_clean_100_path, 128)
    #print(pairs)

    # 初始化一个字典，键是 block 名，值是该 block 的所有样本输入
    block_input_sets_res = {}
    block_input_sets = {}
    for i in range(len(pairs)):
        text = pairs[i][0]
        context = pairs[i][1]
        prompt_speech_16k = load_wav(pairs[i][2],16000)

        collected_res, transformer_inputs = get_input_sets(model, (text, context, prompt_speech_16k))
        
        # 遍历每个 block 的输入，并存储到 block_input_sets
        for block_name, block_input in collected_res.items():
            if block_name not in block_input_sets_res:
                block_input_sets_res[block_name] = []  
            block_input_sets_res[block_name].append(block_input)  
        
        for block_name, block_input in transformer_inputs.items():
            if block_name not in block_input_sets:
                block_input_sets[block_name] = []
            block_input_sets[block_name].append(block_input)  
            
        
    # print("原模型非零参数数量：", count_nonzero_parameters(flow_blocks))
    sparsity_ls = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8]
    for sparsity in sparsity_ls:
        pruned_model = wanda_plus_plus_pruning(flow_blocks, block_input_sets_res, block_input_sets, 0.5, 5,sparsity)
        torch.save(pruned_model.state_dict(),'pruned_model'+ str(sparsity) +'.pth')
    # print("剪枝后非零参数数量：", count_nonzero_parameters(pruned_model))# print(pruned_model)
    # print("pruned_model is flow_blocks:", pruned_model is flow_blocks) 



