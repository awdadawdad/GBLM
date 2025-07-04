import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import gc
import csv
import os
from copy import deepcopy
import random
from collections import defaultdict

from pdb import set_trace as st 

def no_zero(data):
    zero_count = (data == 0).sum().item()
    return zero_count

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model, args):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    print(f"total zero params {count}")
    print(f"total params {total_params}")
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, nsamples, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 强制使用传入的 device（cuda:0），忽略 hf_device_map
    device = device

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def _get_layer_output(layer, inputs, attention_mask, position_ids, rotary_emb):
    """Helper function to get layer output that gracefully handles missing masks/ids."""
    # 输入可能已经带有 batch 维度 (B, L, D)。如果只有 (L, D)，则补一个 batch 维度。
    if inputs.dim() == 2:
        hidden_states = inputs.unsqueeze(0)
    else:
        hidden_states = inputs

    if (position_ids is not None) and (rotary_emb is not None):
        cos, sin = rotary_emb(hidden_states, position_ids=position_ids)
        position_embeddings = (cos, sin)
        return layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
    else:
        return layer(hidden_states, attention_mask=attention_mask)[0]



def get_each_transformer_layer_input(model, dataloader, nsamples, device):
    activation_cache = defaultdict(list)
    def create_hook(name):
        def hook_fn(module, inp, out):
            # inp 是一个元组，我们通常关心第一个元素
            # 使用 detach() 来防止内存泄漏
            activation_cache[name].append(inp[0].detach())
        return hook_fn

    # 2. 遍历层并注册钩子
    layers = model.model.layers
    handles = []
    for i, layer in enumerate(layers):
        # 使用一个唯一的名称作为 key，格式为 model.layers.{i}
        key_name = f"model.layers.{i}"
        handle = layer.register_forward_hook(create_hook(key_name))
        handles.append(handle)

    # 3. 执行前向传播来触发所有钩子
    # 我们这里只处理 nsamples 个样本
    for i, batch in enumerate(dataloader):
        if (nsamples is not None) and (nsamples > 0) and (i >= nsamples):
            break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    # 5. 立即移除钩子，清理现场
    for handle in handles:
        handle.remove()

    # 现在 activation_cache 里就有了所有层的输入激活值
    # 打印结果来验证
    for layer_idx, activations in activation_cache.items():
        pass  # 保留占位，防止空代码块导致 IndentationError
        # 可以根据需要取消下面的注释来打印调试信息
        # print(f"Layer {layer_idx}:")
        # print(f"  - Captured {len(activations)} activation tensors.")
        # print(f"  - Shape of first activation tensor: {activations[0].shape}")
    for key, acts in activation_cache.items():
        shapes = {tuple(a.shape[-1:]) for a in acts}
        if len(shapes) > 1:
            print('冲突层:', key, '捕获到的维度集合:', shapes)
    return activation_cache


activation_cache = {}
def register_input_hooks(block: nn.Module):
    """Register forward hooks for all Linear layers in *block* and return the handles so they can be removed later."""
    handles = []
    for name, submodule in block.named_modules():
        # if isinstance(submodule, nn.Linear): # 仅关注非 down_proj 的 Linear 层
        if isinstance(submodule, nn.Linear) and ("down_proj" not in name):
            def hook_fn(mod, inp, out, module_name=name):
                if module_name not in activation_cache:
                    activation_cache[module_name] = []
                activation_cache[module_name].append(inp[0].detach())
            handles.append(submodule.register_forward_hook(hook_fn))
    return handles

def get_signal(name: str):
    """
    Get recorded input activations for a given layer name.
    """
    return activation_cache.get(name, [])

def compute_rgs_score(block, inps, alpha, attention_mask, position_ids, rotary_emb, count):
    # print("block: ", block)
    sq_grad = {name: torch.zeros_like(param) for name, param in block.named_parameters()}

    handles = register_input_hooks(block)

    

    for inp in inps:
        device = next(block.parameters()).device
        inp = inp.to(device)

        # Correctly call the forward pass with all necessary arguments
        if (position_ids is not None) and (rotary_emb is not None):
            cos, sin = rotary_emb(inp, position_ids=position_ids)
            position_embeddings = (cos, sin)
            output = block(inp, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        else:
            output = block(inp, attention_mask=attention_mask)[0]
        
        loss = torch.norm(output)

        loss.backward(retain_graph=True)

        with torch.no_grad():
            for name, param in block.named_parameters():
                if param.grad is not None:
                    sq_grad[name] += param.grad.detach() ** 2
        block.zero_grad()
    # DEBUG: 打印钩子捕获的键名（前 10 个）
    # 方便确认 activation_cache 的键与参数名是否对齐
    #print("[DEBUG] activation_cache keys (first 10):", list(activation_cache.keys()))

    inps_len = len(inps)
    if(count == 0):
        print("inps_len: ", inps_len)
    for key in sq_grad:
        sq_grad[key] = torch.sqrt(sq_grad[key] / inps_len)

    score_dict = {}
    # print("block: ", block)
    for name, param in block.named_parameters():
        # print("#" * 20)
        # if(name == "model.layers.0.self_attn.q_proj.weight"):
        # print(f"name: {name}")
        # print("#" * 20)
        cache_name = name.rsplit(".", 1)[0]
        if (param.requires_grad
            and name.endswith('weight')
            and ('down_proj' not in name)  # 跳过 down_proj
            and cache_name in activation_cache):
            # print("activation_cache[cache_name]: ",activation_cache[cache_name])
            layer_inps = torch.stack(get_signal(cache_name))
            norm_term = layer_inps.norm(p=2, dim=tuple(range(layer_inps.ndim - 1)))  # shape: [in_features]
            norm_term = norm_term.unsqueeze(0)
            score = param.data.abs() * (alpha * sq_grad[name] + norm_term) # 论文核心算法
            # 把权重的剪枝打分保存起来，后续用于排序和剪枝
            #if(count == 0):
                # print("param.data: ", param.data)
                #print("param.data.abs(): ", param.data.abs())
            score_dict[name] = score
    # print(f"rgs_score size for this block: {score_dict}")

    # 移除注册的 hook，避免后续推理继续缓存激活导致显存爆炸
    for h in handles:
        h.remove()
    activation_cache.clear()

    return score_dict

def wanda_pp(args, model, tokenizer, device, alpha, K, prune_n=0, prune_m=0):
    """
    Args:
        model: 整个模型有很多待剪枝的块，这里是其中一个
        input_sets: 每个块的输入，根据prune_wanda函数，应该是使用prepare_calibration_input获得。不同点在于，prepare_calibration_input可能是每个linear层的，这里需要每个块的
        alpha: scaling factor for score
        K: RO optimization rounds
    Returns:
        pruned model
    """
    count = 0
    # print("param.data: ", param.data)
    # print("param.data.abs(): ", param.data.abs())
    dataloader, testenc = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    input_sets = get_each_transformer_layer_input(model, dataloader, args.nsamples, device)
    
    # Get other necessary arguments for the forward pass
    _, _, attention_mask, position_ids = prepare_calibration_input(model, dataloader, args.nsamples, device)
    rotary_emb = model.model.rotary_emb
    dev = next(model.parameters()).device
    attention_mask = attention_mask.to(dev) if attention_mask is not None else None
    position_ids = position_ids.to(dev) if position_ids is not None else None

    for name, block in model.named_modules():
        
        # Only apply to top-level transformer layers
        if not name.startswith("model.layers.") or len(name.split('.')) > 3:
            continue
        
        layer_name = name
        if layer_name not in input_sets:
            continue

        X_l = input_sets[layer_name]

        # 记录当前层的 dtype（可能是 bfloat16 / float16）
        orig_dtype = next(block.parameters()).dtype

        # --- 临时转换到 FP32 做剪枝与微调，防止 NaN 溢出 ---
        block = block.to(torch.float32)
        X_l   = [x.to(torch.float32) for x in X_l]

        rgs_score = compute_rgs_score(block, X_l, alpha, attention_mask, position_ids, rotary_emb, count)
        block_orig = deepcopy(block)
        for k in range(K):
            
            # teacher 使用冻结的 FP32 权重
            # block_orig = deepcopy(block)
            # Step 2: 随机选择一些样本，比如32条
            X_hat_l = [x.clone() for x in random.sample(X_l, min(len(X_l), 32))]  # 深拷贝，避免原列表被改动

            
            # Step 3: Pruning by RGS
            for name, param in block.named_parameters():
                if name in rgs_score:
                    score = rgs_score[name].float()
                    if prune_n != 0:
                        # 结构化 n:m 剪枝：每 m 列选出 n 个最小得分置零
                        W_mask_local = torch.zeros_like(score, dtype=torch.bool)
                        for ii in range(score.shape[1]):
                            if ii % prune_m == 0:
                                tmp = score[:, ii:(ii + prune_m)]
                                smallest_idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                                W_mask_local.scatter_(1, ii + smallest_idx, True)
                        param.data.masked_fill_(W_mask_local, 0.0)
                    else:
                        # 非结构化（行内排序）剪枝
                        k_cols = int(score.shape[1] * args.sparsity_ratio)
                        idx = torch.argsort(score, dim=1)[:, :k_cols]
                        mask = torch.zeros_like(score, dtype=torch.bool)
                        mask.scatter_(1, idx, True)
                        param.data.masked_fill_(mask, 0.0)
            
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
                x_m = x_m.to(device, dtype=torch.float32)
                # 通过 _get_layer_output 保证传入 position_embeddings
                with torch.no_grad():
                    y_teacher = _get_layer_output(block_orig, x_m, attention_mask, position_ids, rotary_emb)
                y_student = _get_layer_output(block, x_m, attention_mask, position_ids, rotary_emb)
                loss = torch.nn.functional.mse_loss(y_student, y_teacher)
                loss.backward()
                optimizer.step() 
            
            # —— 释放本轮临时对象，回收显存 ——
            del optimizer, X_hat_l
            torch.cuda.empty_cache()

            for n, p in block.named_parameters():
                if torch.isnan(p).any():
                    print("kkkkkkkkkkkkkkkkkkkkkkkkk = ", k)
                    print("NaN in param -->", n)
        # Final pruning after RO
        final_score = compute_rgs_score(block, X_l, alpha, attention_mask, position_ids, rotary_emb, count)
        #print("final_score: ",final_score)
        for name, param in block.named_parameters():
            if name in final_score:
                score = final_score[name].float()
                if prune_n != 0:
                    W_mask_final = torch.zeros_like(score, dtype=torch.bool)
                    for ii in range(score.shape[1]):
                        if ii % prune_m == 0:
                            tmp = score[:, ii:(ii + prune_m)]
                            smallest_idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                            W_mask_final.scatter_(1, ii + smallest_idx, True)
                    param.data.masked_fill_(W_mask_final, 0.0)
                else:
                    k_cols = int(score.shape[1] * args.sparsity_ratio)
                    idx = torch.argsort(score, dim=1)[:, :k_cols]
                    mask = torch.zeros_like(score, dtype=torch.bool)
                    mask.scatter_(1, idx, True)
                    param.data.masked_fill_(mask, 0.0)
        count += 1

        # 剪枝与微调结束，恢复该层为原始 dtype（BF16/FP16）
        block.to(orig_dtype)
        # 删除 block_orig 等大对象，防止长期占用显存
        del block_orig, rgs_score, final_score, X_l
        torch.cuda.empty_cache()
    # —— 函数结束：清空全局缓存并强制回收显存 ——
    activation_cache.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return model
