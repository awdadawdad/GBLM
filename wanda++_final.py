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

from pdb import set_trace as st 

def no_zero(data):
    zero_count = (data == 0).sum().item()
    return zero_count

def plot_subsampled_matrix_and_save(matrix, output_prefix, subsample_factor):
    odd_subsampled_matrix = matrix[::subsample_factor, ::subsample_factor]
    even_subsampled_matrix = matrix[1::subsample_factor, 1::subsample_factor]
    ones_matrix = np.ones_like(odd_subsampled_matrix)
    zeros_matrix = np.zeros_like(even_subsampled_matrix)
    # print(ones_matrix)
    # print(zeros_matrix)
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(odd_subsampled_matrix, cmap='gray', interpolation='nearest')
    plt.title('Odd Subsampling')
    plt.grid(which='both', color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 2)
    plt.imshow(even_subsampled_matrix, cmap='gray', interpolation='nearest')
    plt.title('Even Subsampling')
    plt.grid(which='both', color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 3)
    plt.imshow(ones_matrix, cmap='gray', interpolation='nearest')
    plt.title('All Ones')
    plt.grid(which='both', color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 4)
    plt.imshow(zeros_matrix, cmap='gray_r', interpolation='nearest')
    plt.title('All Zeros')
    plt.grid(which='both', color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(output_prefix + '_subsampled_plots.png', dpi=300)
    plt.clf()  # Clear the figure after saving


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
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, nsamples, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 强制使用传入的 device（cuda:0），忽略 hf_device_map
    device = torch.device("cuda:0")

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
    hidden_states = inputs.unsqueeze(0)

    if (position_ids is not None) and (rotary_emb is not None):
        cos, sin = rotary_emb(hidden_states, position_ids=position_ids)
        position_embeddings = (cos, sin)
        return layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
    else:
        return layer(hidden_states, attention_mask=attention_mask)[0]

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, layer_no=-1):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                thresh = torch.sort(W_metric.flatten())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
            
            W[W_mask] = 0

def prune_gradient(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, layer_no=-1):

    layers = model.model.layers
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(args.gradient_path, map_location=torch.device('cpu')) 
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            indexed_name = f"{name}_layer_{i}"
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if not args.gradient_inv:
                W_metric = W_metric.to(dtype=torch.float32) * torch.abs(gradients[indexed_name].to(device=W_metric.device)).to(dtype=torch.float32)#+ small_value)
            else:
                small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                gradient_inv = 1 / (torch.abs(gradients[indexed_name]) + small_value)
                W_metric = W_metric.to(dtype=torch.float32) * gradient_inv.to(device=W_metric.device).to(dtype=torch.float32)
            W_mask = (torch.zeros_like(W)==1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            W[W_mask] = 0


def prune_gblm(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, layer_no=-1):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(args.gradient_path, map_location=torch.device('cpu')) 

    print("loading calibdation data")
    dataloader, testenc = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, args.nsamples, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 强制所有操作在 cuda:0 上进行
        dev = torch.device("cuda:0")
        # 将 inps/outs 转移到目标设备
        inps = inps.to(dev)
        outs = outs.to(dev)
        # attention_mask 或 position_ids 可能为 None，需要判空后再转移到设备
        att_mask_dev = attention_mask.to(dev) if attention_mask is not None else None
        pos_ids_dev = position_ids.to(dev) if position_ids is not None else None

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], layer_id=i, layer_name=name)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name))) ## this is a important function.
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=att_mask_dev, position_ids=pos_ids_dev)[0]

        for h in handles:
            h.remove() 

        for sub_i, name in enumerate(subset):
            indexed_name = f"{name}_layer_{i}"
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            if not args.gradient_inv:
                # small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                W_metric_grad = torch.abs(subset[name].weight.data)* torch.abs(gradients[indexed_name].to(device=W_metric.device))
                W_metric = W_metric.to(dtype=torch.float32) + W_metric_grad.to(dtype=torch.float32)  #+ small_value)
            else:
                small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                gradient_inv = 1 / (torch.abs(gradients[indexed_name]) + small_value)
                W_metric = W_metric.to(dtype=torch.float32)  * gradient_inv.to(device=W_metric.device).to(dtype=torch.float32) 

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data.masked_fill_(W_mask, 0.0)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=att_mask_dev, position_ids=pos_ids_dev)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, layer_no=-1):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, testenc = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, args.nsamples, device)

    # The rotary embedding layer is part of the model, not the individual layers
    rotary_emb = model.model.rotary_emb

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 强制所有操作在 cuda:0 上进行
        dev = torch.device("cuda:0")
        # 将 inps/outs 转移到目标设备
        inps = inps.to(dev)
        outs = outs.to(dev)
        # attention_mask 或 position_ids 可能为 None，需要判空后再转移到设备
        att_mask_dev = attention_mask.to(dev) if attention_mask is not None else None
        pos_ids_dev = position_ids.to(dev) if position_ids is not None else None

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], layer_id=i, layer_name=name)
            print(wrapped_layers[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name))) ## this is a important function.
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = _get_layer_output(layer, inps[j], att_mask_dev, pos_ids_dev, rotary_emb)

        for h in handles:
            h.remove() 

        for sub_i, name in enumerate(subset):
        
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data.masked_fill_(W_mask, 0.0)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = _get_layer_output(layer, inps[j], att_mask_dev, pos_ids_dev, rotary_emb)
        
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0, layer_no=-1):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, testenc = get_loaders(
        "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer
    )
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, args.nsamples, dev)
    
    rotary_emb = model.model.rotary_emb

    layers = model.model.layers
    print("model.model.layers",layers)
    if layer_no != -1:
        layers = layers[layer_no:layer_no+1]

    for i in range(len(layers)):
        layer = layers[i]
        # 强制所有操作在 cuda:0 上进行
        dev = torch.device("cuda:0")
        # 将 inps/outs 转移到目标设备
        inps = inps.to(dev)
        outs = outs.to(dev)
        # attention_mask 或 position_ids 可能为 None，需要判空后再转移到设备
        att_mask_dev = attention_mask.to(dev) if attention_mask is not None else None
        pos_ids_dev = position_ids.to(dev) if position_ids is not None else None

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = _get_layer_output(layer, inps[j], att_mask_dev, pos_ids_dev, rotary_emb)
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            gpts[name].fasterprune(
                args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128
            )
            gpts[name].free()

    model.config.use_cache = True 
    print('Done.')

def wanda_pp(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, layer_no=-1):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    # TODO: 在整个块执行前，获得当前块的输入，方便后续做前向传播
    print("loading calibdation data for Wanda++")
    dataloader, testenc = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, args.nsamples, device)

    # TODO：layers应该是整个待剪枝的模型
    rotary_emb = model.model.rotary_emb
    layers = model.model.layers
    print(" model.model.layers:", layers)
    # Hyperparameters for Wanda++ are now taken from args
    WANDA_PP_ALPHA = args.alpha
    WANDA_PP_K = args.k_rounds
    WANDA_PP_RO_SAMPLES = args.ro_samples
    WANDA_PP_LR = args.lr

    # TODO:第一个循环，对每一个块操作
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 强制所有操作在 cuda:0 上进行
        dev = torch.device("cuda:0")
        # 将 inps/outs 转移到目标设备
        inps = inps.to(dev)
        outs = outs.to(dev)
        # attention_mask 或 position_ids 可能为 None，需要判空后再转移到设备
        att_mask_dev = attention_mask.to(dev) if attention_mask is not None else None
        pos_ids_dev = position_ids.to(dev) if position_ids is not None else None

        def compute_rgs_score(current_layer):
            activation_cache = {name: [] for name in subset}
            def get_hook(name):
                def hook(_, inp, out):
                    activation_cache[name].append(inp[0].detach())
                return hook
            
            handles = []
            for name, mod in current_layer.named_modules():
                if name in subset:
                    handles.append(mod.register_forward_hook(get_hook(name)))
            print("args.nsamples:",args.nsamples)
            for j in range(args.nsamples):
                with torch.no_grad():
                    _get_layer_output(current_layer, inps[j].to(dev), att_mask_dev, pos_ids_dev, rotary_emb)
            for h in handles:
                h.remove()

            sq_grad = {name: torch.zeros_like(p.weight) for name, p in subset.items()}
            for j in range(args.nsamples):
                output = _get_layer_output(current_layer, inps[j].to(dev), att_mask_dev, pos_ids_dev, rotary_emb)
                loss = torch.norm(output)
                loss.backward()

                with torch.no_grad():
                    for name, p in subset.items():
                        if p.weight.grad is not None:
                            sq_grad[name] += p.weight.grad.detach() ** 2
                current_layer.zero_grad(set_to_none=True)

            score_dict = {}
            for name, p in subset.items():
                avg_sq_grad = torch.sqrt(sq_grad[name] / args.nsamples)
                act_tensor = torch.stack(activation_cache[name]).to(dev)
                norm_term = act_tensor.norm(p=2, dim=(0, 1))
                
                # 根据权重形状智能广播
                if norm_term.numel() == p.weight.shape[1]:
                    # norm 对输入维度求得，与列数一致
                    norm_broadcast = norm_term.unsqueeze(0)  # shape (1, in_dim)
                elif norm_term.numel() == p.weight.shape[0]:
                    # 与行数一致
                    norm_broadcast = norm_term.unsqueeze(1)  # shape (out_dim,1)
                else:
                    # 退化到对标量广播，取 L2 范数均值
                    norm_broadcast = norm_term.mean().view(1, 1)

                score = p.weight.data.abs() * (WANDA_PP_ALPHA * avg_sq_grad + norm_broadcast)
                score_dict[name] = score
            
            return score_dict

        print(f"Layer {i}: Computing initial RGS score...")
        initial_rgs_score = compute_rgs_score(layer)

        print(f"Layer {i}: Starting Regional Optimization for {WANDA_PP_K} rounds...")
        for k in range(WANDA_PP_K):
            print(f"  RO round {k+1}/{WANDA_PP_K}")
            layer_orig = deepcopy(layer).eval()
            for p in layer_orig.parameters():
                p.requires_grad = False
            
            ro_indices = random.sample(range(args.nsamples), WANDA_PP_RO_SAMPLES)

            with torch.no_grad():
                for name, p in subset.items():
                    score = initial_rgs_score[name]
                    k_to_prune = int(p.weight.numel() * args.sparsity_ratio)
                    if k_to_prune == 0:
                        W_mask = torch.zeros_like(score, dtype=torch.bool)
                    elif k_to_prune >= score.numel():
                        W_mask = torch.ones_like(score, dtype=torch.bool)
                    else:
                        flat_score = score.flatten()
                        # 选取得分最低的 k 个索引
                        _, low_idx = torch.topk(flat_score, k_to_prune, largest=False)
                        mask_flat = torch.zeros_like(flat_score, dtype=torch.bool)
                        mask_flat[low_idx] = True
                        W_mask = mask_flat.view_as(score)
                    p.weight.data.masked_fill_(W_mask, 0.0)

            layer.train()
            optimizer = torch.optim.AdamW(layer.parameters(), lr=WANDA_PP_LR)
            
            for j in ro_indices:
                optimizer.zero_grad()
                inp_j = inps[j].to(dev)
                
                with torch.no_grad():
                    y_teacher = _get_layer_output(layer_orig, inp_j, att_mask_dev, pos_ids_dev, rotary_emb)
                
                y_student = _get_layer_output(layer, inp_j, att_mask_dev, pos_ids_dev, rotary_emb)
                
                loss = F.mse_loss(y_student, y_teacher)
                loss.backward()
                optimizer.step()

            del layer_orig
            torch.cuda.empty_cache()

        print(f"Layer {i}: Computing final score and pruning...")
        layer.eval()
        final_score = compute_rgs_score(layer)

        with torch.no_grad():
            for name, p in subset.items():
                score = final_score[name]
                k_to_prune = int(p.weight.numel() * args.sparsity_ratio)
                if k_to_prune == 0:
                    W_mask = torch.zeros_like(score, dtype=torch.bool)
                elif k_to_prune >= score.numel():
                    W_mask = torch.ones_like(score, dtype=torch.bool)
                else:
                    flat_score = score.flatten()
                    # 选取得分最低的 k 个索引
                    _, low_idx = torch.topk(flat_score, k_to_prune, largest=False)
                    mask_flat = torch.zeros_like(flat_score, dtype=torch.bool)
                    mask_flat[low_idx] = True
                    W_mask = mask_flat.view_as(score)
                
                p.weight.data.masked_fill_(W_mask, 0.0)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = _get_layer_output(layer, inps[j], att_mask_dev, pos_ids_dev, rotary_emb)
        
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()



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

def get_signal(name: str):
    """
    Get recorded input activations for a given layer name.
    """
    return activation_cache.get(name, [])

def compute_rgs_score(block, inps, alpha):
    sq_grad = {name: torch.zeros_like(param) for name, param in block.named_parameters()}

    activation_cache.clear()
    register_input_hooks(block)  

    for inp in inps:
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

    for name, param in block.named_parameters():
        cache_name = name.rsplit(".", 1)[0]
        if param.requires_grad and name.endswith('weight') and cache_name in activation_cache:
            
            layer_inps = torch.stack(get_signal(cache_name))
            norm_term = layer_inps.norm(p=2, dim=[0,1])
            norm_term = norm_term.unsqueeze(0)
            score = param.data.abs() * (alpha * sq_grad[name] + norm_term) # 论文核心算法
            # 把权重的剪枝打分保存起来，后续用于排序和剪枝
            score_dict[name] = score

    return score_dict

def wanda_plus_plus_pruning(model, input_sets, alpha, K):
    """
    Args:
        model: 整个模型有很多待剪枝的块，这里是其中一个
        input_sets: 每个块的输入，根据prune_wanda函数，应该是使用prepare_calibration_input获得。不同点在于，prepare_calibration_input可能是每个linear层的，这里需要每个块的
        alpha: scaling factor for score
        K: RO optimization rounds
    Returns:
        pruned model
    """
    for name, block in model.named_modules():
        X_l = input_sets[name] 

        rgs_score = compute_rgs_score(block, X_l, alpha)
        
        for k in range(K):
            # 复制未剪枝的 block 作为 teacher（冻结参数）
            block_orig = deepcopy(block)
            # Step 2: 随机选择一些样本，比如32条
            X_hat_l = random.sample(X_l, min(len(X_l), 32))  

            # Step 3: Pruning by RGS
            for name, param in block.named_parameters():
                if name in rgs_score:
                    # Prune lowest X% scores (e.g. 30%)
                    score = rgs_score[name]
                    threshold = torch.quantile(score, 0.3)
                    mask = (score > threshold).float()
                    #print("param.data.shape: ", param.data.shape)
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
                x_m = x_m.to(device)
                with torch.no_grad():
                    y_teacher = block_orig(x_m)
                y_student = block(x_m)
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
                param.data *= mask

    return model

