# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Clip test sequence to avoid positions beyond model's max rotary embedding size
    max_test_tokens = seqlen * 128  # e.g., 128 segments of length seqlen
    testenc_ids = testenc.input_ids[:, :max_test_tokens]

    # Wrap into TokenizerWrapper for compatibility
    testenc = TokenizerWrapper(testenc_ids)

    # NOTE: Downstream eval_ppl expects .input_ids attribute

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        # tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    print("loading calibdation data")
    # Load train and validation datasets with streaming
    traindata = load_dataset('c4', 'en', split='train', streaming=True)
    valdata = load_dataset('c4', 'en', split='validation', streaming=True)
    print("loading success")

    # Generate samples from training set using streaming
    random.seed(seed)
    trainloader = []
    for data in traindata:
        if len(trainloader) == nsamples:
            break
        try:
            trainenc = tokenizer(data['text'], return_tensors='pt')
        except:
            continue
        if trainenc.input_ids.shape[1] > seqlen:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            trainloader.append((inp, tar))
    
    # Process validation dataset
    val_text = ""
    for data in valdata:
        val_text += data['text']
        if len(val_text) > (seqlen * 100): # Collect enough text for validation
             break
    
    valenc = tokenizer(val_text, return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    
    # 如果没有匹配的数据集，返回默认值防止 None
    print(f"Warning: Unknown dataset name '{name}', falling back to wikitext2")
    return get_wikitext2(nsamples, seed, seqlen, tokenizer)