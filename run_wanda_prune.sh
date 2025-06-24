python main.py \
    --model /ssd1/yanwengai/Llama-2-7b-chat-hf \
    --prune_method wanda \
    --nsamples 128 \
    --seed 0 \
    --sparsity_ratio 0.5 \
    --sparsity_type 32:64 \
    --save out/llama_7b/unstructured/wanda/
