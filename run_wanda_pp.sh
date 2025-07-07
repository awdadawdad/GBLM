python main.py \
    --model /ssd3/yanwengai/Qwen3-4B \
    --prune_method wanda_pp \
    --sparsity_ratio  0.5 \
    --sparsity_type unstructured \
    --nsamples 128 \
    --seed 0 \
    --alpha 100 \
    --k_rounds 10 \
    --save out/Llama-3.2-1B-Instruct/unstructured/wanda_pp/