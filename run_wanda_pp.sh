python main.py \
    --model /ssd1/haozeng/models/meta-llama/Llama-3.2-1B-Instruct \
    --prune_method wanda_pp \
    --sparsity_ratio 0.5 \
    --sparsity_type 32:64 \
    --seed 0 \
    --alpha 0.4 \
    --k_rounds 10 \
    --save out/Llama-3.2-1B-Instruct/unstructured/wanda_pp/