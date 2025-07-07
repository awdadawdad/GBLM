python main.py \
    --model /ssd1/haozeng/models/meta-llama/Llama-3.2-1B-Instruct \
    --prune_method wanda \
    --nsamples 128 \
    --seed 0 \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/
