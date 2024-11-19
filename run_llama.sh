#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py \
    --contrast \
    --model open_llama_3b \
    --target_model_name_or_path anonymous-repository/open_llama_3b-multi \
    --torch_dtype bfloat16 \
    --bf16 \
    --bf16_full_eval \
    --distill_num 1 \
    --alpha 0.1 \
    --gamma 1 \
    --distill_num_beams 4 \
    --logging_strategy no \
    --do_eval False \
    --evaluation_strategy no \
    --save_strategy no \
    --no_save \
    --distill_steps 100 &    
wait