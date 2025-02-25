#! /bin/bash

python fine_tune_llm.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --train_file example_data/train_human.json \
    --dataset_text_field sentence \
    --output_dir example_data/output/fine_tuned_model \
    --seq_length 4096 \