#!/bin/bash

MODEL_PATH="$1"
PC_PATH="$2"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    --mixed_precision fp16 \
    infer_pc.py \
    --config config/nautilus_infer.yaml \
    --model_path "$MODEL_PATH" \
    --output_path outputs \
    --batch_size 1 \
    --temperature 0.5 \
    --pc_path "$PC_PATH"