#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)

SERVER_TYPE=$1
MODEL_NAME="${@:2}"

if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ $SERVER_TYPE == "hf" ]; then
    nohup python server/hf-api.py --models $MODEL_NAME --device cuda:0 --max_concurrency 5 --port 8000 2>&1 >./logs/server-hf-${DATE}.log &
elif [ $SERVER_TYPE == "sglang" ]; then
    nohup python server/sglang-api.py --model $MODEL_NAME --device cuda:0 --port 8000 2>&1 >./logs/server-sglang-${DATE}.log &
elif [ $SERVER_TYPE == "vllm" ]; then
    nohup python server/vllm-api.py --models $MODEL_NAME --device cuda:0 --port 8000 2>&1 >./logs/server-vllm-${DATE}.log &
elif [ $SERVER_TYPE == "vllm-light" ]; then
    nohup vllm serve $MODEL_NAME --port 8000 --gpu-memory-utilization 0.7 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 2 --tensor-parallel-size 1 2>&1 >./logs/server-vllm-${DATE}.log &
fi