#!/bin/bash
SERVER_TYPE='hf'
DATE=$(date +%Y%m%d%H%M%S)

if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ "$SERVER_TYPE" == "hf" ]; then
    nohup python server/huggingface.py --models Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max_concurrency 5 --port 8000 2>&1 >./logs/server-hf-${DATE}.log &
elif [ "$SERVER_TYPE" == "sglang" ]; then
    nohup python server/sglang.py --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --port 8000 2>&1 >./logs/server-sglang-${DATE}.log &
fi