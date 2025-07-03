#!/bin/bash
nohup python server.py --model Qwen/Qwen2.5-0.5B --device cuda:0 --max_concurrency 5 --port 8000 --hf-token $HF_TOKEN 2>&1 >./server.log &