#!/bin/bash
nohup python server.py --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B --device cuda:0 --max_concurrency 5 --port 8000 2>&1 >./server.log &