import argparse

from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process


if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


parser = argparse.ArgumentParser("Serve a sglang model via FastAPI.")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="🤗 Hub model ID or local path")
parser.add_argument("--mem-fraction-static", default='0.7', help="memory usage of the KV cache pool")
parser.add_argument("--dtype", default='auto', help="data type of the model")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()


server_process, port = launch_server_cmd(f"python3 -m sglang.launch_server --model-path {args.model} --mem-fraction-static {args.mem_fraction_static} --dtype {args.dtype}", host="0.0.0.0", port=args.port)

wait_for_server(f"http://localhost:{port}")