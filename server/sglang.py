import argparse

from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process


if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


parser = argparse.ArgumentParser("Serve a sglang model via FastAPI.")
parser.add_argument("--model", type=str, help="🤗 Hub model ID or local path")
parser.add_argument("--device", default='cuda:0', help="GPU idx, 'auto', or 'cpu'")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for CLI login")
args = parser.parse_args()


server_process, port = launch_server_cmd(f"python3 -m sglang.launch_server --model-path {args.model}", host="0.0.0.0", port=args.port)

wait_for_server(f"http://localhost:{port}")