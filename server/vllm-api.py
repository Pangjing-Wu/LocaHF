import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import vllm



DEFAULT_MODEL_CONFIGS: Dict[str, Any] = {
    "device": 0, # GPU 0; use "auto" for sharding or -1 for CPU
    "max_concurrency": 1,
    "generation": {
        "temperature": 0.0,
        "top_p": 0.0,
        "max_new_tokens": 128,
    }
}

# parse arguments.
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser("Serve a vLLM model via FastAPI.")
parser.add_argument("--models", type=str, default=["Qwen/Qwen2.5-7B-Instruct"], nargs='+', help="🤗 Hub model ID or local path")
parser.add_argument("--max_concurrency", type=int, default=1)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--gpu_memory_utilization", type=float, nargs='+', default=[0.95])
parser.add_argument("--max_model_len", type=int, default=8192)
parser.add_argument("--max_num_seqs", type=int, default=2)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for CLI login")
args = parser.parse_args()

# set up logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("Logging initialized (level=INFO)")


assert len(args.models) == len(args.gpu_memory_utilization), logging.error("models and gpu_memory_utilization must have the same length")
assert sum(args.gpu_memory_utilization) <= 1.0, logging.error("gpu_memory_utilization must be less than or equal to 1.0")


# Hugging Face CLI login
logging.info("Logging into Hugging Face CLI...")
try:
    from huggingface_hub import login as hf_login_func
    if args.hf_token:
        hf_login_func(token=args.hf_token, add_to_git_credential=True)
except Exception as e:
    logging.error(f"Hugging Face login failed: {e}")
    exit(1)


# load tokenizers and models.
# ---------------------------------------------------------------------
infos = dict()
semas = dict()
vllm_models = dict()
tokenizers = dict()

for model, vram_utilization in zip(args.models, args.gpu_memory_utilization):
    logging.info(f"Loading vllm model {model} ...")
    vllm_models[model] = vllm.LLM(
        model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=vram_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        task="generate"
        )
    semas[model] = asyncio.Semaphore(args.max_concurrency)
    tokenizers[model] = vllm_models[model].get_tokenizer()

# build FastAPI app.
# ---------------------------------------------------------------------
app = FastAPI(title="Local vLLM", version="1.0")


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None


async def run_gen(model: str, messages: List[Message], gcfg: Dict[str, Any]) -> str:
    loop = asyncio.get_event_loop()
    async with semas[model]:
        out = await loop.run_in_executor(
            None,
            lambda: vllm_models[model].generate(messages, sampling_params=vllm.SamplingParams(**gcfg))
        )
    return out[0].outputs[0].text


def build_prompt(model: str, msgs: List[Message]) -> str:
    try:
        
        return tokenizers[model].apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        logging.warning("[WARN] tokenizer.apply_chat_template failed - using fallback prompt template.")
        return "\n".join(f"{m.role}: {m.content}" for m in msgs) + "\nassistant:"


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.model not in args.models:
        raise HTTPException(400, "Model name mismatch")
    
    gcfg = {
        "temperature":req.temperature    or DEFAULT_MODEL_CONFIGS["generation"]["temperature"],
        "top_p":      req.top_p          or DEFAULT_MODEL_CONFIGS["generation"]["top_p"],
        "max_tokens": req.max_new_tokens or DEFAULT_MODEL_CONFIGS["generation"]["max_new_tokens"],
    }
    prompt = build_prompt(req.model, req.messages)
    try:
        completion = await run_gen(req.model, prompt, gcfg)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))

    return {
        "id": "cmpl-local",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion},
            "finish_reason": "stop"
        }],
        "model": req.model,
    }


@app.get("/v1/models")
def meta():
    return {
        "name": args.models,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_concurrency": args.max_concurrency,
        "generation_defaults": DEFAULT_MODEL_CONFIGS["generation"],
    }


if __name__ == "__main__":
    logging.info(f"Ready - max concurrency = {args.max_concurrency}")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
        )
    except Exception:
        logging.exception("Fatal error in main loop")
        raise