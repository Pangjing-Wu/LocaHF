import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, pipeline



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
parser = argparse.ArgumentParser("Serve a HF chat model via FastAPI.")
parser.add_argument("--models", type=str, nargs='+', help="🤗 Hub model ID or local path")
parser.add_argument("--device", default='cuda:0', help="GPU idx, 'auto', or 'cpu'")
parser.add_argument("--max_concurrency", type=int, default=1)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for CLI login")
args = parser.parse_args()

# set up logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("Logging initialised (level=INFO)")


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
tokenizers = dict()
hf_pipelines = dict()
infos = dict()
semas = dict()

for model in args.models:
    logging.info(f"Loading tokenizer for {model} ...")
    tokenizers[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    logging.info(f"Loading model on device={args.device} ...")
    hf_pipelines[model] = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizers[model],
        trust_remote_code=True,
        device_map=args.device
    )
    infos[model] = {
        "param_m": sum(p.numel() for p in hf_pipelines[model].model.parameters()) // 1_000_000_000,
        "max_concurrency": args.max_concurrency,
    }
    semas[model] = asyncio.Semaphore(args.max_concurrency)


# build FastAPI app.
# ---------------------------------------------------------------------
app = FastAPI(title="LocaHF", version="1.0")


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None


def build_prompt(model: str, msgs: List[Message]) -> str:
    try:
        return tokenizers[model].apply_chat_template(
            [m.model_dump() for m in msgs],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        logging.warning("[WARN] tokenizer.apply_chat_template failed - using fallback prompt template.")
        return "\n".join(f"{m.role}: {m.content}" for m in msgs) + "\nassistant:"


async def run_gen(model: str, prompt: str, gcfg: Dict[str, Any]) -> str:
    loop = asyncio.get_event_loop()
    async with semas[model]:
        out = await loop.run_in_executor(
            None,
            lambda: hf_pipelines[model](prompt, **gcfg, do_sample=True)[0]["generated_text"],
        )
    return out[len(prompt):]


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.model not in args.models:
        raise HTTPException(400, "Model name mismatch")
    
    gcfg = {
        "temperature":    req.temperature    or DEFAULT_MODEL_CONFIGS["generation"]["temperature"],
        "top_p":          req.top_p          or DEFAULT_MODEL_CONFIGS["generation"]["top_p"],
        "max_new_tokens": req.max_new_tokens or DEFAULT_MODEL_CONFIGS["generation"]["max_new_tokens"],
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
        "device": args.device,
        "max_concurrency": args.max_concurrency,
        "generation_defaults": DEFAULT_MODEL_CONFIGS["generation"],
        "param_count": [f'{infos[model]["param_m"]}B' for model in args.models],
    }


if __name__ == "__main__":
    logging.info(f"Ready - {infos[args.models[0]]['param_m']}B params | max concurrency = {infos[args.models[0]]['max_concurrency']}")
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