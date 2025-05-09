import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, pipeline

from utils import load_json_cfg, deep_merge_dict


CONFIG_PATH = Path(__file__).with_name("configs.json")

DEFAULT_MODEL_CONFIGS: Dict[str, Any] = {
    "device": 0, # GPU 0; use "auto" for sharding or -1 for CPU
    "max_concurrency": 1,
    "generation": {
        "temperature": 0.0,
        "top_p": 0.0,
        "max_new_tokens": 128,
    }
}

DEFAULT_LOG_CONFIG: Dict[str, Any] = {
    "level": "info",
    "path": "./server.log",
}

# parse arguments.
# ---------------------------------------------------------------------
# arguments setting levels: arguments > configs.json > default

parser = argparse.ArgumentParser("Serve a HF chat model via FastAPI.")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="🤗 Hub model ID or local path")
parser.add_argument("--device", default='cuda:0', help="GPU idx, 'auto', or 'cpu'")
parser.add_argument("--max_concurrency", type=int, default=1)
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# load configs from configs.json
configs = load_json_cfg(CONFIG_PATH)
model_configs = deep_merge_dict(DEFAULT_MODEL_CONFIGS, configs.get("model", {}))
log_configs   = deep_merge_dict(DEFAULT_LOG_CONFIG, configs.get("log", {}))

# update model_configs from arguments
model_configs['device'] = args.device
model_configs['max_concurrency'] = args.max_concurrency

# set up logging
# ----------------------------------------------------------------------
_log_level_name = str(log_configs.get("level", "info")).lower()
_numeric_level  = getattr(logging, _log_level_name.upper())

logging.basicConfig(
    filename=log_configs["path"],
    level=_numeric_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Logging initialised → %s (level=%s)", log_configs["path"], _log_level_name)


# load tokenizer and model.
# ---------------------------------------------------------------------
logging.info(f"Loading tokenizer for {args.model} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

logging.info(f"Loading model on device={model_configs['device']} ...")
pipe_kw: Dict[str, Any] = dict(
    task="text-generation",
    model=args.model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map=model_configs['device']
)

hf_pipeline = pipeline(**pipe_kw)
param_m = sum(p.numel() for p in hf_pipeline.model.parameters()) // 1_000_000_000
sema = asyncio.Semaphore(model_configs["max_concurrency"])


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


def build_prompt(msgs: List[Message]) -> str:
    try:
        return tokenizer.apply_chat_template(
            [m.model_dump() for m in msgs],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        logging.warning("[WARN] tokenizer.apply_chat_template failed - using fallback prompt template.")
        return "\n".join(f"{m.role}: {m.content}" for m in msgs) + "\nassistant:"


async def run_gen(prompt: str, gcfg: Dict[str, Any]) -> str:
    loop = asyncio.get_event_loop()
    async with sema:
        out = await loop.run_in_executor(
            None,
            lambda: hf_pipeline(prompt, **gcfg, do_sample=True)[0]["generated_text"],
        )
    return out[len(prompt):]


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.model != args.model:
        raise HTTPException(400, "Model name mismatch")
    gcfg = {
        "temperature":    req.temperature    or model_configs["generation"]["temperature"],
        "top_p":          req.top_p          or model_configs["generation"]["top_p"],
        "max_new_tokens": req.max_new_tokens or model_configs["generation"]["max_new_tokens"],
    }
    prompt = build_prompt(req.messages)
    try:
        completion = await run_gen(prompt, gcfg)
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
        "model": args.model,
    }


@app.get("/v1/models")
def meta():
    return {
        "name": args.model,
        "device": model_configs["device"],
        "max_concurrency": model_configs["max_concurrency"],
        "generation_defaults": model_configs["generation"],
        "param_count": f'{int(param_m)}B',
    }


if __name__ == "__main__":
    logging.info(f"Ready - {int(param_m)}B params | max concurrency = {model_configs['max_concurrency']}")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level=_log_level_name,
        )
    except Exception:
        logging.exception("Fatal error in main loop")
        raise