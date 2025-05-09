# LocaHF: Lightweight OpenAI-compatible FastAPI wrapper for any Hugging-Face chat model
works with Qwen, Llama-2/3, Mistral, and every LLMs on Hugging Face.

🚀 LocaHF Highlights 🚀
----------
* **Preloaded Model:** Model is loaded once at startup, saving time for users—no need to reload for each request.
* **Unified API Interface:** Simplifies your code by unifying local model calls with API calls.
* **Guaranteed GPU Control:** Locks specified GPU `(--device)` to prevent conflicts.


Function
----------
✓ `/v1/chat/completions` reproduces the OpenAI Chat API (no streaming)  
✓ `/v1/models` returns live metadata about the running model  

Quick start
-----------
1.  Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

2.  Add or edit the optional **configs.json**

    ```json
    "models": {
        "Qwen/Qwen2.5-0.5B": {
            "generation": {
                "temperature": 0.0,
                "top_p": 0.0,
                "max_new_tokens": 128
            }
        }
    }
    ```

3.  Launch
    ```bash
    nohup python main.py --model Qwen/Qwen2.5-0.5B --port 8000 2>&1 >/dev/null &
    ```

4.  Talk to it with the official **openai** client. See [client.py](./client.py).

TODO
-----------
1. Support vLLM with image input.
2. Concurrency test.
3. Welcome any suggestion.