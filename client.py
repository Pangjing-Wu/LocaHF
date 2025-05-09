from openai import OpenAI
import requests

client = OpenAI(
    base_url="http://localhost:8000/v1", 
    api_key="sk-local"
)

# chat
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B",
    messages=[{"role": "user", "content": "Hello!, Who are you?"}],
    temperature=0.8,
    top_p=0.9,
    frequency_penalty=1.0,
    max_tokens=128
)
print(resp.choices[0].message.content)

# model info
print(requests.get("http://localhost:8000/v1/models").json())
