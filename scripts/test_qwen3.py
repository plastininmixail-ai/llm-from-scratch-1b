import httpx, json, sys
payload = {
    "model": "qwen3-nothink",
    "messages": [
        {"role": "user", "content": "Привет, скажи 2+2"}
    ],
    "max_tokens": 30
}
r = httpx.post("http://127.0.0.1:11435/v1/chat/completions", json=payload, timeout=30.0)
print(r.status_code)
print(r.text[:500])
