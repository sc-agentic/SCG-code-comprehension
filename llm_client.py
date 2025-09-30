import httpx

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

timeout: httpx.Timeout = httpx.Timeout(200.0)
client: httpx.AsyncClient = httpx.AsyncClient(timeout=timeout)


async def call_llm(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = await client.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()
