# OpenDify

[![CN](https://img.shields.io/badge/CN-中文版本-red?style=flat&logo=github)](README.md)

## Introduction

OpenDify is a **high‑performance FastAPI proxy service** that converts the [Dify](https://dify.ai) API into an **OpenAI‑compatible API format**.
It allows you to interact with your Dify apps using any OpenAI‑style SDK or client, streamlining integration without modifying existing OpenAI‑based workflows.

Ideal for developers who want Dify’s multi‑modal AI capabilities with existing OpenAI client tooling.

---

## ✨ Features

- **Full OpenAI API Compatibility** — Mirrors `/v1/chat/completions` and `/v1/models` endpoints.
- **Streaming Response Support** — Low‑latency streaming from Dify to clients.
- **Multi‑Modal Message Support** — Handles `text` & `image_url` message content seamlessly.
- **Tool Call Conversion** — Converts Dify tool call events to OpenAI function call format.
- **Automatic File Upload** — Supports remote URL & base64 image uploads to Dify.
- **Global HTTP Connection Pool** — Persistent connections with HTTP/2 enabled.
- **Environment‑Driven Config** — Full control via `.env` file.

---

## 🛠 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
VALID_API_KEYS=your_proxy_access_key_1,your_proxy_access_key_2
DIFY_API_KEYS=your_dify_api_key
DIFY_API_BASE=https://your_dify_instance_base
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

### 3. Start the Service
**Development Mode**
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**Production Mode**
```bash
python app.py
```

---

## 🚀 Example Usage (Python OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your_proxy_access_key"
)

# Streaming Chat Completion
response = client.chat.completions.create(
    model="YourDifyAppName",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 🔧 Configuration Variables

| Variable              | Description                                 | Default   |
|-----------------------|---------------------------------------------|-----------|
| `VALID_API_KEYS`      | API keys allowed to access this proxy       | `-`       |
| `DIFY_API_KEYS`       | Keys to authenticate with the Dify API      | `-`       |
| `DIFY_API_BASE`       | Base URL of your Dify API service           | `-`       |
| `TIMEOUT`             | Request timeout in seconds                  | `30.0`    |
| `SERVER_HOST`         | Proxy host binding                          | `127.0.0.1` |
| `SERVER_PORT`         | Proxy port                                  | `8000`    |

---

## 📡 API Endpoints

- **`POST /v1/chat/completions`** — Chat completion endpoint (streaming + non‑stream)
- **`GET /v1/models`** — Lists available Dify apps/models mapped to OpenAI model IDs

---

## 📈 Performance Notes

- **Streaming Design** — First‑packet‑send strategy reduces initial latency.
- **Connection Pool** — Shared across requests with HTTP/2 for higher throughput.
- **Tool Call Safety Parsing** — Pydantic model ensures tool calls are compatible.
- **Cache Strategies** — TTL caching avoids unnecessary Dify API calls.

---

## 📜 License
MIT — free to use, modify, and distribute.
