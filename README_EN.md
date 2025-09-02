# OpenDify

[![CN](https://img.shields.io/badge/CN-中文版本-red?style=flat&logo=github)](README.md)

## Introduction

A high-performance FastAPI proxy service that converts Dify API to OpenAI-compatible API. Seamlessly integrate with Dify services using OpenAI clients.

### ⚡ Performance Features

- **Fast JSON Processing**: Using `ujson` for enhanced serialization performance
- **Connection Pool Optimization**: Global HTTP connection reuse with HTTP/2 support
- **Smart Caching**: TTL caching to reduce redundant API calls
- **Streaming Optimization**: First-packet-send strategy for reduced latency
- **Memory Efficiency**: Optimized buffer management and memory usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file:

```env
VALID_API_KEYS=your_dify_api_key_1,your_dify_api_key_2
DIFY_API_BASE=https://your_dify_api_base
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

### 3. Start Service

```bash
# Development
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Production
python app.py
```

## Usage Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your_api_key"
)

# Streaming chat
response = client.chat.completions.create(
    model="YourDifyAppName",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|--------|
| `VALID_API_KEYS` | Dify API keys list (comma-separated) | - |
| `DIFY_API_BASE` | Dify API base URL | - |
| `CONVERSATION_MEMORY_MODE` | Conversation memory mode | 1 |
| `TIMEOUT` | Request timeout (seconds) | 30.0 |
| `SERVER_HOST` | Service host | 127.0.0.1 |
| `SERVER_PORT` | Service port | 8000 |

## API Endpoints

- `POST /v1/chat/completions` - Chat completions interface
- `GET /v1/models` - List available models

## Performance Tips

- **Production Deployment**: Use `uvicorn` production mode for optimal performance
- **Linux/Mac**: Automatically enables `uvloop` event loop optimization (skipped on Windows)
- **High Concurrency**: Adjust connection pool size and timeout configurations
- **Memory Optimization**: Regular cache cleanup to prevent memory leaks
