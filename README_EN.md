# OpenDify

[![CN](https://img.shields.io/badge/CN-中文版本-red?style=flat&logo=github)](README.md)

## Introduction

This project is a FastAPI application that converts the Dify API into an OpenAI-compatible API. This proxy service allows you to interact with the Dify service using an OpenAI client, while performing protocol conversion between the two. This version has been optimized for performance, including:

1.  Using `ujson` to replace the standard `json`, improving JSON serialization speed.
2.  Globally reusing `AsyncClient` to avoid repeatedly creating connection pools.
3.  Using `io.StringIO` for streaming responses to improve memory efficiency.
4.  Precompiling Base64 mapping tables to reduce runtime overhead.
5.  TTL caching of application information to reduce frequent API calls.

## Dependency Installation

Use `pip` to install project dependencies:

```bash
pip install -r requirements.txt
```

## Running

1.  **Configure environment variables:**

    *   `VALID_API_KEYS`: A list of valid Dify API keys, separated by commas.
    *   `CONVERSATION_MEMORY_MODE`: Conversation memory mode, defaults to `1`.
    *   `DIFY_API_BASE`: The base URL of the Dify API.
    *   `TIMEOUT`: Request timeout duration, defaults to `30.0` seconds.
    *   `SERVER_HOST`: The host the service listens on, defaults to `127.0.0.1`.
    *   `SERVER_PORT`: The port the service listens on, defaults to `8000`.

    You can configure these parameters through a `.env` file or system environment variables. For example:

    ```
    VALID_API_KEYS=your_api_key_1,your_api_key_2
    DIFY_API_BASE=https://your_dify_api_base
    ```

2.  **Run the FastAPI application:**

    ```bash
    python -m uvicorn app:app --reload --host 127.0.0.1 --port 1234
    ```

    Replace `127.0.0.1` with the host you want to listen on, and `1234` with the port you want to listen on.

## API Key Configuration

You need to configure the Dify API key in the environment variable `VALID_API_KEYS`.

## API Usage Example

You can use the following `test.py` file to verify that the service has started successfully:

```python
import openai
from openai import OpenAI

# Create an OpenAI client instance
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="sk-abc123"  # You can use any value
)

# Use the new API call method
response = client.chat.completions.create(
    model="AgentCoder",  # Note: Use the name of your Dify application
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'content') and delta.content is not None:
        print(delta.content, end="", flush=True)
```

Make sure to modify the `base_url` to your service address, and modify the `api_key` and `model` parameters as needed.

Run the `test.py` file. If the result is output correctly, it means that the service has started successfully.
