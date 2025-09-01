# OpenDify

[![CN](https://img.shields.io/badge/CN-中文版本-red?style=flat&logo=github)](README.md)

## Introduction

This project is a FastAPI application that converts the Dify API into an OpenAI-compatible API, **with extended support for Function Calling**. This proxy service allows you to interact with the Dify service using an OpenAI client, while performing protocol conversion between the two. This version has been optimized for performance, including:

- ✅ Using `orjson` to replace the standard `json`, improving JSON serialization speed.
- ✅ Globally reusing `AsyncClient` to avoid repeatedly creating connection pools.
- ✅ Using `io.StringIO` for streaming responses to improve memory efficiency.
- ✅ Precompiling Base64 mapping tables to reduce runtime overhead.
- ✅ TTL caching of application information to reduce frequent API calls.

## Quick Start

### Dependency Installation

Use `pip` to install project dependencies:

```bash
pip install -r requirements.txt
```

### Environment Configuration

Copy the `.env.example` file and rename it to `.env`. Fill in the configuration values as needed. Example:

```env
VALID_API_KEYS=your_api_key_1,your_api_key_2
DIFY_API_BASE=https://your_dify_api_base
CONVERSATION_MEMORY_MODE=1
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

### Running the Application

Run the FastAPI application with `uvicorn`:

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Replace `127.0.0.1` and `8000` with your desired host and port.

## Usage Instructions

### API Key Configuration
Set your Dify API Key in the `VALID_API_KEYS` environment variable.

### API Usage Example
See the API usage example in the `test.py` file.

## Feature Details

### How It Works

#### Function Definition Transformation
When an OpenAI client sends a request with the `functions` parameter, OpenDify converts these function definitions (including name, description, and parameters) into a text format that the Dify model can understand. This information is appended as system instructions to the Dify request's `query`.

#### Function Call Parsing
When the Dify model generates a response and identifies that it needs to call a function, it outputs the function call instruction in a specific JSON format within the response. OpenDify parses this `function_call` from the Dify response and converts it back into an OpenAI-compatible format.

#### Streaming and Non-Streaming Support
Function Calling is supported in both streaming and non-streaming responses. In streaming mode, the function call information is sent as a separate `delta` chunk.
