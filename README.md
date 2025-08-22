# FastAPI Dify 代理服务 - 性能优化版

[![EN](https://img.shields.io/badge/EN-English%20Version-blue?style=flat&logo=github)](README_EN.md)

## 简介

本项目是一个 FastAPI 应用，用于将 Dify API 转换为 OpenAI 兼容的 API。此代理服务允许您使用 OpenAI 客户端与 Dify 服务进行交互，并在两者交互时进行协议转换。此版本在性能上进行了优化，包括：

1.  使用 `ujson` 替换标准 `json`，提升 JSON 序列化速度。
2.  全局复用 `AsyncClient`，避免重复创建连接池。
3.  流式响应使用 `io.StringIO` 提升内存效率。
4.  预编译 Base64 映射表，减少运行时开销。
5.  TTL 缓存应用信息，减少频繁 API 调用。

## 依赖安装

使用 `pip` 安装项目依赖：

```bash
pip install -r requirements.txt
```

## 运行

1.  **配置环境变量：**

    *   `VALID_API_KEYS`:  有效的 Dify API Key 列表，多个 Key 使用逗号分隔。
    *   `CONVERSATION_MEMORY_MODE`:  会话记忆模式，默认为 `1`。
    *   `DIFY_API_BASE`:  Dify API 的基础 URL。
    *   `TIMEOUT`:  请求超时时间，默认为 `30.0` 秒。
    *   `SERVER_HOST`:  服务监听的 Host，默认为 `127.0.0.1`。
    *   `SERVER_PORT`:  服务监听的端口，默认为 `8000`。

    您可以通过 `.env` 文件或系统环境变量来配置这些参数。例如：

    ```
    VALID_API_KEYS=your_api_key_1,your_api_key_2
    DIFY_API_BASE=https://your_dify_api_base
    ```

2.  **运行 FastAPI 应用：**

    ```bash
    python -m uvicorn app:app --reload --host 127.0.0.1 --port 1234
    ```

    将 `127.0.0.1` 替换为您希望监听的 Host，`1234` 替换为您希望监听的端口。

    将 `127.0.0.1` 替换为您希望监听的 Host，`1234` 替换为您希望监听的端口。

## API 使用示例

您可以使用以下 `test.py` 文件来验证服务是否启动成功：

```python
import openai
from openai import OpenAI

# 创建OpenAI客户端实例
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="sk-abc123"  # 可以使用任意值
)

# 使用新的API调用方式
response = client.chat.completions.create(
    model="AgentCoder",  # 注意：使用 Dify 应用的名称
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'content') and delta.content is not None:
        print(delta.content, end="", flush=True)
```

请确保将 `base_url` 修改为您的服务地址，并根据需要修改 `api_key` 和 `model` 参数。

运行 `test.py` 文件，如果能够正确输出结果，则说明服务启动成功。
```

## API Key 配置

需要在环境变量 `VALID_API_KEYS` 中配置 Dify API Key。
