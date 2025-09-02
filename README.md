# OpenDify

[![EN](https://img.shields.io/badge/EN-English%20Version-blue?style=flat&logo=github)](README_EN.md)

## 简介

一个高性能的 FastAPI 代理服务，将 Dify API 转换为 OpenAI 兼容的 API。让您能够使用 OpenAI 客户端无缝对接 Dify 服务。

### ⚡ 性能特性

- **高速 JSON 处理**: 使用 `ujson` 提升序列化性能
- **连接池优化**: 全局复用 HTTP 连接，支持 HTTP/2
- **智能缓存**: TTL 缓存减少重复 API 调用
- **流式优化**: 首包即发，降低响应延迟
- **内存高效**: 优化缓冲区管理和内存使用

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件：

```env
VALID_API_KEYS=your_dify_api_key_1,your_dify_api_key_2
DIFY_API_BASE=https://your_dify_api_base
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

### 3. 启动服务

```bash
# 开发环境
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

# 生产环境
python app.py
```

## 使用示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your_api_key"
)

# 流式对话
response = client.chat.completions.create(
    model="你的Dify应用名称",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `VALID_API_KEYS` | Dify API 密钥列表（逗号分隔） | - |
| `DIFY_API_BASE` | Dify API 基础地址 | - |
| `CONVERSATION_MEMORY_MODE` | 会话记忆模式 | 1 |
| `TIMEOUT` | 请求超时时间（秒） | 30.0 |
| `SERVER_HOST` | 服务监听地址 | 127.0.0.1 |
| `SERVER_PORT` | 服务监听端口 | 8000 |

## API 端点

- `POST /v1/chat/completions` - 聊天补全接口
- `GET /v1/models` - 获取可用模型列表

## 性能建议

- **生产部署**: 使用 `uvicorn` 的生产模式启动
- **Linux/Mac**: 自动启用 `uvloop` 事件循环优化（Windows 下会自动跳过）
- **高并发**: 调整连接池大小和超时配置
- **内存优化**: 定期清理缓存，避免内存泄漏
