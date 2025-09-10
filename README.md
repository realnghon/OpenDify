# OpenDify

[![EN](https://img.shields.io/badge/EN-English%20Version-blue?style=flat&logo=github)](README_EN.md)

## 简介

OpenDify 是一个 **高性能的 FastAPI 代理服务**，可以将 [Dify](https://dify.ai) API 转换为 **OpenAI 兼容格式 API**。
借助 OpenDify，您可以用任何 OpenAI 风格的 SDK 或客户端直接访问 Dify 应用，无需修改现有基于 OpenAI 的代码或工作流。

适用于希望在保持 OpenAI 客户端生态的同时，接入 Dify 多模态 AI 能力的开发者。

---

## ✨ 功能特性

- **完全 OpenAI API 兼容** —— 支持 `/v1/chat/completions` 与 `/v1/models` 接口
- **支持流式响应** —— 从 Dify 到客户端的低延迟数据流转发
- **多模态消息支持** —— 可处理 `text` 与 `image_url` 类型消息
- **工具调用格式转换** —— 将 Dify 工具调用事件转换为 OpenAI 函数调用格式
- **自动文件上传** —— 支持远程 URL 和 Base64 图片上传至 Dify
- **全局 HTTP 连接池** —— 保持长连接并启用 HTTP/2，提升吞吐量
- **环境变量可配置** —— 通过 `.env` 文件完全控制运行参数

---

## 🛠 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境
在项目根目录创建 `.env` 文件：
```env
VALID_API_KEYS=your_proxy_access_key_1,your_proxy_access_key_2
DIFY_API_KEYS=your_dify_api_key
DIFY_API_BASE=https://your_dify_instance_base
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

### 3. 启动服务
**开发模式**
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**生产模式**
```bash
python app.py
```

---

## 🚀 使用示例 (Python OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your_proxy_access_key"
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

---

## 🔧 配置项说明

| 环境变量               | 说明                                  | 默认值       |
|------------------------|---------------------------------------|--------------|
| `VALID_API_KEYS`       | 允许访问代理的 API Key 列表           | `-`          |
| `DIFY_API_KEYS`        | 用于请求 Dify API 的密钥               | `-`          |
| `DIFY_API_BASE`        | Dify API 的基础 URL                   | `-`          |
| `TIMEOUT`              | 请求超时时间（秒）                     | `30.0`       |
| `SERVER_HOST`          | 代理服务监听主机                       | `127.0.0.1`  |
| `SERVER_PORT`          | 代理服务监听端口                       | `8000`       |

---

## 📡 API 接口

- **`POST /v1/chat/completions`** — 聊天补全接口（流式、非流式均支持）
- **`GET /v1/models`** — 获取 Dify 应用映射到 OpenAI 模型 ID 的列表

---

## 📈 性能提示

- **流式优化** —— 首包即发，降低用户等待延迟
- **连接池复用** —— 启用 HTTP/2 提升高并发性能
- **工具调用安全解析** —— 使用 Pydantic 确保 `tool_calls` 格式与 OpenAI 兼容
- **缓存策略** —— TTL 缓存减少重复的 API 请求

---

## 📜 许可
MIT — 免费使用、修改和分发
