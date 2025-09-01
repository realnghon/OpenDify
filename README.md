# OpenDify

[![EN](https://img.shields.io/badge/EN-English%20Version-blue?style=flat&logo=github)](README_EN.md)

## 简介

本项目是一个使用 FastAPI 构建的代理应用，可将 Dify 的 API 转换为 OpenAI 兼容的 API，**并扩展支持 Function Calling**。此代理服务允许您使用 OpenAI 客户端与 Dify 服务进行交互，并在两者交互时进行协议转换。此版本在性能上进行了多方面优化，包括：

- ✅ 使用 `orjson` 替换标准 `json`，显著提升 JSON 序列化速度。
- ✅ 全局复用 `AsyncClient`，避免重复创建连接池。
- ✅ 流式响应使用 `io.StringIO`，提高内存处理效率。
- ✅ 预编译 Base64 映射表，减少运行时开销。
- ✅ TTL 缓存应用信息，减少对 Dify API 的频繁调用。

## 快速开始

### 依赖安装

使用 `pip` 安装项目依赖：

```bash
pip install -r requirements.txt
```

### 环境配置

请复制 `.env.example` 文件并重命名为 `.env`，然后根据您的需求填写相应的配置信息。示例如下：

```env
VALID_API_KEYS=your_api_key_1,your_api_key_2
DIFY_API_BASE=https://your_dify_api_base
CONVERSATION_MEMORY_MODE=1
TIMEOUT=30.0
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

### 运行应用

使用 `uvicorn` 运行 FastAPI 应用：

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

将 `127.0.0.1` 替换为您希望监听的主机名，将 `8000` 替换为您希望监听的端口号。

## 使用说明

### API Key 配置
在环境变量 `VALID_API_KEYS` 中配置有效的 Dify API Key。

### API 使用示例
请在 `test.py` 文件中查看 API 使用示例。

## 特性详解

### 工作原理

#### 函数定义转换
当 OpenAI 客户端发送包含 `functions` 参数的请求时，OpenDify 会将这些函数定义（包括名称、描述和参数）转换为 Dify 模型能够理解的文本格式。这些信息会作为系统指令的一部分，附加到 Dify 请求的 `query` 中。

#### 函数调用解析
Dify 模型在生成响应时，如果识别出需要调用函数，会按照特定 JSON 格式在响应中输出函数调用指令。OpenDify 会从 Dify 的响应中解析出这些 `function_call`，并将其转换回 OpenAI 兼容的格式。

#### 流式与非流式支持
Function Calling 在流式和非流式响应中均受支持。在流式模式下，函数调用信息会以独立的 `delta` 块形式发送。
