import asyncio
import logging
import time
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import ujson
import io
from typing import Dict, List, Optional, AsyncGenerator, Any
from functools import lru_cache
import base64
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]
CONVERSATION_MEMORY_MODE = int(os.getenv('CONVERSATION_MEMORY_MODE', '1'))
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")
TIMEOUT = float(os.getenv("TIMEOUT", 30.0))

# ========================
# 优化常量
# ========================
CONNECTION_POOL_SIZE = 100
CONNECTION_TIMEOUT = TIMEOUT
KEEPALIVE_TIMEOUT = 5.0
BUFFER_SIZE = 8192
TTL_APP_CACHE = timedelta(minutes=5)

# 预编译零宽字符映射表
_CHAR_MAP = {
    '0': '\u200b',
    '1': '\u200c',
    '2': '\u200d',
    '3': '\ufeff',
    '4': '\u2060',
    '5': '\u180e',
    '6': '\u2061',
    '7': '\u2062',
}
_CHAR_TO_VAL = {v: k for k, v in _CHAR_MAP.items()}


class DifyModelManager:
    """单例管理Dify模型与API密钥映射"""
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}
        self.api_key_to_name = {}
        self._app_cache = {}  # api_key -> (app_name, cached_time)

        # HTTP长客户端，全局复用连接池
        limits = httpx.Limits(
            max_keepalive_connections=CONNECTION_POOL_SIZE,
            max_connections=CONNECTION_POOL_SIZE,
            keepalive_expiry=KEEPALIVE_TIMEOUT,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=CONNECTION_TIMEOUT),
            limits=limits,
        )
        self.load_api_keys()

    def load_api_keys(self):
        keys_str = os.getenv('DIFY_API_KEYS', '')
        self.api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        logger.info(f"Loaded {len(self.api_keys)} API keys")

    async def fetch_app_info(self, api_key: str) -> Optional[str]:
        try:
            now = datetime.utcnow()
            if api_key in self._app_cache:
                cached_name, cached_time = self._app_cache[api_key]
                if now - cached_time < TTL_APP_CACHE:
                    return cached_name

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            rsp = await self._client.get(
                f"{DIFY_API_BASE}/info",
                headers=headers,
                params={"user": "default_user"}
            )
            if rsp.status_code == 200:
                app_info = rsp.json()
                app_name = app_info.get("name", "Unknown App")
                self._app_cache[api_key] = (app_name, now)
                return app_name
            else:
                logger.error(f"Fetch app info failed: {rsp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error in fetch_app_info: {e}")
            return None

    async def refresh_model_info(self):
        """异步刷新模型映射"""
        self.name_to_api_key.clear()
        self.api_key_to_name.clear()

        tasks = [self.fetch_app_info(key) for key in self.api_keys]
        names = await asyncio.gather(*tasks)
        for key, name in zip(self.api_keys, names):
            if name:
                self.name_to_api_key[name] = key
                self.api_key_to_name[key] = name
                logger.debug(f"Mapped app '{name}' to key: {key[:8]}...")

    def get_api_key(self, model: str) -> Optional[str]:
        return self.name_to_api_key.get(model)

    def get_available_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify"
            }
            for name in self.name_to_api_key.keys()
        ]

    async def close(self):
        await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """供全局使用同一个连接池"""
        return self._client


# --------------------------
# 全局单例
# --------------------------
model_manager = DifyModelManager()
app = FastAPI(title="Dify to OpenAI API Proxy", version="2.0-optimized")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# 工具函数
# --------------------------
@lru_cache
def encode_conversation_id(conversation_id: str) -> str:
    """将conversation_id编码为零宽字符串"""
    if not conversation_id:
        return ""
    encoded = base64.b64encode(conversation_id.encode()).decode()
    chars = []
    for c in encoded:
        val = (
            (ord(c) - ord('A')) if c.isupper() else
            (ord(c) - ord('a') + 26) if c.islower() else
            (int(c) + 52) if c.isdigit() else
            62 if c == '+' else 63 if c == '/' else 0
        )
        a, b = (val >> 3) & 0x7, val & 0x7
        chars.append(_CHAR_MAP[str(a)])
        if c != '=':
            chars.append(_CHAR_MAP[str(b)])
    return ''.join(chars)


def decode_conversation_id(content: str) -> Optional[str]:
    """从零宽字符序列还原conversation_id"""
    try:
        vals = []
        for ch in reversed(content):
            if ch not in _CHAR_TO_VAL:
                break
            vals.insert(0, _CHAR_TO_VAL[ch])
        if not vals:
            return None

        bytes_out = []
        for i in range(0, len(vals), 2):
            first = int(vals[i], 8)
            second = int(vals[i+1], 8) if i+1 < len(vals) else 0
            val = (first << 3) | second
            if val < 26:
                bytes_out.append(chr(val + ord('A')))
            elif val < 52:
                bytes_out.append(chr(val - 26 + ord('a')))
            elif val < 62:
                bytes_out.append(str(val - 52))
            elif val == 62:
                bytes_out.append('+')
            else:
                bytes_out.append('/')

        padding = len(bytes_out) % 4
        if padding:
            bytes_out.extend(['='] * (4 - padding))
        base64_str = ''.join(bytes_out)
        return base64.b64decode(base64_str).decode()
    except Exception as e:
        logger.debug(f"decode conversation_id failed: {e}")
        return None


# --------------------------
# 自定义异常
# --------------------------
class HTTPUnauthorized(HTTPException):
    def __init__(self, message):
        super().__init__(
            status_code=401,
            detail={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )


# --------------------------
# 依赖注入
# --------------------------
async def verify_api_key(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or len(parts := auth_header.split()) != 2 or parts[0].lower() != 'bearer':
        raise HTTPUnauthorized("Invalid Authorization header")
    key = parts[1]
    if key not in VALID_API_KEYS:
        raise HTTPUnauthorized("Invalid API key")
    return key


# --------------------------
# 业务函数 - JSON转换
# --------------------------
def transform_openai_to_dify(openai_request: Dict, endpoint: str) -> Optional[Dict]:
    if endpoint != "/chat/completions":
        return None

    messages = openai_request.get("messages", [])
    stream = openai_request.get("stream", False)
    system_content, user_query = "", ""

    # 提取system
    for m in messages:
        if m.get("role") == "system":
            system_content = m.get("content", "")
            break

    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = None
        if len(messages) > 1:
            for m in reversed(messages[:-1]):
                if m.get("role") == "assistant":
                    conversation_id = decode_conversation_id(m.get("content", ""))
                    if conversation_id:
                        break
        user_query = messages[-1]["content"] if messages and messages[-1].get("role") != "system" else ""
        if system_content and not conversation_id:
            user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"
        return {
            "inputs": {},
            "query": user_query,
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": conversation_id,
            "user": openai_request.get("user", "default_user")
        }
    else:
        # history模式
        user_query = messages[-1]["content"] if messages and messages[-1].get("role") != "system" else ""
        if len(messages) > 1:
            history_msg = []
            has_system = any(m.get("role") == "system" for m in messages[:-1])
            for m in messages[:-1]:
                role, content = m.get("role", ""), m.get("content", "")
                if role and content:
                    history_msg.append(f"{role}: {content}")
            if system_content and not has_system:
                history_msg.insert(0, f"system: {system_content}")
            if history_msg:
                history_txt = "\n\n".join(history_msg)
                user_query = f"<history>\n{history_txt}\n</history>\n\n用户当前问题: {user_query}"
        elif system_content:
            user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"
        return {
            "inputs": {},
            "query": user_query,
            "response_mode": "streaming" if stream else "blocking",
            "user": openai_request.get("user", "default_user")
        }


def stream_transform(dify_response: Dict, model: str, stream: bool) -> Dict:
    if stream:
        return dify_response

    answer = dify_response.get("answer", "")
    if not answer and "agent_thoughts" in dify_response:
        thoughts = dify_response.get("agent_thoughts", [])
        for t in reversed(thoughts):
            if t.get("thought"):
                answer = t.get("thought")
                break
    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = dify_response.get("conversation_id", "")
        history = dify_response.get("conversation_history", [])
        has_id = any(
            msg.get("role") == "assistant" and decode_conversation_id(msg.get("content", ""))
            for msg in history
        )
        if conversation_id and not has_id:
            answer += encode_conversation_id(conversation_id)
            logger.debug(f"Appended encoded conversation_id to answer")

    return {
        "id": dify_response.get("message_id", ""),
        "object": "chat.completion",
        "created": dify_response.get("created", int(time.time())),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }]
    }


# --------------------------
# 核心路由 - /chat/completions
# --------------------------
async def stream_response(dify_request: Dict, api_key: str, model: str) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    endpoint = f"{DIFY_API_BASE}/chat-messages"
    message_id = None

    # 使用高效StringIO缓冲
    buf = io.StringIO()
    last_flush = time.time()
    FLUSH_THRESHOLD = 0.05  # 50ms
    MAX_BUF_LEN = 256

    def dump_chunk(content: str, final=False):
        openai_chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if not final else {},
                "finish_reason": "stop" if final else None
            }]
        }
        return f"data: {ujson.dumps(openai_chunk)}\n\n"

    async with model_manager.client.stream(
        'POST',
        endpoint,
        json=dify_request,
        headers=headers,
    ) as rsp:
        if rsp.status_code != 200:
            yield dump_chunk("Stream error", final=True)
            return

        # 优化：使用原生字节缓冲，高效分块流处理，不再用低效字符串拼接
        buffer = bytearray()
        async for raw in rsp.aiter_bytes():
            if not raw:
                continue
            buffer.extend(raw)
            # 拆分所有完整行，剩余半行保留待下次完整收集
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                # 注意strip只处理bytes
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                try:
                    # 仅data字段处理，安全解码出字符串
                    chunk = ujson.loads(line[6:].decode("utf-8", "ignore"))
                except Exception:
                    continue
                ev = chunk.get("event")
                if ev == "message" or ev == "agent_message":
                    ans = chunk.get("answer", "")
                    if not ans:
                        continue
                    buf.write(ans)
                    if buf.tell() >= MAX_BUF_LEN or time.time() - last_flush > FLUSH_THRESHOLD:
                        msg = buf.getvalue()
                        buf.seek(0)
                        buf.truncate(0)
                        yield dump_chunk(msg)
                        last_flush = time.time()
                elif ev == "message_end":
                    remainder = buf.getvalue()
# =============================
# 性能说明与单元测试建议
# =============================
#
# 1. 本流式响应实现为「高吞吐大文本」以及「大并发」场景设计，所有拼接操作均在 bytes 层完成，
#    避免了 Python 字符串不可变所带来的频繁内存拷贝和 GC 问题。
# 2. 若项目未来迁移到 PyPy/JIT、更底层高性能方案（如 anyio 内置 AsyncLineReader），
#    这一实现模式仍然安全、并利于横向并发扩展。
# 3. 建议重点单测用例：
#    - 特大文本/超长分行
#    - 多行混杂部分乱码
#    - 行极端不落在块边界（需跨raw多次组装一行）
#    - 并发连接数 > 100 时资源占用与延迟表现
#    - 逐步悬停（首包延迟 < 50ms 测试）
# 4. 如需极致精简亦可替换为 anyio.streams.AsyncLineReader，但推荐优先用本更透明、兼容且可追踪的做法。
#
# —— by OpenDify Code Assistant
                    if remainder:
                        yield dump_chunk(remainder)
                    yield dump_chunk("", final=True)
                    yield "data: [DONE]\n\n"
                    return
        # 补充：如果流正常结束，处理buffer中剩余内容
        if buffer:
            # 可能还有残留未处理的 data: 行
            for line in buffer.split(b"\n"):
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                try:
                    chunk = ujson.loads(line[6:].decode("utf-8", "ignore"))
                except Exception:
                    continue
                ev = chunk.get("event")
                if ev == "message" or ev == "agent_message":
                    ans = chunk.get("answer", "")
                    if not ans:
                        continue
                    buf.write(ans)
            remainder = buf.getvalue()
            if remainder:
                yield dump_chunk(remainder)
            yield dump_chunk("", final=True)
            yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        openai_request = await request.json()
        logger.info(f"Received request: {ujson.dumps(openai_request, ensure_ascii=False)}")
        model = openai_request.get("model", "claude-3-5-sonnet-v2")
        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            msg = f"Model {model} not configured"
            raise HTTPException(status_code=404, detail={"error": {"message": msg}})

        dify_req = transform_openai_to_dify(openai_request, "/chat/completions")
        if not dify_req:
            raise HTTPException(status_code=400, detail={"error": {"message": "Invalid format"}})

        stream = openai_request.get("stream", False)
        if stream:
            return StreamingResponse(
                stream_response(dify_req, dify_key, model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # 非流式调用
        headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
        endpoint = f"{DIFY_API_BASE}/chat-messages"
        resp = await model_manager.client.post(endpoint, json=dify_req, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        dify_resp = resp.json()
        openai_resp = stream_transform(dify_resp, model, stream=False)
        return JSONResponse(openai_resp)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "internal_error"}})


@app.get("/v1/models")
async def list_models():
    await model_manager.refresh_model_info()
    models = model_manager.get_available_models()
    resp = {"object": "list", "data": models}
    return JSONResponse(resp, headers={"access-control-allow-origin": "*"})


# --------------------------
# 应用生命周期
# --------------------------
@app.on_event("startup")
async def startup():
    if not VALID_API_KEYS:
        logger.warning("VALID_API_KEYS 缺省，请先配置环境变量")
    await model_manager.refresh_model_info()
    logger.info("FastAPI Dify Proxy started")


@app.on_event("shutdown")
async def shutdown():
    await model_manager.close()
    logger.info("Server shutdown")


if __name__ == '__main__':
    import uvicorn
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
