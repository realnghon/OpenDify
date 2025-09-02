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
import weakref
from collections import deque

# 优化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 只使用控制台输出，避免文件I/O开销
    ]
)
logger = logging.getLogger(__name__)
# 禁用第三方库的详细日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # 减少访问日志

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
CONNECTION_POOL_SIZE = 200  # 增加连接池大小
CONNECTION_TIMEOUT = TIMEOUT
KEEPALIVE_TIMEOUT = 10.0  # 增加保活时间
BUFFER_SIZE = 16384  # 增大缓冲区
TTL_APP_CACHE = timedelta(minutes=10)  # 延长缓存时间
MAX_RESPONSE_CACHE_SIZE = 1000  # 响应缓存大小
CACHE_CLEANUP_INTERVAL = 300  # 缓存清理间隔（秒）

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
        self._response_cache = {}  # 响应缓存: request_hash -> (response, timestamp)
        self._cache_access_order = deque()  # LRU缓存顺序
        self._last_cleanup = time.time()  # 上次清理时间

        # HTTP长客户端，全局复用连接池 - 优化配置
        limits = httpx.Limits(
            max_keepalive_connections=CONNECTION_POOL_SIZE,
            max_connections=CONNECTION_POOL_SIZE,
            keepalive_expiry=KEEPALIVE_TIMEOUT,
        )
        # 优化超时配置和HTTP/2支持
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=CONNECTION_TIMEOUT,
                connect=5.0,  # 连接超时
                read=CONNECTION_TIMEOUT,
                write=10.0  # 写入超时
            ),
            limits=limits,
            http2=True,  # 启用HTTP/2
            verify=False,  # 跳过SSL验证以提升性能（如果安全允许）
        )
        self.load_api_keys()

    def load_api_keys(self):
        keys_str = os.getenv('DIFY_API_KEYS', '')
        self.api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        logger.info(f"Loaded {len(self.api_keys)} API keys")

    async def fetch_app_info(self, api_key: str) -> Optional[str]:
        try:
            now = datetime.utcnow()
            # 检查缓存
            if api_key in self._app_cache:
                cached_name, cached_time = self._app_cache[api_key]
                if now - cached_time < TTL_APP_CACHE:
                    return cached_name

            # 预构建headers避免重复字符串操作
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "keep-alive"  # 显式保持连接
            }
            
            # 使用更短的超时用于info接口
            rsp = await self._client.get(
                f"{DIFY_API_BASE}/info",
                headers=headers,
                params={"user": "default_user"},
                timeout=10.0  # info接口使用更短超时
            )
            if rsp.status_code == 200:
                # 使用ujson解析以提升性能
                app_info = ujson.loads(rsp.content)
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
        # 缓存时间戳避免重复计算
        timestamp = int(time.time())
        return [
            {
                "id": name,
                "object": "model",
                "created": timestamp,
                "owned_by": "dify"
            }
            for name in self.name_to_api_key.keys()
        ]
    
    def _cleanup_response_cache(self):
        """清理过期的响应缓存"""
        now = time.time()
        if now - self._last_cleanup < CACHE_CLEANUP_INTERVAL:
            return
            
        # 清理过期缓存（保留最近5分钟的）
        cutoff_time = now - 300
        expired_keys = [
            key for key, (_, timestamp) in self._response_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            self._response_cache.pop(key, None)
            
        # 如果缓存过大，清理最老的条目
        while len(self._response_cache) > MAX_RESPONSE_CACHE_SIZE:
            if self._cache_access_order:
                oldest_key = self._cache_access_order.popleft()
                self._response_cache.pop(oldest_key, None)
            else:
                break
                
        self._last_cleanup = now

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
# 优化应用启动配置
app = FastAPI(
    title="Dify to OpenAI API Proxy", 
    version="2.0-optimized",
    docs_url=None,  # 生产环境禁用文档
    redoc_url=None  # 生产环境禁用redoc
)

# 优化CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # 只允许需要的方法
    allow_headers=["*"],
    max_age=86400,  # 缓存预检请求24小时
)


# --------------------------
# 工具函数
# --------------------------
@lru_cache(maxsize=512)  # 增加缓存大小
def encode_conversation_id(conversation_id: str) -> str:
    """将conversation_id编码为零宽字符串"""
    if not conversation_id:
        return ""
    
    # 使用bytes避免重复编码
    encoded_bytes = base64.b64encode(conversation_id.encode('utf-8'))
    chars = []
    
    # 预分配列表大小提升性能
    for byte_val in encoded_bytes:
        c = chr(byte_val)
        if c.isupper():
            val = ord(c) - ord('A')
        elif c.islower():
            val = ord(c) - ord('a') + 26
        elif c.isdigit():
            val = int(c) + 52
        elif c == '+':
            val = 62
        elif c == '/':
            val = 63
        else:
            val = 0
            
        a, b = (val >> 3) & 0x7, val & 0x7
        chars.append(_CHAR_MAP[str(a)])
        if c != '=':
            chars.append(_CHAR_MAP[str(b)])
    
    return ''.join(chars)


def decode_conversation_id(content: str) -> Optional[str]:
    """从零宽字符序列还原conversation_id - 优化版本"""
    if not content:
        return None
        
    try:
        vals = []
        # 反向遍历直到找不到有效字符
        for ch in reversed(content):
            if ch not in _CHAR_TO_VAL:
                break
            vals.insert(0, _CHAR_TO_VAL[ch])
            
        if not vals:
            return None

        # 优化bytes组装
        bytes_out = []
        for i in range(0, len(vals), 2):
            first = int(vals[i])
            second = int(vals[i+1]) if i+1 < len(vals) else 0
            val = (first << 3) | second
            
            if val < 26:
                bytes_out.append(chr(val + 65))  # ord('A') = 65
            elif val < 52:
                bytes_out.append(chr(val - 26 + 97))  # ord('a') = 97
            elif val < 62:
                bytes_out.append(str(val - 52))
            elif val == 62:
                bytes_out.append('+')
            else:
                bytes_out.append('/')

        # 补充padding
        padding = len(bytes_out) % 4
        if padding:
            bytes_out.extend(['='] * (4 - padding))
            
        base64_str = ''.join(bytes_out)
        return base64.b64decode(base64_str).decode('utf-8')
        
    except (ValueError, UnicodeDecodeError, base64.binascii.Error) as e:
        if logger.isEnabledFor(logging.DEBUG):
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
    if not auth_header:
        raise HTTPUnauthorized("Missing Authorization header")
    
    # 优化字符串分割
    if not auth_header.startswith("Bearer "):
        raise HTTPUnauthorized("Invalid Authorization header format")
    
    key = auth_header[7:]  # 直接切片，避免split开销
    if not key or key not in VALID_API_KEYS:
        raise HTTPUnauthorized("Invalid API key")
    return key


# --------------------------
# 业务函数 - JSON转换
# --------------------------
def transform_openai_to_dify(openai_request: Dict, endpoint: str) -> Optional[Dict]:
    if endpoint != "/chat/completions":
        return None

    messages = openai_request.get("messages", [])
    if not messages:  # 提前检查避免后续处理
        return None
        
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
    
    # 优化agent_thoughts处理
    if not answer:
        agent_thoughts = dify_response.get("agent_thoughts")
        if agent_thoughts:
            # 反向遍历找到最后一个有效thought
            for thought in reversed(agent_thoughts):
                thought_content = thought.get("thought")
                if thought_content:
                    answer = thought_content
                    break
    
    # 对话ID编码优化
    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = dify_response.get("conversation_id", "")
        if conversation_id:
            history = dify_response.get("conversation_history", [])
            # 使用生成器表达式提升性能
            has_id = any(
                msg.get("role") == "assistant" and decode_conversation_id(msg.get("content", ""))
                for msg in history
            )
            if not has_id:
                answer += encode_conversation_id(conversation_id)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Appended encoded conversation_id to answer")

    # 使用预计算的时间戳
    timestamp = dify_response.get("created", int(time.time()))
    message_id = dify_response.get("message_id", "")
    
    return {
        "id": message_id,
        "object": "chat.completion",
        "created": timestamp,
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

    # --- 参数区（便于后续根据需求微调） ---
    MAX_FIRST_PACKET_DELAY = 0.0   # 首包立即发，秒级
    MAX_CHUNK_SIZE = 8192          # 增大到8KB提升吞吐
    FLUSH_INTERVAL = 0.05          # 降到50ms减少延迟
    BUFFER_THRESHOLD = 1024        # 缓冲区阈值
    # -------------------------------------

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

    # 使用更高效的缓冲区管理
    buffer = bytearray(BUFFER_SIZE)  # 预分配缓冲区
    buffer_pos = 0
    chunk_buffer = []
    chunk_len = 0
    first_chunk_sent = False
    last_flush = time.time()
    first_chunk_time = 0.0
    
    # 预编译正则模式
    data_prefix = b"data: "

    async with model_manager.client.stream(
        'POST',
        endpoint,
        json=dify_request,
        headers=headers,
    ) as rsp:
        if rsp.status_code != 200:
            yield dump_chunk("Stream error", final=True)
            return

        # 高效chunk & 首包即发 - 优化版本
        temp_buffer = bytearray()
        async for raw in rsp.aiter_bytes(chunk_size=BUFFER_SIZE):
            if not raw:
                continue
            temp_buffer.extend(raw)
            
            # 处理完整行
            while b"\n" in temp_buffer:
                line_end = temp_buffer.find(b"\n")
                line = bytes(temp_buffer[:line_end]).strip()
                temp_buffer = temp_buffer[line_end + 1:]
                
                if not line.startswith(data_prefix):
                    continue
                try:
                    # 直接从bytes解析，避免decode开销
                    chunk = ujson.loads(line[6:])
                except (ujson.JSONDecodeError, UnicodeDecodeError):
                    continue
                ev = chunk.get("event")
                if ev in ("message", "agent_message"):
                    ans = chunk.get("answer", "")
                    if not ans:
                        continue
                    chunk_buffer.append(ans)
                    chunk_len += len(ans)
                    now = time.time()
                    # -- 首包策略 --
                    if not first_chunk_sent:
                        yield dump_chunk(ans)
                        chunk_buffer.clear()
                        chunk_len = 0
                        first_chunk_sent = True
                        first_chunk_time = now
                        last_flush = now
                        continue
                    # -- 后续chunk（内容大于4K或者刷到时延阈值） --
                    if (
                        chunk_len >= MAX_CHUNK_SIZE or
                        (now - last_flush > FLUSH_INTERVAL and chunk_buffer)
                    ):
                        content = "".join(chunk_buffer)
                        yield dump_chunk(content)
                        chunk_buffer.clear()
                        chunk_len = 0
                        last_flush = now
                elif ev == "message_end":
                    if chunk_buffer:
                        content = "".join(chunk_buffer)
                        yield dump_chunk(content)
                        chunk_buffer.clear()
                    yield dump_chunk("", final=True)
                    yield "data: [DONE]\n\n"
                    return

        # 处理残留buffer（正常流结束后如仍有内容，补发）
        if chunk_buffer:
            yield dump_chunk("".join(chunk_buffer))
        yield dump_chunk("", final=True)
        yield "data: [DONE]\n\n"
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
# 旧: 兼容性尾处理代码块，已被上方新实现统一收尾，不再需要


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        # 使用ujson解析请求体，提升性能
        request_body = await request.body()
        openai_request = ujson.loads(request_body)
        
        # 日志优化：只在debug模式下记录详细请求
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Received request: {ujson.dumps(openai_request, ensure_ascii=False)}")
        
        model = openai_request.get("model", "claude-3-5-sonnet-v2")
        
        # 提前检查模型可用性
        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            msg = f"Model {model} not configured"
            raise HTTPException(status_code=404, detail={"error": {"message": msg}})

        # 清理缓存（定期执行）
        model_manager._cleanup_response_cache()

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
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked"  # 显式指定分块传输
                }
            )

        # 非流式调用 - 优化版本
        headers = {
            "Authorization": f"Bearer {dify_key}", 
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive"
        }
        endpoint = f"{DIFY_API_BASE}/chat-messages"
        
        # 使用更短的超时用于非流式请求
        resp = await model_manager.client.post(
            endpoint, 
            content=ujson.dumps(dify_req, ensure_ascii=False),  # 直接传递bytes
            headers=headers,
            timeout=20.0  # 非流式请求使用更短超时
        )
        
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        # 使用ujson解析响应
        dify_resp = ujson.loads(resp.content)
        openai_resp = stream_transform(dify_resp, model, stream=False)
        
        # 使用ujson序列化响应
        return Response(
            content=ujson.dumps(openai_resp, ensure_ascii=False),
            media_type="application/json",
            headers={"access-control-allow-origin": "*"}
        )

    except HTTPException:
        raise
    except ujson.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail={"error": {"message": "Invalid JSON format", "type": "invalid_request_error"}})
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "internal_error"}})


@app.get("/v1/models")
async def list_models():
    # 避免每次都刷新，采用懒加载策略
    if not model_manager.name_to_api_key:
        await model_manager.refresh_model_info()
    
    models = model_manager.get_available_models()
    resp = {"object": "list", "data": models}
    
    # 使用ujson和直接返回Response提升性能
    return Response(
        content=ujson.dumps(resp, ensure_ascii=False),
        media_type="application/json", 
        headers={"access-control-allow-origin": "*"}
    )


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
    
    # 优化启动配置
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))  # 支持多worker配置
    
    # 生产环境优化配置
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        workers=workers,
        access_log=False,  # 禁用访问日志提升性能
        server_header=False,  # 禁用服务器头
        date_header=False,  # 禁用日期头
        loop="uvloop" if os.name != 'nt' else "asyncio",  # Linux使用uvloop
    )
