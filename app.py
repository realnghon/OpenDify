import asyncio
import logging
import time
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import httpx
import os
import orjson
import io
import gzip
from typing import Dict, List, Optional, AsyncGenerator, Any
from functools import lru_cache
import base64
import hashlib
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
if not DIFY_API_BASE:
    logger.error("DIFY_API_BASE environment variable is not set!")
    raise ValueError("DIFY_API_BASE is required")
TIMEOUT = float(os.getenv("TIMEOUT", 30.0))

# ========================
# 优化常量
# ========================
CONNECTION_POOL_SIZE = 100
CONNECTION_TIMEOUT = TIMEOUT
KEEPALIVE_TIMEOUT = 5.0
BUFFER_SIZE = 8192
TTL_APP_CACHE = timedelta(minutes=5)

from collections import deque
from threading import Lock
import weakref
from cachetools import TTLCache, LRUCache

# ========================
# 对象池优化
# ========================
class ObjectPool:
    """通用对象池，减少频繁内存分配"""
    def __init__(self, factory, max_size=100):
        self.factory = factory
        self.pool = deque(maxlen=max_size)
        self.lock = Lock()
    
    def get(self):
        with self.lock:
            return self.pool.popleft() if self.pool else self.factory()
    
    def put(self, obj):
        try:
            if hasattr(obj, 'reset'):
                obj.reset()
        except Exception as e:
            logger.warning(f"Failed to reset object in pool: {e}")
        with self.lock:
            self.pool.append(obj)

class BufferPool:
    """字节数组池，用于流式处理"""
    def __init__(self, size=BUFFER_SIZE, max_count=50):
        self.size = size
        self.pool = deque([bytearray(size) for _ in range(max_count)], maxlen=max_count)
        self.lock = Lock()
    
    def get(self):
        with self.lock:
            if self.pool:
                buf = self.pool.popleft()
                buf.clear()
                return buf
            return bytearray(self.size)
    
    def put(self, buf):
        if len(buf) <= self.size * 2:  # 避免过大的buffer占用内存
            with self.lock:
                self.pool.append(buf)

# 全局对象池
buffer_pool = BufferPool()
chunk_list_pool = ObjectPool(lambda: [], max_size=50)

class CacheManager:
    """缓存管理器，支持多层缓存策略"""
    def __init__(self):
        # 请求级缓存（较小，但命中率高）
        self.request_cache = LRUCache(maxsize=1000)
        # 应用信息缓存（TTL缓存）
        self.app_cache = TTLCache(maxsize=200, ttl=300)  # 5分钟
        # 为不同缓存使用独立的锁，提高并发性能
        self.request_lock = Lock()
        self.app_lock = Lock()
    
    def get_request_cache(self, key):
        with self.request_lock:
            return self.request_cache.get(key)
    
    def set_request_cache(self, key, value):
        with self.request_lock:
            self.request_cache[key] = value
    
    def get_app_cache(self, key):
        with self.app_lock:
            return self.app_cache.get(key)
    
    def set_app_cache(self, key, value):
        with self.app_lock:
            self.app_cache[key] = value

# 全局缓存管理器
cache_manager = CacheManager()

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
        # 移除旧的_app_cache，使用全局缓存管理器

        # HTTP长客户端，全局复用连接池
        limits = httpx.Limits(
            max_keepalive_connections=CONNECTION_POOL_SIZE,
            max_connections=CONNECTION_POOL_SIZE,
            keepalive_expiry=KEEPALIVE_TIMEOUT,
        )
        # 启用HTTP/2支持
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=CONNECTION_TIMEOUT),
            limits=limits,
            http2=True,  # 启用HTTP/2
        )
        self.load_api_keys()

    def load_api_keys(self):
        keys_str = os.getenv('DIFY_API_KEYS', '')
        self.api_keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        logger.info(f"Loaded {len(self.api_keys)} API keys")

    async def fetch_app_info(self, api_key: str) -> Optional[str]:
        try:
            # 使用新的缓存管理器
            cached_name = cache_manager.get_app_cache(api_key)
            if cached_name:
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
                cache_manager.set_app_cache(api_key, app_name)
                return app_name
            else:
                logger.error(f"Fetch app info failed: {rsp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error in fetch_app_info: {e}")
            return None

    async def refresh_model_info(self):
        """异步刷新模型映射，使用并行处理和重试机制"""
        self.name_to_api_key.clear()
        self.api_key_to_name.clear()

        # 使用asyncio.gather并行处理多个请求
        async def fetch_with_retry(key, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return await self.fetch_app_info(key)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to fetch app info for key {key[:8]}... after {max_retries} attempts: {e}")
                        return None
                    await asyncio.sleep(0.1 * (attempt + 1))  # 指数退避
            return None

        # 并行处理所有API密钥
        tasks = [fetch_with_retry(key) for key in self.api_keys]
        names = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, name in zip(self.api_keys, names):
            if name and not isinstance(name, Exception):
                self.name_to_api_key[name] = key
                self.api_key_to_name[key] = name
                logger.debug(f"Mapped app '{name}' to key: {key[:8]}...")
            elif isinstance(name, Exception):
                logger.error(f"Exception for key {key[:8]}...: {name}")

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

# 添加gzip压缩中间件（必须在CORS之前添加）
app.add_middleware(GZipMiddleware, minimum_size=1000)  # 超过1KB的响应才压缩

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 生成请求的缓存键
def generate_cache_key(openai_request: Dict, model: str) -> str:
    """生成请求的缓存键"""
    # 移除可能变化的字段，只保留影响响应的关键字段
    cache_data = {
        "model": model,
        "messages": openai_request.get("messages", []),
        "functions": openai_request.get("functions", []),
        "function_call": openai_request.get("function_call"),
        "temperature": openai_request.get("temperature", 1.0),
        "max_tokens": openai_request.get("max_tokens"),
        "user": openai_request.get("user", "default_user")
    }
    cache_str = orjson.dumps(cache_data, sort_keys=True).decode()
    return hashlib.md5(cache_str.encode()).hexdigest()


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


def format_functions_for_dify(functions: List[Dict], function_call: Optional[Dict] = None) -> str:
    """将 OpenAI functions 转换为 Dify 可理解的文本格式"""
    functions_text = "你是一个能够调用函数的AI助手。以下是可用的函数列表：\n\n"
    
    for func in functions:
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        
        functions_text += f"函数名: {name}\n"
        functions_text += f"描述: {description}\n"
        
        if parameters.get("properties"):
            functions_text += "参数:\n"
            for param_name, param_info in parameters["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                required = param_name in parameters.get("required", [])
                req_text = "（必需）" if required else "（可选）"
                functions_text += f"  - {param_name} ({param_type}){req_text}: {param_desc}\n"
        functions_text += "\n"
    
    if function_call:
        if isinstance(function_call, dict) and function_call.get("name"):
            functions_text += f"请使用 {function_call['name']} 函数来处理用户请求。\n\n"
        elif function_call == "auto":
            functions_text += "请根据用户请求选择合适的函数进行调用。\n\n"
    
    functions_text += "当你需要调用函数时，请使用以下 JSON 格式：\n"
    functions_text += "```json\n{\"function_call\": {\"name\": \"函数名\", \"arguments\": \"参数JSON字符串\"}}\n```\n\n"
    
    return functions_text


import re
import json

# 预编译正则表达式，提高性能
_FUNCTION_CALL_PATTERN = re.compile(r'```json\s*({.*?})\s*```', re.DOTALL)

def parse_function_call_from_response(content: str) -> Optional[Dict]:
    """从 Dify 响应中解析 function call"""
    # 使用预编译的正则表达式
    matches = _FUNCTION_CALL_PATTERN.findall(content)
    
    for match in matches:
        try:
            data = json.loads(match)
            if "function_call" in data:
                func_call = data["function_call"]
                if "name" in func_call and "arguments" in func_call:
                    # 解析 arguments 字符串
                    args_str = func_call["arguments"]
                    if isinstance(args_str, str):
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse function arguments: {args_str}")
                            args = {}
                    else:
                        args = args_str
                    
                    return {
                        "name": func_call["name"],
                        "arguments": json.dumps(args) if args else "{}"
                    }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON in function call: {e}")
            continue
    
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
    functions = openai_request.get("functions", [])
    function_call = openai_request.get("function_call")
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
        
        # 处理 function calling
        if functions:
            functions_text = format_functions_for_dify(functions, function_call)
            if system_content:
                user_query = f"系统指令: {system_content}\n\n{functions_text}\n\n用户问题: {user_query}"
            else:
                user_query = f"{functions_text}\n\n用户问题: {user_query}"
        
        return {
            "inputs": {},
            "query": user_query,
            "response_mode": "streaming" if stream else "blocking",
            "user": openai_request.get("user", "default_user")
        }


def stream_transform(dify_response: Dict, model: str, stream: bool, has_functions: bool = False) -> Dict:
    if stream:
        return dify_response

    answer = dify_response.get("answer", "")
    if not answer and "agent_thoughts" in dify_response:
        thoughts = dify_response.get("agent_thoughts", [])
        for t in reversed(thoughts):
            if t.get("thought"):
                answer = t.get("thought")
                break
    
    # 检查是否有 function call
    function_call = None
    if has_functions and answer:
        function_call = parse_function_call_from_response(answer)
    
    if CONVERSATION_MEMORY_MODE == 2:
        conversation_id = dify_response.get("conversation_id", "")
        history = dify_response.get("conversation_history", [])
        has_id = any(
            msg.get("role") == "assistant" and decode_conversation_id(msg.get("content", ""))
            for msg in history
        )
        if conversation_id and not has_id and not function_call:
            answer += encode_conversation_id(conversation_id)
            logger.debug(f"Appended encoded conversation_id to answer")

    response = {
        "id": dify_response.get("message_id", ""),
        "object": "chat.completion",
        "created": dify_response.get("created", int(time.time())),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer if not function_call else None
            },
            "finish_reason": "function_call" if function_call else "stop"
        }]
    }
    
    # 如果有 function call，添加到 message 中
    if function_call:
        response["choices"][0]["message"]["function_call"] = function_call
        # 移除 JSON 代码块，保留其他内容
        cleaned_content = _FUNCTION_CALL_PATTERN.sub('', answer).strip()
        if cleaned_content:
            response["choices"][0]["message"]["content"] = cleaned_content
    
    return response


# --------------------------
# 核心路由 - /chat/completions
# --------------------------
async def stream_response(dify_request: Dict, api_key: str, model: str, has_functions: bool = False) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    endpoint = f"{DIFY_API_BASE}/chat-messages"
    message_id = None

    # --- 参数区（便于后续根据需求微调） ---
    MAX_FIRST_PACKET_DELAY = 0.0   # 首包立即发，秒级
    MAX_CHUNK_SIZE = 4096          # 后续包累积，2KB~4KB较优
    FLUSH_INTERVAL = 0.10          # 100ms内至少强制发一次
    # -------------------------------------

    def dump_chunk(content: str, final=False, function_call=None):
        chunk_data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }]
        }
        
        if function_call:
            chunk_data["choices"][0]["delta"]["function_call"] = function_call
            chunk_data["choices"][0]["finish_reason"] = "function_call"
        elif final:
            chunk_data["choices"][0]["finish_reason"] = "stop"
        elif content:
            chunk_data["choices"][0]["delta"]["content"] = content
        
        return f"data: {orjson.dumps(chunk_data).decode()}\n\n"

    # 使用对象池获取buffer和chunk_buffer
    buffer = buffer_pool.get()
    chunk_buffer = chunk_list_pool.get()
    
    try:
        chunk_len = 0
        first_chunk_sent = False
        last_flush = time.time()
        first_chunk_time = 0.0
        full_response = ""  # 用于检测 function call

        async with model_manager.client.stream(
            'POST',
            endpoint,
            json=dify_request,
            headers=headers,
        ) as rsp:
            if rsp.status_code != 200:
                error_detail = f"HTTP {rsp.status_code}"
                try:
                    error_content = await rsp.aread()
                    error_detail += f": {error_content.decode()[:200]}"  # 限制错误消息长度
                except Exception:
                    pass
                logger.error(f"Dify API error: {error_detail}")
                yield dump_chunk(f"Stream error: {error_detail}", final=True)
                return

            # 高效chunk & 首包即发
            async for raw in rsp.aiter_bytes():
                if not raw:
                    continue
                buffer.extend(raw)
                # 拆分所有完整行，剩余半行保留待下次完整收集
                while b"\n" in buffer:
                    line_end = buffer.find(b"\n")
                    line = buffer[:line_end]
                    buffer[:] = buffer[line_end + 1:]
                    line = line.strip()
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        chunk = orjson.loads(line[6:])
                    except (orjson.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse JSON chunk: {e}")
                        continue
                    ev = chunk.get("event")
                    if ev in ("message", "agent_message"):
                        ans = chunk.get("answer", "")
                        if not ans:
                            continue
                        full_response += ans  # 累积完整响应
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
                        # 检查是否有 function call
                        function_call = None
                        if has_functions and full_response:
                            function_call = parse_function_call_from_response(full_response)
                        
                        if chunk_buffer:
                            content = "".join(chunk_buffer)
                            if function_call:
                                # 移除 JSON 代码块
                                content = _FUNCTION_CALL_PATTERN.sub('', content).strip()
                            if content:
                                yield dump_chunk(content)
                            chunk_buffer.clear()
                        
                        if function_call:
                            yield dump_chunk("", function_call=function_call)
                        else:
                            yield dump_chunk("", final=True)
                        yield "data: [DONE]\n\n"
                        return

            # 处理残留buffer（正常流结束后如仍有内容，补发）
            if chunk_buffer:
                yield dump_chunk("".join(chunk_buffer))
            yield dump_chunk("", final=True)
            yield "data: [DONE]\n\n"
    except asyncio.TimeoutError:
        logger.error("Stream request timeout")
        yield dump_chunk("Request timeout", final=True)
        yield "data: [DONE]\n\n"
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        yield dump_chunk("Connection error", final=True)
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception(f"Unexpected error in stream_response: {e}")
        yield dump_chunk(f"Internal error: {str(e)}", final=True)
        yield "data: [DONE]\n\n"
    finally:
        # 归还对象到池中
        buffer_pool.put(buffer)
        chunk_list_pool.put(chunk_buffer)
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
        openai_request = await request.json()
        logger.info(f"Received request: {orjson.dumps(openai_request).decode()}")
        model = openai_request.get("model", "claude-3-5-sonnet-v2")
        functions = openai_request.get("functions", [])
        has_functions = bool(functions)
        stream = openai_request.get("stream", False)
        
        # 检查缓存（非流式请求且无function calling）
        if not stream and not has_functions:
            cache_key = generate_cache_key(openai_request, model)
            cached_response = cache_manager.get_request_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for request: {cache_key[:8]}...")
                return JSONResponse(cached_response)
        
        dify_key = model_manager.get_api_key(model)
        if not dify_key:
            msg = f"Model {model} not configured"
            raise HTTPException(status_code=404, detail={"error": {"message": msg}})

        dify_req = transform_openai_to_dify(openai_request, "/chat/completions")
        if not dify_req:
            raise HTTPException(status_code=400, detail={"error": {"message": "Invalid format"}})

        if stream:
            return StreamingResponse(
                stream_response(dify_req, dify_key, model, has_functions),
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
        openai_resp = stream_transform(dify_resp, model, stream=False, has_functions=has_functions)
        
        # 将响应放入缓存（非流式且无function calling）
        if not stream and not has_functions:
            cache_manager.set_request_cache(cache_key, openai_resp)
            logger.debug(f"Cached response for request: {cache_key[:8]}...")
        
        return JSONResponse(openai_resp)

    except asyncio.TimeoutError:
        logger.error("Non-stream request timeout")
        raise HTTPException(status_code=504, detail={"error": {"message": "Request timeout", "type": "timeout"}})
    except httpx.ConnectError as e:
        logger.error(f"Connection error in non-stream request: {e}")
        raise HTTPException(status_code=503, detail={"error": {"message": "Service unavailable", "type": "connection_error"}})
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
