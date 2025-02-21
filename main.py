import json
import logging
import asyncio
from flask import Flask, request, Response, stream_with_context
import httpx
import time
from dotenv import load_dotenv
import os
import ast

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置httpx的日志级别为WARNING，减少不必要的输出
logging.getLogger("httpx").setLevel(logging.WARNING)

# 加载环境变量
load_dotenv()

# 获取会话ID记忆功能开关配置
ENABLE_CONVERSATION_MEMORY = os.getenv('ENABLE_CONVERSATION_MEMORY', 'true').lower() == 'true'

class DifyModelManager:
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}  # 应用名称到API Key的映射
        self.api_key_to_name = {}  # API Key到应用名称的映射
        self.load_api_keys()

    def load_api_keys(self):
        """从环境变量加载API Keys"""
        api_keys_str = os.getenv('DIFY_API_KEYS', '')
        if api_keys_str:
            self.api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
            logger.info(f"Loaded {len(self.api_keys)} API keys")

    async def fetch_app_info(self, api_key):
        """获取Dify应用信息"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = await client.get(
                    f"{DIFY_API_BASE}/info",
                    headers=headers,
                    params={"user": "default_user"}
                )
                
                if response.status_code == 200:
                    app_info = response.json()
                    return app_info.get("name", "Unknown App")
                else:
                    logger.error(f"Failed to fetch app info for API key: {api_key[:8]}...")
                    return None
        except Exception as e:
            logger.error(f"Error fetching app info: {str(e)}")
            return None

    async def refresh_model_info(self):
        """刷新所有应用信息"""
        self.name_to_api_key.clear()
        self.api_key_to_name.clear()
        
        for api_key in self.api_keys:
            app_name = await self.fetch_app_info(api_key)
            if app_name:
                self.name_to_api_key[app_name] = api_key
                self.api_key_to_name[api_key] = app_name
                logger.info(f"Mapped app '{app_name}' to API key: {api_key[:8]}...")

    def get_api_key(self, model_name):
        """根据模型名称获取API Key"""
        return self.name_to_api_key.get(model_name)

    def get_available_models(self):
        """获取可用模型列表"""
        return [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify"
            }
            for name in self.name_to_api_key.keys()
        ]

# 创建模型管理器实例
model_manager = DifyModelManager()

# 从环境变量获取API基础URL
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")

app = Flask(__name__)

def get_api_key(model_name):
    """根据模型名称获取对应的API密钥"""
    api_key = model_manager.get_api_key(model_name)
    if not api_key:
        logger.warning(f"No API key found for model: {model_name}")
    return api_key

def transform_openai_to_dify(openai_request, endpoint):
    """将OpenAI格式的请求转换为Dify格式"""
    
    if endpoint == "/chat/completions":
        messages = openai_request.get("messages", [])
        stream = openai_request.get("stream", False)
        
        # 尝试从历史消息中提取conversation_id
        conversation_id = None
        if len(messages) > 1:
            # 遍历历史消息，找到最近的assistant消息
            for msg in reversed(messages[:-1]):  # 除了最后一条消息
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # 尝试解码conversation_id
                    conversation_id = decode_conversation_id(content)
                    if conversation_id:
                        break
        
        dify_request = {
            "inputs": {},
            "query": messages[-1]["content"] if messages else "",
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": conversation_id,
            "user": openai_request.get("user", "default_user")
        }

        return dify_request
    
    return None

def transform_dify_to_openai(dify_response, model="claude-3-5-sonnet-v2", stream=False):
    """将Dify格式的响应转换为OpenAI格式"""
    
    if not stream:
        answer = dify_response.get("answer", "")
        
        # 只在启用会话记忆功能时处理conversation_id
        if ENABLE_CONVERSATION_MEMORY:
            conversation_id = dify_response.get("conversation_id", "")
            history = dify_response.get("conversation_history", [])
            
            # 检查历史消息中是否已经有会话ID
            has_conversation_id = False
            if history:
                for msg in history:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if decode_conversation_id(content) is not None:
                            has_conversation_id = True
                            break
            
            # 只在新会话且历史消息中没有会话ID时插入
            if conversation_id and not has_conversation_id:
                logger.info(f"[Debug] Inserting conversation_id: {conversation_id}, history_length: {len(history)}")
                encoded = encode_conversation_id(conversation_id)
                answer = answer + encoded
                logger.info(f"[Debug] Response content after insertion: {repr(answer)}")
        
        return {
            "id": dify_response.get("message_id", ""),
            "object": "chat.completion",
            "created": dify_response.get("created", int(time.time())),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }]
        }
    else:
        # 流式响应的转换在stream_response函数中处理
        return dify_response

def create_openai_stream_response(content, message_id, model="claude-3-5-sonnet-v2"):
    """创建OpenAI格式的流式响应"""
    return {
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": content
            },
            "finish_reason": None
        }]
    }

def encode_conversation_id(conversation_id):
    """将conversation_id编码为不可见的字符序列"""
    if not conversation_id:
        return ""
    
    # 使用Base64编码减少长度
    import base64
    encoded = base64.b64encode(conversation_id.encode()).decode()
    
    # 使用8种不同的零宽字符表示3位数字
    # 这样可以将编码长度进一步减少
    char_map = {
        '0': '\u200b',  # 零宽空格
        '1': '\u200c',  # 零宽非连接符
        '2': '\u200d',  # 零宽连接符
        '3': '\ufeff',  # 零宽非断空格
        '4': '\u2060',  # 词组连接符
        '5': '\u180e',  # 蒙古语元音分隔符
        '6': '\u2061',  # 函数应用
        '7': '\u2062',  # 不可见乘号
    }
    
    # 将Base64字符串转换为八进制数字
    result = []
    for c in encoded:
        # 将每个字符转换为8进制数字（0-7）
        if c.isalpha():
            if c.isupper():
                val = ord(c) - ord('A')
            else:
                val = ord(c) - ord('a') + 26
        elif c.isdigit():
            val = int(c) + 52
        elif c == '+':
            val = 62
        elif c == '/':
            val = 63
        else:  # '='
            val = 0
            
        # 每个Base64字符可以产生2个3位数字
        first = (val >> 3) & 0x7
        second = val & 0x7
        result.append(char_map[str(first)])
        if c != '=':  # 不编码填充字符的后半部分
            result.append(char_map[str(second)])
    
    return ''.join(result)

def decode_conversation_id(content):
    """从消息内容中解码conversation_id"""
    try:
        # 零宽字符到3位数字的映射
        char_to_val = {
            '\u200b': '0',  # 零宽空格
            '\u200c': '1',  # 零宽非连接符
            '\u200d': '2',  # 零宽连接符
            '\ufeff': '3',  # 零宽非断空格
            '\u2060': '4',  # 词组连接符
            '\u180e': '5',  # 蒙古语元音分隔符
            '\u2061': '6',  # 函数应用
            '\u2062': '7',  # 不可见乘号
        }
        
        # 提取最后一段零宽字符序列
        space_chars = []
        for c in reversed(content):
            if c not in char_to_val:
                break
            space_chars.append(c)
        
        if not space_chars:
            return None
            
        # 将零宽字符转换回Base64字符串
        space_chars.reverse()
        base64_chars = []
        for i in range(0, len(space_chars), 2):
            first = int(char_to_val[space_chars[i]], 8)
            if i + 1 < len(space_chars):
                second = int(char_to_val[space_chars[i + 1]], 8)
                val = (first << 3) | second
            else:
                val = first << 3
                
            # 转换回Base64字符
            if val < 26:
                base64_chars.append(chr(val + ord('A')))
            elif val < 52:
                base64_chars.append(chr(val - 26 + ord('a')))
            elif val < 62:
                base64_chars.append(str(val - 52))
            elif val == 62:
                base64_chars.append('+')
            else:
                base64_chars.append('/')
                
        # 添加Base64填充
        padding = len(base64_chars) % 4
        if padding:
            base64_chars.extend(['='] * (4 - padding))
            
        # 解码Base64字符串
        import base64
        base64_str = ''.join(base64_chars)
        return base64.b64decode(base64_str).decode()
        
    except Exception as e:
        logger.debug(f"Failed to decode conversation_id: {e}")
        return None

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        openai_request = request.get_json()
        logger.info(f"Received request: {json.dumps(openai_request, ensure_ascii=False)}")
        
        model = openai_request.get("model", "claude-3-5-sonnet-v2")
        
        # 验证模型是否支持
        api_key = get_api_key(model)
        if not api_key:
            error_msg = f"Model {model} is not supported. Available models: {', '.join(model_manager.name_to_api_key.keys())}"
            logger.error(error_msg)
            return {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }, 404
            
        dify_request = transform_openai_to_dify(openai_request, "/chat/completions")
        
        if not dify_request:
            logger.error("Failed to transform request")
            return {
                "error": {
                    "message": "Invalid request format",
                    "type": "invalid_request_error",
                }
            }, 400

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stream = openai_request.get("stream", False)
        dify_endpoint = f"{DIFY_API_BASE}/chat-messages"
        logger.info(f"Sending request to Dify endpoint: {dify_endpoint}, stream={stream}")

        if stream:
            def generate():
                client = httpx.Client(timeout=None)
                
                def flush_chunk(chunk_data):
                    """Helper function to flush chunks immediately"""
                    return chunk_data.encode('utf-8')
                
                def calculate_delay(buffer_size):
                    """
                    根据缓冲区大小动态计算延迟
                    buffer_size: 缓冲区中剩余的字符数量
                    """
                    if buffer_size > 30:  # 缓冲区内容较多，快速输出
                        return 0.001  # 5ms延迟
                    elif buffer_size > 20:  # 中等数量，适中速度
                        return 0.002  # 10ms延迟
                    elif buffer_size > 10:  # 较少内容，稍慢速度
                        return 0.01  # 20ms延迟
                    else:  # 内容很少，使用较慢的速度
                        return 0.02  # 30ms延迟
                
                def send_char(char, message_id):
                    """Helper function to send single character"""
                    openai_chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": char
                            },
                            "finish_reason": None
                        }]
                    }
                    chunk_data = f"data: {json.dumps(openai_chunk)}\n\n"
                    return flush_chunk(chunk_data)
                
                # 初始化缓冲区
                output_buffer = []
                
                try:
                    with client.stream(
                        'POST',
                        dify_endpoint,
                        json=dify_request,
                        headers={
                            **headers,
                            'Accept': 'text/event-stream',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive'
                        }
                    ) as response:
                        generate.message_id = None
                        buffer = ""
                        
                        for raw_bytes in response.iter_raw():
                            if not raw_bytes:
                                continue
                                
                            try:
                                buffer += raw_bytes.decode('utf-8')
                                
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    line = line.strip()
                                    
                                    if not line or not line.startswith('data: '):
                                        continue
                                        
                                    try:
                                        json_str = line[6:]
                                        dify_chunk = json.loads(json_str)
                                        
                                        if dify_chunk.get("event") == "message" and "answer" in dify_chunk:
                                            current_answer = dify_chunk["answer"]
                                            if not current_answer:
                                                continue
                                                
                                            message_id = dify_chunk.get("message_id", "")
                                            if not generate.message_id:
                                                generate.message_id = message_id
                                            
                                            # 将当前批次的字符添加到输出缓冲区
                                            for char in current_answer:
                                                output_buffer.append((char, generate.message_id))
                                            
                                            # 根据缓冲区大小动态调整输出速度
                                            while output_buffer:
                                                char, msg_id = output_buffer.pop(0)
                                                yield send_char(char, msg_id)
                                                # 根据剩余缓冲区大小计算延迟
                                                delay = calculate_delay(len(output_buffer))
                                                time.sleep(delay)
                                            
                                            # 立即继续处理下一个请求
                                            continue
                                        
                                        elif dify_chunk.get("event") == "message_end":
                                            # 快速输出剩余内容
                                            while output_buffer:
                                                char, msg_id = output_buffer.pop(0)
                                                yield send_char(char, msg_id)
                                                time.sleep(0.001)  # 固定使用最小延迟快速输出剩余内容
                                            
                                            # 只在启用会话记忆功能时处理conversation_id
                                            if ENABLE_CONVERSATION_MEMORY:
                                                conversation_id = dify_chunk.get("conversation_id")
                                                history = dify_chunk.get("conversation_history", [])
                                                
                                                has_conversation_id = False
                                                if history:
                                                    for msg in history:
                                                        if msg.get("role") == "assistant":
                                                            content = msg.get("content", "")
                                                            if decode_conversation_id(content) is not None:
                                                                has_conversation_id = True
                                                                break
                                                
                                                # 只在新会话且历史消息中没有会话ID时插入
                                                if conversation_id and not has_conversation_id:
                                                    logger.info(f"[Debug] Inserting conversation_id in stream: {conversation_id}")
                                                    encoded = encode_conversation_id(conversation_id)
                                                    logger.info(f"[Debug] Stream encoded content: {repr(encoded)}")
                                                    for char in encoded:
                                                        yield send_char(char, generate.message_id)
                                            
                                            final_chunk = {
                                                "id": generate.message_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": "stop"
                                                }]
                                            }
                                            yield flush_chunk(f"data: {json.dumps(final_chunk)}\n\n")
                                            yield flush_chunk("data: [DONE]\n\n")
                                        
                                    except json.JSONDecodeError as e:
                                        logger.error(f"JSON decode error: {str(e)}")
                                        continue
                                        
                            except Exception as e:
                                logger.error(f"Error processing chunk: {str(e)}")
                                continue

                finally:
                    client.close()

            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache, no-transform',
                    'Connection': 'keep-alive',
                    'Transfer-Encoding': 'chunked',
                    'X-Accel-Buffering': 'no',
                    'Content-Encoding': 'none'
                },
                direct_passthrough=True
            )
        else:
            async def sync_response():
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            dify_endpoint,
                            json=dify_request,
                            headers=headers
                        )
                        
                        if response.status_code != 200:
                            error_msg = f"Dify API error: {response.text}"
                            logger.error(f"Request failed: {error_msg}")
                            return {
                                "error": {
                                    "message": error_msg,
                                    "type": "api_error",
                                    "code": response.status_code
                                }
                            }, response.status_code

                        dify_response = response.json()
                        logger.info(f"Received response from Dify: {json.dumps(dify_response, ensure_ascii=False)}")
                        logger.info(f"[Debug] Response content: {repr(dify_response.get('answer', ''))}")
                        openai_response = transform_dify_to_openai(dify_response, model=model)
                        conversation_id = dify_response.get("conversation_id")
                        if conversation_id:
                            # 在响应头中传递conversation_id
                            return Response(
                                json.dumps(openai_response),
                                content_type='application/json',
                                headers={
                                    'Conversation-Id': conversation_id
                                }
                            )
                        else:
                            return openai_response
                except httpx.RequestError as e:
                    error_msg = f"Failed to connect to Dify: {str(e)}"
                    logger.error(error_msg)
                    return {
                        "error": {
                            "message": error_msg,
                            "type": "api_error",
                            "code": "connection_error"
                        }
                    }, 503

            return asyncio.run(sync_response())

    except Exception as e:
        logger.exception("Unexpected error occurred")
        return {
            "error": {
                "message": str(e),
                "type": "internal_error",
            }
        }, 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回可用的模型列表"""
    logger.info("Listing available models")
    
    # 刷新模型信息
    asyncio.run(model_manager.refresh_model_info())
    
    # 获取可用模型列表
    available_models = model_manager.get_available_models()
    
    response = {
        "object": "list",
        "data": available_models
    }
    logger.info(f"Available models: {json.dumps(response, ensure_ascii=False)}")
    return response

if __name__ == '__main__':
    # 启动时初始化模型信息
    asyncio.run(model_manager.refresh_model_info())
    
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 5000))
    logger.info(f"Starting server on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
