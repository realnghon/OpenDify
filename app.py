import asyncio
import logging
import time
import uuid
import json
import base64
import tempfile
import os
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]
DIFY_API_KEYS = [key.strip() for key in os.getenv("DIFY_API_KEYS", "").split(",") if key]
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")
TIMEOUT = float(os.getenv("TIMEOUT", 30.0))

# === Data Models (Similar to Go structs) ===

class MessageImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "high"
    mime_type: Optional[str] = None

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[MessageImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None
    reasoning_content: Optional[str] = None

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction

class GeneralOpenAIRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream_options: Optional[Dict[str, Any]] = None

class DifyFile(BaseModel):
    type: str
    transfer_mode: str
    url: Optional[str] = None
    upload_file_id: Optional[str] = Field(None, alias="UploadFileId")

    class Config:
        populate_by_name = True

class DifyChatRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    query: str
    response_mode: str
    user: str
    files: Optional[List[DifyFile]] = None
    auto_generate_name: bool = False

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionsStreamResponseChoiceDelta(BaseModel):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    def set_content_string(self, content: str):
        self.content = content

    def get_content_string(self) -> str:
        return self.content or ""

    def set_reasoning_content(self, content: str):
        self.reasoning_content = content

class ChatCompletionsStreamResponseChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionsStreamResponseChoiceDelta
    finish_reason: Optional[str] = None

class ChatCompletionsStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionsStreamResponseChoice]
    usage: Optional[Usage] = None

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str

class OpenAITextResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class DifyChunkChatCompletionResponse(BaseModel):
    event: str
    task_id: Optional[str] = None
    id: Optional[str] = None
    answer: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    # 宽松解析 tool_calls，避免因字段不匹配被 Pydantic 丢掉
    tool_calls: Optional[Any] = None

    def get_tool_calls_as_openai(self) -> Optional[List[ToolCall]]:
        """将原始 tool_calls 字段安全转换为 OpenAI ToolCall 格式"""
        try:
            if not self.tool_calls:
                return None
            tool_calls_converted: List[ToolCall] = []
            for tool in self.tool_calls:
                # tool 可能是 dict 或已是 ToolCall
                if isinstance(tool, ToolCall):
                    tool_calls_converted.append(tool)
                elif isinstance(tool, dict):
                    func = tool.get("function", {})
                    arguments_val = func.get("arguments", "")
                    if not isinstance(arguments_val, str):
                        import json as _json
                        try:
                            arguments_val = _json.dumps(arguments_val, ensure_ascii=False)
                        except Exception:
                            arguments_val = str(arguments_val)
                    tool_calls_converted.append(ToolCall(
                        id=tool.get("id", ""),
                        type=tool.get("type", "function"),
                        function=ToolCallFunction(
                            name=func.get("name", ""),
                            arguments=arguments_val
                        )
                    ))
            return tool_calls_converted
        except Exception as e:
            import logging as _logging
            _logging.error(f"Failed to parse tool_calls: {e}")
            return None

class DifyChatCompletionResponse(BaseModel):
    conversation_id: str
    answer: str
    metadata: Dict[str, Any]

# === HTTP Client ===
http_client = httpx.AsyncClient(
    timeout=TIMEOUT,
    verify=False
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI Dify Proxy...")
    if not VALID_API_KEYS:
        logger.warning("VALID_API_KEYS not configured")
    yield
    logger.info("Shutting down...")
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Authentication ===
async def verify_api_key(request: Request) -> str:
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )

    api_key = auth[7:]
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )
    return api_key

# === File Upload Helper ===
async def upload_dify_file(image_url: str, user: str, dify_api_key: str) -> Optional[DifyFile]:
    """Upload base64 image to Dify and return file info"""
    try:
        # Extract base64 data
        if "," in image_url:
            base64_data = image_url.split(",")[1]
        else:
            base64_data = image_url

        # Decode base64
        decoded_data = base64.b64decode(base64_data)

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(decoded_data)
            temp_file.flush()

            # Prepare multipart form data
            files = {
                'file': ('image.jpg', decoded_data, 'image/jpeg')
            }
            data = {
                'user': user
            }
            headers = {
                'Authorization': f'Bearer {dify_api_key}'
            }

            # Upload to Dify
            upload_url = f"{DIFY_API_BASE}/files/upload"
            response = await http_client.post(
                upload_url,
                files=files,
                data=data,
                headers=headers
            )

            # Clean up temp file
            os.unlink(temp_file.name)

            if response.status_code == 200:
                result = response.json()
                return DifyFile(
                    type="image",
                    transfer_mode="local_file",
                    upload_file_id=result.get("id")
                )

    except Exception as e:
        logger.error(f"Failed to upload file: {e}")

    return None

# === Request Conversion ===
async def request_openai_to_dify(request: GeneralOpenAIRequest, dify_api_key: str) -> DifyChatRequest:
    """Convert OpenAI request to Dify request format"""
    user = request.user or str(uuid.uuid4())
    files = []
    content_builder = []

    for message in request.messages:
        role_prefix = {
            "system": "SYSTEM: ",
            "assistant": "ASSISTANT: ",
            "user": "USER: "
        }.get(message.role, f"{message.role.upper()}: ")

        if isinstance(message.content, str):
            content_builder.append(f"{role_prefix}\n{message.content}\n")
        else:
            # Handle complex message content
            text_parts = []
            for content_item in message.content:
                if content_item.type == "text":
                    text_parts.append(content_item.text or "")
                elif content_item.type == "image_url" and content_item.image_url:
                    image_url = content_item.image_url.url
                    if image_url.startswith("http"):
                        # Remote image
                        files.append(DifyFile(
                            type="image",
                            transfer_mode="remote_url",
                            url=image_url
                        ))
                    else:
                        # Base64 image - upload to Dify
                        uploaded_file = await upload_dify_file(image_url, user, dify_api_key)
                        if uploaded_file:
                            files.append(uploaded_file)

            if text_parts:
                content_builder.append(f"{role_prefix}\n{''.join(text_parts)}\n")

    query = "".join(content_builder)
    # Always use streaming mode for Dify - Agent Chat App doesn't support blocking mode
    response_mode = "streaming"

    return DifyChatRequest(
        inputs={},
        query=query,
        response_mode=response_mode,
        user=user,
        files=files if files else None,
        auto_generate_name=False
    )

# === Stream Response Conversion ===
def stream_response_dify_to_openai(dify_response: DifyChunkChatCompletionResponse, model: str, completion_id: str, created: int) -> Optional[ChatCompletionsStreamResponse]:
    """Convert Dify stream chunk to OpenAI format"""
    response = ChatCompletionsStreamResponse(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[]
    )

    choice = ChatCompletionsStreamResponseChoice(
        index=0,
        delta=ChatCompletionsStreamResponseChoiceDelta()
    )

    # Handle different Dify event types
    if dify_response.event.startswith("workflow_"):
        # Workflow events - add as reasoning content for debugging
        if dify_response.data and dify_response.data.get("workflow_id"):
            text = f"Workflow: {dify_response.data['workflow_id']}"
            if dify_response.event == "workflow_finished":
                text += f" {dify_response.data.get('status', '')}"
            choice.delta.set_reasoning_content(text + "\n")

    elif dify_response.event.startswith("node_"):
        # Node events - add as reasoning content for debugging
        if dify_response.data and dify_response.data.get("node_type"):
            text = f"Node: {dify_response.data['node_type']}"
            if dify_response.event == "node_finished":
                text += f" {dify_response.data.get('status', '')}"
            choice.delta.set_reasoning_content(text + "\n")

    elif dify_response.event in ["message", "agent_message"]:
        # Main message content
        content = dify_response.answer or ""

        # Handle special thinking tags
        if content == '<details style="color:gray;background-color: #f8f8f8;padding: 8px;border-radius: 4px;" open> <summary> Thinking... </summary>\n':
            content = "<think>"
        elif content == "</details>":
            content = "</think>"

        choice.delta.set_content_string(content)

    # Handle tool calls if present
    # 新的 ToolCall 转换逻辑（即便 tool_calls 是原始 dict 也可处理）
    tool_calls_safe = getattr(dify_response, "get_tool_calls_as_openai", lambda: None)()
    if tool_calls_safe:
        # 为了调试 Dify 返回的 tool_calls，这里打印原始 JSON 内容
        try:
            import logging as _logging
            _logging.getLogger(__name__).info(f"Raw tool_calls from Dify chunk: {getattr(dify_response, 'tool_calls', None)}")
        except Exception:
            pass
        choice.delta.tool_calls = tool_calls_safe

    response.choices.append(choice)
    return response

# === Stream Handler ===
async def dify_stream_handler(dify_response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
    """Handle Dify streaming response and convert to OpenAI format"""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    usage = Usage()

    if dify_response.status_code != 200:
        # Handle error response
        try:
            response_text = await dify_response.aread()
            error_content = json.loads(response_text.decode())
            error_msg = {
                "error": {
                    "message": error_content.get("message", f"Dify API error: Status {dify_response.status_code}"),
                    "type": error_content.get("type", "server_error"),
                    "code": error_content.get("code", dify_response.status_code)
                }
            }
        except Exception:
            error_msg = {
                "error": {
                    "message": f"Dify API error: Status {dify_response.status_code}",
                    "type": "server_error",
                    "code": dify_response.status_code
                }
            }
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        async for line in dify_response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue

            data = line[6:].strip()
            if data == "[DONE]":
                break

            try:
                chunk_data = json.loads(data)
                dify_chunk = DifyChunkChatCompletionResponse(**chunk_data)

                if dify_chunk.event == "message_end":
                    # Extract usage info
                    if dify_chunk.metadata and "usage" in dify_chunk.metadata:
                        usage_data = dify_chunk.metadata["usage"]
                        usage = Usage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0)
                        )

                    # Send final chunk with finish reason
                    final_response = ChatCompletionsStreamResponse(
                        id=completion_id,
                        created=created,
                        model=model,
                        choices=[ChatCompletionsStreamResponseChoice(
                            index=0,
                            delta=ChatCompletionsStreamResponseChoiceDelta(),
                            finish_reason="stop"
                        )],
                        usage=usage
                    )
                    yield f"data: {final_response.model_dump_json()}\n\n"
                    break

                elif dify_chunk.event == "error":
                    # Handle error event
                    error_msg = {
                        "error": {
                            "message": chunk_data.get("message", "Unknown error"),
                            "type": "server_error",
                            "code": chunk_data.get("code", 500)
                        }
                    }
                    yield f"data: {json.dumps(error_msg)}\n\n"
                    break

                else:
                    # Convert and send regular chunk
                    openai_response = stream_response_dify_to_openai(dify_chunk, model, completion_id, created)
                    if openai_response and openai_response.choices:
                        # 调试输出原始 chunk JSON，直接查看 tool_calls
                        try:
                            import logging as _logging
                            _logging.getLogger(__name__).info(f"Full raw Dify chunk: {json.dumps(chunk_data, ensure_ascii=False)}")
                        except Exception:
                            pass
                        yield f"data: {openai_response.model_dump_json()}\n\n"

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {data}, error: {e}")
                continue

    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        error_msg = {
            "error": {
                "message": f"Stream processing error: {str(e)}",
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_msg)}\n\n"

    yield "data: [DONE]\n\n"

# === Stream Collector for Non-Stream Requests ===
async def collect_dify_stream_response(dify_response: httpx.Response, model: str) -> OpenAITextResponse:
    """Collect streaming response and convert to non-streaming OpenAI format"""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    usage = Usage()
    collected_content = []

    if dify_response.status_code != 200:
        # Handle error response
        error_message = f"Dify API error: Status {dify_response.status_code}"
        try:
            response_text = await dify_response.aread()
            error_content = json.loads(response_text.decode())
            error_message = error_content.get("message", error_message)
            logger.error(f"Dify API error response: {error_content}")
        except Exception as e:
            logger.error(f"Failed to parse Dify error response: {e}")

        raise HTTPException(
            status_code=dify_response.status_code,
            detail={
                "error": {
                    "message": error_message,
                    "type": "server_error",
                    "code": dify_response.status_code
                }
            }
        )

    try:
        async for line in dify_response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue

            data = line[6:].strip()
            if data == "[DONE]":
                break

            try:
                chunk_data = json.loads(data)
                dify_chunk = DifyChunkChatCompletionResponse(**chunk_data)

                if dify_chunk.event == "message_end":
                    # Extract usage info
                    if dify_chunk.metadata and "usage" in dify_chunk.metadata:
                        usage_data = dify_chunk.metadata["usage"]
                        usage = Usage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0)
                        )
                    break

                elif dify_chunk.event == "error":
                    # Handle error event
                    error_msg = chunk_data.get("message", "Unknown error")
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": {
                                "message": error_msg,
                                "type": "server_error",
                                "code": chunk_data.get("code", 500)
                            }
                        }
                    )

                elif dify_chunk.event in ["message", "agent_message"]:
                    # Collect message content
                    content = dify_chunk.answer or ""

                    # Handle special thinking tags
                    if content == '<details style="color:gray;background-color: #f8f8f8;padding: 8px;border-radius: 4px;" open> <summary> Thinking... </summary>\n':
                        content = "<think>"
                    elif content == "</details>":
                        content = "</think>"

                    collected_content.append(content)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {data}, error: {e}")
                continue

    except Exception as e:
        logger.error(f"Stream collection error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Stream processing error: {str(e)}",
                    "type": "server_error",
                    "code": 500
                }
            }
        )

    # Build final response
    final_content = "".join(collected_content)

    return OpenAITextResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[ChatCompletionChoice(
            index=0,
            message=Message(
                role="assistant",
                content=final_content
            ),
            finish_reason="stop"
        )],
        usage=usage
    )


# === Models List Endpoint ===
class ModelManager:
    def __init__(self):
        self.name_to_api_key: Dict[str, str] = {}
        self.last_refresh_time: Optional[int] = None

    async def refresh_model_info(self):
        if not DIFY_API_KEYS or not DIFY_API_BASE:
            logger.warning("Dify API configuration missing")
            return
        try:
            dify_api_key = DIFY_API_KEYS[0]
            headers = {
                "Authorization": f"Bearer {dify_api_key}",
                "Content-Type": "application/json"
            }
            # 先尝试 /models，如404则退回到 /info
            resp = await http_client.get(f"{DIFY_API_BASE}/models", headers=headers)
            if resp.status_code == 404:
                logger.warning(f"/models endpoint not found, trying /info for each API key instead")
                # 对每个 API Key 分别调用 /info
                for key in DIFY_API_KEYS:
                    headers_each = {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    }
                    info_resp = await http_client.get(f"{DIFY_API_BASE}/info", headers=headers_each)
                    if info_resp.status_code == 200:
                        app_info = info_resp.json()
                        app_name = app_info.get("name")
                        if app_name:
                            self.name_to_api_key[app_name] = key
                            logger.info(f"Added model(app) from /info: {app_name}")
                    else:
                        logger.error(f"Failed to fetch /info for key {key[:6]}..., status: {info_resp.status_code}")
                self.last_refresh_time = int(time.time())
                return
            elif resp.status_code != 200:
                logger.error(f"Failed to refresh model info: {resp.status_code} {resp.text}")
                return

            logger.info(f"Dify /models raw response: {resp.text}")
            result = resp.json()
            self.name_to_api_key.clear()
            # 针对可能不在 data 里的情况，直接遍历 root 列表
            models_list = result.get("data", result)
            if isinstance(models_list, dict) and "models" in models_list:
                models_list = models_list["models"]
            if not isinstance(models_list, list):
                logger.warning(f"Unexpected /models format: {models_list}")
                models_list = []
            for m in models_list:
                model_id = m.get("id") or m.get("name")
                if model_id:
                    logger.info(f"Adding model: {model_id}")
                    self.name_to_api_key[model_id] = dify_api_key
            self.last_refresh_time = int(time.time())
        except Exception as e:
            logger.error(f"Error refreshing model info: {e}")

    def get_available_models(self) -> List[Dict[str, Any]]:
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


model_manager = ModelManager()


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    避免每次都刷新，采用懒加载策略获取模型列表
    """
    if not model_manager.name_to_api_key:
        await model_manager.refresh_model_info()
    models = model_manager.get_available_models()
    resp = {"object": "list", "data": models}
    # 使用ujson和直接返回Response提升性能
    return Response(
        content=json.dumps(resp, ensure_ascii=False),
        media_type="application/json",
        headers={"access-control-allow-origin": "*"}
    )

@app.post("/v1/chat/completions")
async def chat_completions(
    request: GeneralOpenAIRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Validate Dify API configuration
        dify_api_key = DIFY_API_KEYS[0] if DIFY_API_KEYS else ""
        if not dify_api_key or not DIFY_API_BASE:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Server configuration error: Dify API not configured",
                        "type": "server_error",
                        "code": 500
                    }
                }
            )

        # Convert request
        dify_request = await request_openai_to_dify(request, dify_api_key)

        # Debug: Log the Dify request
        logger.info(f"Dify request: {dify_request.model_dump()}")

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {dify_api_key}",
            "Content-Type": "application/json"
        }

        # Send request to Dify - Always use streaming since Agent Chat App doesn't support blocking
        if request.stream:
            # Streaming request - keep stream open for StreamingResponse
            async def stream_with_dify():
                async with http_client.stream(
                    "POST",
                    f"{DIFY_API_BASE}/chat-messages",
                    json=dify_request.model_dump(),
                    headers=headers,
                    timeout=60.0
                ) as dify_response:
                    async for chunk in dify_stream_handler(dify_response, request.model):
                        yield chunk

            return StreamingResponse(
                stream_with_dify(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming request - collect streaming response and convert
            async with http_client.stream(
                "POST",
                f"{DIFY_API_BASE}/chat-messages",
                json=dify_request.model_dump(),
                headers=headers,
                timeout=60.0
            ) as dify_response:
                openai_response = await collect_dify_stream_response(dify_response, request.model)
                return openai_response

    except HTTPException as he:
        # Re-raise HTTPException with proper error format
        if hasattr(he, 'detail') and isinstance(he.detail, dict):
            raise he
        else:
            # Convert plain HTTPException to proper error format
            raise HTTPException(
                status_code=he.status_code,
                detail={
                    "error": {
                        "message": str(he.detail) if he.detail else f"HTTP {he.status_code} error",
                        "type": "server_error",
                        "code": he.status_code
                    }
                }
            )
    except Exception as e:
        logger.error(f"Chat completions error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "server_error",
                    "code": 500
                }
            }
        )

def main():
    import uvicorn
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 8000))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
