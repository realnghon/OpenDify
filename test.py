import openai
from openai import OpenAI

# 创建OpenAI客户端实例
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-abc123"  # 可以使用任意值
)

# 使用新的API调用方式
response = client.chat.completions.create(
    model="AgentCoder",  # 使用 Dify 应用的名称
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'content') and delta.content is not None:
        print(delta.content, end="", flush=True)
