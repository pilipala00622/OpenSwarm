"""
与 trpc 网关对接的同步示例（与你给的片段一致）

直接使用 OpenAI 客户端，api_key 格式：APP_ID:APP_KEY?provider=moonshot&timeout=600&model=模型名
是否支持 tools 取决于网关实现，可在 create 时传入 tools 试一次。
"""

import os
from openai import OpenAI

# 方式一：环境变量里直接放完整 api_key
# export TRPC_OPENAI_API_KEY='APP_ID:APP_KEY?provider=moonshot&timeout=600&model=kimi-k2-0905-preview'

# 方式二：用 APP_ID + APP_KEY 拼
APP_ID = os.environ.get("TRPC_APP_ID", "your-app-id")
APP_KEY = os.environ.get("TRPC_APP_KEY", "your-app-key")
MODEL = os.environ.get("TRPC_MODEL", "kimi-k2-0905-preview")
BASE_URL = os.environ.get("TRPC_OPENAI_BASE_URL", "http://trpc-gpt-eval.production.polaris:8080/v1")

api_key = os.environ.get("TRPC_OPENAI_API_KEY")
if not api_key:
    api_key = f"{APP_ID}:{APP_KEY}?provider=moonshot&timeout=600&model={MODEL}"

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)

print(response.choices[0].message.content)

# 若网关支持 tools，可传入 tools 测试：
# response = client.chat.completions.create(
#     model=MODEL,
#     messages=[...],
#     tools=[{"type": "function", "function": {"name": "search", "description": "...", "parameters": {...}}}],
#     tool_choice="auto",
# )
# if response.choices[0].message.tool_calls:
#     print("tool_calls:", response.choices[0].message.tool_calls)
