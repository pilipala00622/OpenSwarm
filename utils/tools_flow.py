"""
Tools 全部逻辑：从 schema 到请求、tool_calls、执行、回传的完整流程说明与输出

用于打印/返回当前 Agent 的 tools 定义与执行链路，便于排查或文档化。
"""

import json
from typing import Dict, List, Any, Optional


def get_tool_schemas_json(agent) -> List[Dict[str, Any]]:
    """返回 Agent 当前所有工具的 OpenAI 格式 schema（即发往 API 的 tools 列表）"""
    if not hasattr(agent, "get_tool_schemas"):
        return []
    return agent.get_tool_schemas()


def get_tools_full_logic(agent) -> Dict[str, Any]:
    """
    返回 tools 全部逻辑：schemas + 流程步骤说明 + 代码位置索引。

    Returns:
        {
            "schemas": [...],           # 发往 API 的 tools（OpenAI function 格式）
            "tool_names": [...],        # 工具名列表
            "flow": [...],              # 流程步骤文字说明
            "code_refs": {...},         # 关键逻辑所在模块/方法
        }
    """
    schemas = get_tool_schemas_json(agent) if agent else []
    tool_names = [s["function"]["name"] for s in schemas]

    flow = [
        "1. Agent.get_tool_schemas() → 得到 OpenAI 格式的 tools 列表（name, description, parameters）",
        "2. Agent.process_message(messages) 内调用 llm_client.chat(messages, tools=schemas, ...)",
        "3. 网关返回 response：若支持 function calling 则含 tool_calls[{id, function: {name, arguments}}]",
        "4. MainRollout 若见 response.tool_calls：调用 agent.execute_tool_calls(tool_calls)",
        "5. execute_tool_calls 对每个 tc：json.loads(arguments) → agent.execute_tool(name, arguments)",
        "6. 各 Tool.execute(**kwargs) 返回 ToolResult(content, success, error)；to_str() 给模型看",
        "7. 将 {role:'tool', tool_call_id, content: result.to_str()} 追加到 messages",
        "8. 循环继续 process_message(messages) 直到无 tool_calls 或达到 max_steps",
    ]

    code_refs = {
        "schema 生成": "agent/agent.py Agent.get_tool_schemas() → tool/base.py BaseTool.to_openai_schema()",
        "发请求带 tools": "utils/trpc_openai_client.py TrpcOpenAIClient.chat(messages, tools=...)",
        "解析 tool_calls": "agent/agent.py Agent.execute_tool_calls(tool_calls)",
        "执行单工具": "agent/agent.py Agent.execute_tool(name, arguments) → BaseTool.execute(**kwargs)",
        "结果回传": "rollout/main_rollout.py 将 tool result 消息 append 到 self.messages，下一轮 process_message",
    }

    return {
        "schemas": schemas,
        "tool_names": tool_names,
        "flow": flow,
        "code_refs": code_refs,
    }


def print_tools_full_logic(agent, indent: str = "  "):
    """打印 tools 全部逻辑（schemas JSON + 流程 + 代码索引）"""
    logic = get_tools_full_logic(agent)
    print(indent + "=== Tools 全部逻辑 ===")
    print(indent + "tool_names:", logic["tool_names"])
    print(indent + "schemas (发往 API 的 tools):")
    print(json.dumps(logic["schemas"], ensure_ascii=False, indent=2))
    print(indent + "flow:")
    for step in logic["flow"]:
        print(indent + "  " + step)
    print(indent + "code_refs:")
    for k, v in logic["code_refs"].items():
        print(indent + "  " + k + ": " + v)
