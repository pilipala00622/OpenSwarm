# OpenSwarm — 多智能体编排框架 v0.3.1

> 一个轻量级、可扩展的多 AI 智能体协作框架，对齐 Oh-My-OpenCode 核心功能。  

---

## 目录

- [项目概览](#项目概览)
- [v0.3.1 更新日志](#v031-更新日志)
- [项目结构](#项目结构)
- [安装与运行](#安装与运行)
- [核心概念](#核心概念)
  - [Agent 智能体](#agent-智能体)
  - [Rollout 执行循环](#rollout-执行循环)
  - [Tool 工具系统](#tool-工具系统)
- [六大核心功能](#六大核心功能)
  - [1. Category 任务类别系统](#1-category-任务类别系统)
  - [2. Tool 权限控制](#2-tool-权限控制)
  - [3. 后台任务执行](#3-后台任务执行)
  - [4. Task 依赖系统](#4-task-依赖系统)
  - [5. Handoff 跨会话交接](#5-handoff-跨会话交接)
  - [6. 外部记忆与上下文压缩](#6-外部记忆与上下文压缩)
- [可靠性与可观测性](#可靠性与可观测性)
  - [并发控制与限流](#并发控制与限流)
  - [子 Agent 超时控制](#子-agent-超时控制)
  - [Token 用量统计](#token-用量统计)
  - [Graceful Shutdown](#graceful-shutdown)
- [执行追踪 (Tracing)](#执行追踪-tracing)
- [错误恢复与 Checkpoint](#错误恢复与-checkpoint)
- [动态 Scaling Rules](#动态-scaling-rules)
- [配置参考](#配置参考)
  - [AgentConfig](#agentconfig)
  - [RolloutConfig](#rolloutconfig)
  - [SubRolloutConfig](#subrolloutconfig)
  - [TaskTool 配置](#tasktool-配置)
- [自定义工具](#自定义工具)
- [完整示例](#完整示例)
- [存储格式](#存储格式)
- [已知限制与 TODO](#已知限制与-todo)
- [声明](#声明)

---

## 项目概览

Open Swarm 是一个多智能体编排框架，主 Agent 通过创建子 Agent 并分派任务来协作完成复杂工作。v0.3.0 对齐了 Oh-My-OpenCode 的核心功能集，v0.3.1 进行了全面的可靠性加固，修复了 22 个问题（含 3 个 Critical），新增并发控制、超时、Token 统计等生产就绪能力。

**六大核心功能**：

| 功能 | 说明 |
|---|---|
| **Category System** | 8 种内置任务类别，每种映射到不同的模型/温度/推理强度 |
| **Tool 权限控制** | 黑名单、白名单、禁止委派三层工具访问控制 |
| **后台任务** | 异步启动子 Agent，主 Agent 不阻塞，稍后获取结果 |
| **Task 依赖** | 任务间 blockedBy/blocks 依赖关系，自动并行调度，含环检测 |
| **Handoff** | 跨会话上下文交接，自动提取已完成/待完成工作 |
| **外部记忆** | Anthropic 风格的 LLM 摘要 + 文件持久化上下文压缩 |

**v0.3.1 新增能力**：

| 能力 | 说明 |
|---|---|
| **并发限流** | `asyncio.Semaphore` 控制子 Agent 最大并发数（默认 10） |
| **子 Agent 超时** | `asyncio.wait_for` 对单个子任务施加总时长限制 |
| **Token 统计** | LLM 调用自动采集 prompt/completion/total tokens，Tracer 汇总输出 |
| **原子写入** | TaskStore 使用 write-to-tmp + `os.replace` 防止文件损坏 |
| **Graceful Shutdown** | `interrupt()` 时自动取消所有后台任务 |

---

## v0.3.1 更新日志

> 2026-03-15 — 全面可靠性加固，修复 22 个问题（3 Critical / 4 High / 6 Medium / 4 Low / 5 Arch），2 个标记 TODO。

### Critical 修复

| 修复 | 说明 |
|---|---|
| **Category 模型生效** | 子 Agent 不再无条件复用父 Agent 的 LLM Client。当 Category 指定的 model/api_key/base_url 与父 Agent 不同时，自动创建独立 `LLMClient` 实例 |
| **LLM 重试全挂抛异常** | `LLMClient.chat()` 三次重试全部失败后不再伪装成正常回复，改为抛出 `RuntimeError`；Rollout 循环进入错误恢复流程 |
| **空回复防护** | 新增 `_is_empty_response()` 检测，空回复时注入系统提示要求 LLM 重新作答，避免空转到 max_steps |

### High 修复

| 修复 | 说明 |
|---|---|
| **并发竞态保护** | `subagent_counter` 改为 `itertools.count()`，共享状态加 `asyncio.Lock` |
| **Memory 线程安全** | `_llm_summarise` 使用 `model=` 参数覆盖而非修改共享 `llm_client.model_id` |
| **TaskStore 原子写入** | `_save_task` 改为 write-to-tmp + `os.replace`，防止进程崩溃导致 JSON 损坏 |
| **依赖环检测** | `TaskStore.create()` 新增 DFS 环检测，循环依赖抛 `ValueError` |

### Medium 修复

| 修复 | 说明 |
|---|---|
| **配置字段透传** | `top_p` / `reasoning_effort` / `thinking_budget` 现在从 AgentConfig 正确传递到 LLM API |
| **fork_context 快照** | 并行 tool calls 前快照父消息列表，确保所有子 Agent 看到相同上下文 |
| **CancelledError 捕获** | 后台任务回调兼容 Python 3.9+ 的 `asyncio.CancelledError`（继承自 `BaseException`） |
| **Handoff 工作提取** | `create()` 从 tasks 快照自动推断 completed_work / remaining_work |
| **JSON 解析错误反馈** | 工具参数 JSON 解析失败时返回明确错误 ToolResult，不再静默传空参数 |
| **CreateSubagent overwrite** | 新增 `overwrite` 参数支持更新已有子 Agent 配置 |

### Low 修复

- Checkpoint 列表限制上限 3 个，避免内存膨胀
- `_save_result` 修复相对路径边缘 case
- `datetime.utcnow()` 迁移到 `datetime.now(UTC)`（Python 3.12+ 兼容）
- `AgentMemory` 提供 `set_llm_client()` 公开方法，不再直接访问私有属性

### 架构增强

- **子 Agent 超时**：`_run_subtask` 使用 `asyncio.wait_for(coro, timeout=subtask_timeout)`
- **Token 统计**：`LLMClient` 返回 `usage`，`RolloutTracer` 汇总，终端输出 prompt/completion/total tokens
- **并发限流**：`asyncio.Semaphore(max_concurrent_subagents=10)` 控制最大并发子 Agent 数
- **结果质量标记**：`max_steps_reached` 结果加 `[WARNING]` 前缀
- **Graceful Shutdown**：`interrupt()` 自动 `cancel_all_background()` 取消后台任务

### TODO（待后续版本完成）

- **Handoff 消息历史注入**：load 操作应将 `context_messages` 注入 Rollout 消息历史，需设计 tool→rollout 回调机制
- **Agent Registry 持久化**：`CreateSubagentTool` 创建的 agent 配置需持久化到文件，以配合 Handoff 跨会话恢复

---

## 项目结构

```
agent-swarm/
├── __init__.py                  # 包入口，导出所有公共 API
│
├── agent/
│   ├── __init__.py
│   └── agent.py                 # Agent 核心类 + AgentConfig 配置
│
├── rollout/
│   ├── __init__.py
│   ├── base.py                  # BaseRollout 抽象基类、RolloutConfig、RolloutResult
│   ├── main_rollout.py          # MainRollout — 主 Agent 执行循环
│   └── sub_rollout.py           # SubRollout — 子 Agent 轻量执行循环
│
├── tool/
│   ├── __init__.py
│   ├── base.py                  # BaseTool 抽象接口 + ToolResult
│   ├── search.py                # SearchTool — Serper API 网络搜索
│   ├── code_runner.py           # CodeRunnerTool — Python/JS 代码执行
│   └── verify.py                # VerifyTool — 事实验证
│
├── swarm_tool/
│   ├── __init__.py
│   ├── create_subagent.py       # CreateSubagentTool — 动态创建子 Agent 配置
│   ├── task.py                  # TaskTool (assign_task) — 核心任务分派工具
│   ├── background.py            # BackgroundOutputTool / BackgroundCancelTool
│   ├── task_management.py       # TaskCreateTool / TaskGetTool / TaskListTool / TaskUpdateTool
│   └── handoff_tool.py          # HandoffTool — 跨会话交接工具
│
├── utils/
│   ├── __init__.py
│   ├── llm_client.py            # LLMClient — OpenAI 兼容 LLM 客户端
│   ├── trpc_openai_client.py    # TrpcOpenAIClient — tRPC OpenAI 客户端
│   ├── eval_llm_client.py       # EvalLLMClient — 评估用客户端
│   ├── category.py              # CategoryRegistry / CategoryConfig — 任务类别系统
│   ├── task_store.py            # TaskStore / Task — 持久化任务管理与依赖追踪
│   ├── handoff.py               # HandoffManager / HandoffDocument — 跨会话交接
│   ├── memory.py                # AgentMemory — 外部记忆 + LLM 摘要
│   ├── tracer.py                # RolloutTracer — 结构化执行追踪
│   └── tools_flow.py            # 工具流程可视化辅助
│
├── examples/
│   ├── simple_agent.py          # 单 Agent 简单示例
│   └── multi_agent.py           # 多 Agent 完整功能展示
│
├── run_examples/                # 多种配置的运行脚本
├── result/                      # 运行结果存储目录
├── setup.py                     # 安装脚本
└── requirements.txt             # 依赖声明
```

---

## 安装与运行

```bash
# 安装
pip install -e .

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export SERPER_API_KEY="your-serper-key"    # 可选，搜索工具需要

# 运行示例
python examples/multi_agent.py
python run_examples/run_kimi_kimi.py
```

依赖：`openai >= 1.0`、`httpx`

---

## 核心概念

### Agent 智能体

`Agent` 是框架的核心执行单元，负责：
- 管理与 LLM 的对话
- 执行工具调用（含权限过滤）
- 并行执行多个工具调用（`asyncio.gather`）

```python
from open_swarm import Agent, AgentConfig

config = AgentConfig(
    name="researcher",
    system_prompt="你是一个研究助手。",
    model_id="kimi-k2.5",
    temperature=0.7,
    blocked_tools=["code_runner"],  # 禁用代码执行
    can_delegate=True,              # 允许创建子 Agent
)

agent = Agent(config, tools=[search_tool, verify_tool, task_tool])
```

**并行工具执行**：当 LLM 一次返回多个 tool_calls 时，`Agent.execute_tool_calls()` 使用 `asyncio.gather` 并行执行所有工具调用，显著加速多子 Agent 并发场景。

### Rollout 执行循环

Rollout 管理 Agent 的完整执行循环：接收用户消息 → LLM 推理 → 工具执行 → 循环直到完成。

**两种 Rollout**：

| 类型 | 用途 | 默认步数 | 特性 |
|---|---|---|---|
| `MainRollout` | 主 Agent | 50 步 | 完整功能：memory、tracing、checkpoint、scaling rules、结果存储 |
| `SubRollout` | 子 Agent | 20 步 | 轻量级：step hint、更低错误容忍、无 memory/tracing |

```python
from open_swarm import MainRollout, RolloutConfig

rollout = MainRollout(RolloutConfig(
    max_steps=30,
    enable_memory=True,
    enable_tracing=True,
    storage_path="result/output.jsonl",
))
result = await rollout.run(agent, "分析最新的 AI 发展趋势")
```

### Tool 工具系统

所有工具继承自 `BaseTool`，实现统一的 OpenAI Function Calling 接口。

**内置工具一览**：

| 工具名 | 类 | 功能 |
|---|---|---|
| `search` | `SearchTool` | Serper API 网络搜索 |
| `code_runner` | `CodeRunnerTool` | Python/JavaScript 代码沙盒执行 |
| `verify_result` | `VerifyTool` | 使用 LLM 交叉验证事实 |
| `create_subagent` | `CreateSubagentTool` | 动态创建子 Agent 配置（支持 `overwrite` 更新） |
| `assign_task` | `TaskTool` | 分派任务给子 Agent（核心） |
| `background_output` | `BackgroundOutputTool` | 获取后台任务结果 |
| `background_cancel` | `BackgroundCancelTool` | 取消后台任务 |
| `task_create` | `TaskCreateTool` | 创建受管理的任务 |
| `task_get` | `TaskGetTool` | 按 ID 查询任务 |
| `task_list` | `TaskListTool` | 列出所有活跃任务 |
| `task_update` | `TaskUpdateTool` | 更新任务状态 |
| `handoff` | `HandoffTool` | 创建/加载跨会话交接文档 |

---

## 六大核心功能

### 1. Category 任务类别系统

**概念**：Category 回答"这是什么类型的工作"，决定子 Agent 的模型选择、温度、提示词、推理强度和工具权限。

**8 种内置类别**：

| 类别 | 描述 | 推荐模型 | 温度 | 推理强度 |
|---|---|---|---|---|
| `visual-engineering` | 前端、UI/UX、设计、动画 | gemini-3.1-pro | 0.8 | medium |
| `ultrabrain` | 深度逻辑推理、复杂架构决策 | gpt-5.4 | 0.3 | high (thinking: 32K) |
| `deep` | 目标驱动的自主问题解决 | gpt-5.3-codex | 0.5 | high |
| `artistry` | 高度创意/艺术性任务 | gemini-3.1-pro | 1.2 | medium |
| `quick` | 简单任务：单文件修改、拼写修正 | claude-haiku-4-5 | 0.3 | low |
| `unspecified-low` | 通用任务，低工作量 | claude-sonnet-4-6 | 0.5 | medium |
| `unspecified-high` | 通用任务，高工作量 | claude-opus-4-6 | 0.5 | high (thinking: 16K) |
| `writing` | 文档、文章、技术写作 | gemini-3-flash | 0.6 | medium |

**自定义类别**：

```python
from open_swarm import CategoryRegistry

registry = CategoryRegistry(custom_categories={
    "korean-writer": {
        "description": "韩语技术写作",
        "temperature": 0.6,
        "model": "gpt-5.3",
        "prompt_append": "请用韩语写作，保持专业且自然的风格。",
    }
})
```

**使用方式 — 分派任务时指定 category**：

```python
# LLM 通过工具调用指定 category
await task_tool.execute(
    prompt="修复 README 中的拼写错误",
    category="quick",          # → 使用 claude-haiku-4-5，低温度，快速完成
)

await task_tool.execute(
    prompt="分析微服务架构的扩展性问题",
    category="ultrabrain",     # → 使用 gpt-5.4，extended thinking 32K tokens
)
```

**Category 配置项** (`CategoryConfig`)：

| 字段 | 类型 | 说明 |
|---|---|---|
| `description` | str | 类别描述，注入到子 Agent 的 system prompt |
| `model` | str | 推荐的 LLM 模型 ID |
| `temperature` | float | 采样温度 |
| `max_tokens` | int | 最大响应 token 数 |
| `prompt_append` | str | 追加到系统提示词的文本 |
| `reasoning_effort` | str | 推理强度 ("low"/"medium"/"high") |
| `thinking_budget` | int | Extended thinking token 预算 (0=关闭) |
| `blocked_tools` | list | 该类别下被屏蔽的工具名 |
| `allowed_tools_only` | list | 白名单模式：非空时只允许列表内的工具 |
| `can_delegate` | bool | 是否允许子 Agent 继续委派（默认 False） |

---

### 2. Tool 权限控制

Agent 支持三层工具访问控制，在初始化时一次性计算，运行时零开销。

**三层过滤逻辑**：

```
全部工具 → 黑名单过滤 → 白名单过滤 → can_delegate 过滤 → 最终可用工具
```

| 层级 | 配置项 | 说明 |
|---|---|---|
| 黑名单 | `blocked_tools` | 列表中的工具被禁用 |
| 白名单 | `allowed_tools_only` | 非空时，只有列表中的工具可用 |
| 委派控制 | `can_delegate` | 为 False 时，移除 `create_subagent` 和 `assign_task` |

**示例**：

```python
# 只读分析 Agent：禁用代码执行和委派
config = AgentConfig(
    name="analyst",
    blocked_tools=["code_runner"],
    can_delegate=False,
)

# 严格白名单：只允许搜索和验证
config = AgentConfig(
    name="fact_checker",
    allowed_tools_only=["search", "verify_result"],
)
```

**被屏蔽的工具调用会返回明确的错误信息**：

```
Tool 'code_runner' is blocked for this agent
```

---

### 3. 后台任务执行

通过 `run_in_background=true` 参数，子 Agent 以 `asyncio.create_task` 方式在后台执行，主 Agent 立即拿回 task_id 继续工作。

**工作流程**：

```
主 Agent 调用 assign_task(run_in_background=true)
    ↓
返回 "subagent_3" task_id，主 Agent 继续处理其他工作
    ↓
后台子 Agent 异步执行中...
    ↓
主 Agent 调用 background_output(task_id="subagent_3") 获取结果
    ↓
如果仍在运行 → "Task is still running. Try again later."
如果已完成   → 返回子 Agent 的完整输出
```

**相关工具**：

| 工具 | 功能 |
|---|---|
| `assign_task(run_in_background=true)` | 启动后台任务，立即返回 task_id |
| `background_output(task_id)` | 获取后台任务结果 |
| `background_cancel(task_id)` | 取消仍在运行的后台任务 |

**技术实现**：
- 使用 `asyncio.create_task()` 创建异步任务
- `add_done_callback()` 自动在任务完成时收集结果（兼容 Python 3.9+ `CancelledError`）
- 结果存储在 `TaskTool._background_results` 字典中，访问受 `asyncio.Lock` 保护
- Graceful Shutdown：`interrupt()` 时自动 `cancel_all_background()` 取消所有后台任务

---

### 4. Task 依赖系统

结构化的任务管理系统，支持依赖关系和自动并行调度。

**任务模型** (`Task`)：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 唯一标识符，格式 `T-{uuid8}`（如 `T-a1b2c3d4`） |
| `subject` | str | 祈使句任务描述 |
| `description` | str | 详细描述 |
| `status` | str | `pending` → `in_progress` → `completed` / `deleted` |
| `blocked_by` | list | 依赖的任务 ID 列表 |
| `blocks` | list | 被本任务阻塞的任务 ID 列表（自动维护） |
| `owner` | str | 负责的 Agent 名称 |
| `result` | str | 任务输出（完成时设置） |

**依赖关系示例**：

```
[Build Frontend] (T-001)  ──┐
                              ├──→ [Integration Tests] (T-003) ──→ [Deploy] (T-004)
[Build Backend]  (T-002)  ──┘
```

```python
from open_swarm import TaskStore

store = TaskStore(store_dir=".agent_tasks")

t1 = store.create("Build Frontend")
t2 = store.create("Build Backend")
t3 = store.create("Integration Tests", blocked_by=[t1.id, t2.id])
t4 = store.create("Deploy", blocked_by=[t3.id])

# t1 和 t2 无依赖，可以并行执行
ready = store.get_ready_tasks()  # → [t1, t2]

# 完成 t1 后，自动从 t3 的 blocked_by 中移除 t1
store.update(t1.id, status="completed")
# t3.blocked_by 现在只剩 [t2.id]

# 完成 t2 后，t3 自动解锁
store.update(t2.id, status="completed")
ready = store.get_ready_tasks()  # → [t3]
```

**持久化**：任务以 JSON 文件存储在配置目录中（默认 `.agent_tasks/`），使用 write-to-tmp + `os.replace` 原子写入，支持跨进程、跨会话持续存在。

**依赖环检测**：`create()` 方法内置 DFS 环检测，循环依赖（如 A → B → A）会被即时拒绝并抛出 `ValueError`。

**4 个 CRUD 工具**：

| 工具 | 功能 |
|---|---|
| `task_create(subject, blocked_by=[...])` | 创建任务，可设依赖 |
| `task_get(id)` | 查询单个任务详情 |
| `task_list(status?, owner?)` | 列出任务（支持过滤） |
| `task_update(id, status, result)` | 更新状态，完成时自动 unblock 下游 |

---

### 5. Handoff 跨会话交接

当一个会话的工作需要在新会话中继续时，Agent 可以创建 Handoff 文档，捕获当前上下文状态。

**Handoff 文档内容** (`HandoffDocument`)：

| 字段 | 说明 |
|---|---|
| `id` | 唯一标识，格式 `H-YYYYMMDD_HHMMSS_{uuid6}` |
| `agent_name` | 创建交接的 Agent 名称 |
| `summary` | LLM 生成或启发式提取的工作摘要 |
| `completed_work` | 已完成的工作列表（自动从 tasks 快照提取） |
| `remaining_work` | 待完成的工作列表（自动从 tasks 快照提取） |
| `key_files` | 相关文件路径列表（自动从消息中提取） |
| `key_decisions` | 会话中做出的重要决策 |
| `context_messages` | 压缩的消息历史（system + 最近 20 条） |
| `task_snapshot` | 活跃任务的快照 |
| `notes` | Agent 或用户的自由备注 |

**工作流程**：

```python
# 在当前会话结束前创建 handoff
handoff(action="create", notes="用户希望明天继续分析")
# → 返回 handoff_id: H-20260315_143022_a1b2c3

# 在新会话中加载
handoff(action="load", handoff_id="H-20260315_143022_a1b2c3")
# → 返回完整的上下文摘要，可注入新 Agent 的 context

# 加载最近一次 handoff
handoff(action="load_latest")

# 列出所有可用的 handoff
handoff(action="list")
```

**存储**：Handoff 文档以 JSON 文件存储在配置目录中（默认 `.agent_handoffs/`）。

**`to_context_message()` 方法**：将 HandoffDocument 转换为可直接注入新会话的 user message，格式化包含 Summary、Completed Work、Remaining Work、Key Decisions、Relevant Files、Active Tasks 等章节。

---

### 6. 外部记忆与上下文压缩

对齐 Anthropic 多智能体系统的记忆设计，三大原则：
1. **写入磁盘**：每个压缩阶段持久化为独立 Markdown 文件
2. **传路径而非文本**：内存中只记录文件路径索引，按需加载
3. **LLM 摘要**：使用 Agent 自身的 LLM 生成高质量摘要，无 LLM 时回退到启发式提取

**工作原理**：

```
消息数超过 max_context_messages (默认 50)
    ↓
保留 system message + 最近 keep_recent (默认 20) 条消息
    ↓
中间部分交给 LLM 生成摘要（或启发式提取）
    ↓
摘要 + 关键发现写入 .md 文件持久化到磁盘
    ↓
用 [Memory] 摘要消息替代压缩部分
    ↓
压缩后消息列表: [system, memory_summary, ...recent_20]
```

**Markdown 文件结构**：

```markdown
# phase_1
**Compressed at**: 2026-03-15 14:30:22
**Messages compressed**: 35

## Summary
<LLM 生成的摘要>

## Key Findings
- 发现 1
- 发现 2

## Raw Messages (JSON)
<完整原始消息 JSON，可用于重建>
```

**LLM 摘要提示词要求**：
- 捕获"做了什么"、"关键发现"、"未完成的线索"
- 最多 7 个要点
- 不超过 400 词
- 使用对话中的主要语言

> **v0.3.1 线程安全改进**：LLM 摘要不再临时修改共享 `LLMClient` 的 `model_id`，而是通过 `model=` 参数覆盖传递，避免并发协程间的状态污染。

**配置项**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `enable_memory` | True | 是否启用记忆系统 |
| `max_context_messages` | 50 | 触发压缩的消息数阈值 |
| `memory_keep_recent` | 20 | 压缩时保留最近多少条消息 |
| `memory_dir` | `.agent_memory/` | 记忆文件存储目录 |
| `enable_llm_summarise` | True | 是否使用 LLM 生成摘要 |

---

## 可靠性与可观测性

v0.3.1 新增一系列生产就绪能力，提升系统在真实部署中的稳定性。

### 并发控制与限流

Scaling Rules 是文本 prompt，靠 LLM 自觉遵守。为防止 LLM 在单个 turn 中发起过多并发子任务，`TaskTool` 内置硬性限流：

```python
task_tool = TaskTool(
    agent_registry=agent_registry,
    max_concurrent_subagents=10,   # asyncio.Semaphore 限流，默认 10
    ...
)
```

超出限制的子任务会排队等待，而非被拒绝。同时所有共享状态访问受 `asyncio.Lock` 保护，消除并发竞态。

### 子 Agent 超时控制

防止子 Agent 卡死或陷入死循环：

```python
task_tool = TaskTool(
    agent_registry=agent_registry,
    subtask_timeout=300.0,   # 单个子任务最多 300 秒，默认 None（无限制）
    ...
)
```

超时触发 `asyncio.TimeoutError`，子任务返回超时错误结果。

### Token 用量统计

`LLMClient.chat()` 自动从 API 响应中提取 `usage` 数据（prompt_tokens / completion_tokens / total_tokens），通过 `RolloutTracer` 汇总，终端模式下自动输出：

```
[Token Usage] Prompt: 45,200 | Completion: 8,600 | Total: 53,800
```

### Graceful Shutdown

当 `MainRollout.interrupt()` 被调用时：
1. 设置中断标志，下一个循环迭代终止
2. 查找 `assign_task` 工具，调用 `cancel_all_background()` 取消所有后台任务
3. 记录取消数量到 Tracer

---

## 执行追踪 (Tracing)

`RolloutTracer` 提供结构化的事件追踪，用于可观测性和事后分析。

**事件类型**：

| 事件类型 | 触发时机 |
|---|---|
| `llm_call` | 每次 LLM API 调用 |
| `tool_exec` | 每次工具执行 |
| `subagent_spawn` | 子 Agent 启动 |
| `subagent_complete` | 子 Agent 完成 |
| `token_usage` | LLM 调用返回 token 用量（prompt / completion / total） |
| `error` | 发生错误 |
| `recovery` | 错误恢复（checkpoint 恢复、重试） |
| `checkpoint` | 保存检查点 |
| `compression` | 记忆压缩 |

**输出格式**（JSONL）：

```json
{"ts": 0.123, "type": "llm_call", "agent": "orchestrator", "step": 0, "model": "kimi-k2.5", "message_count": 3}
{"ts": 0.124, "type": "token_usage", "agent": "orchestrator", "prompt_tokens": 1520, "completion_tokens": 340, "total_tokens": 1860}
{"ts": 1.456, "type": "tool_exec", "agent": "orchestrator", "step": 0, "tool": "search"}
{"ts": 2.789, "type": "subagent_spawn", "agent": "orchestrator", "step": 1, "subagent_id": "subagent_1"}
```

**摘要统计**：

```python
summary = tracer.summary()
# {
#   "total_events": 42,
#   "total_time_seconds": 125.6,
#   "llm_calls": 12,
#   "tool_executions": 8,
#   "subagents_spawned": 3,
#   "errors": 1,
#   "prompt_tokens": 45200,
#   "completion_tokens": 8600,
#   "total_tokens": 53800,
#   "tool_usage_breakdown": {"search": 3, "verify_result": 2, "assign_task": 3}
# }
```

---

## 错误恢复与 Checkpoint

MainRollout 和 SubRollout 都实现了自动错误恢复机制：

**流程**：

```
步骤执行出错
    ↓
连续错误计数 +1
    ↓
如果 < max_consecutive_errors (默认 3):
    → 通知 Agent 错误信息，让它换个方式重试
    ↓
如果 = max_consecutive_errors 且有 checkpoint:
    → 恢复到最近的 checkpoint，重置错误计数
    ↓
如果无 checkpoint 可恢复:
    → 终止执行，返回 ERROR 状态
```

**Checkpoint 机制**：

- 每隔 `checkpoint_interval` 步（默认 10）自动保存
- 保存完整消息快照（deepcopy）
- **最多保留 3 个 checkpoint**（Ring Buffer 模式），防止内存膨胀
- 恢复时注入系统提示"Multiple consecutive errors occurred. State has been restored."

**增强的错误检测**（v0.3.1）：

- **LLM 重试全挂**：`LLMClient.chat()` 三次重试失败后抛出 `RuntimeError`，Rollout 进入错误恢复流程
- **空回复检测**：`_is_empty_response()` 识别无内容无工具调用的响应，注入提示要求重新作答
- **错误响应识别**：`_is_complete()` 检查 `finish_reason == "error"`，不再误判为正常完成

---

## 动态 Scaling Rules

MainRollout 默认在系统提示词中注入资源分配规则：

```
简单查询 → 0 个子 Agent，直接回答
中等查询 → 1-2 个子 Agent
复杂查询 → 3-5 个子 Agent
非常复杂 → 5-10 个子 Agent
```

可通过 `RolloutConfig.scaling_rules` 自定义，或设 `enable_scaling_rules=False` 禁用。

---

## 配置参考

### AgentConfig

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 智能体名称 |
| `system_prompt` | str | "You are a helpful assistant." | 系统提示词 |
| `model_id` | str | "kimi-k2.5" | LLM 模型标识 |
| `api_key` | str | None | API 密钥（未设置时使用 `OPENAI_API_KEY` 环境变量） |
| `api_base_url` | str | None | API 地址（未设置时使用 `OPENAI_BASE_URL` 环境变量） |
| `subagent_model_id` | str | None | 子 Agent 模型（默认同 `model_id`） |
| `subagent_api_key` | str | None | 子 Agent API 密钥 |
| `subagent_api_base_url` | str | None | 子 Agent API 地址 |
| `max_tokens` | int | 4096 | 单次请求最大 token 数 |
| `temperature` | float | 0.7 | 采样温度 (0.0-2.0) |
| `top_p` | float | 1.0 | Nucleus 采样参数 |
| `reasoning_effort` | str | "medium" | 推理强度 ("low"/"medium"/"high") |
| `thinking_budget` | int | 0 | Extended thinking token 预算 (0=关闭) |
| `blocked_tools` | list | [] | 工具黑名单 |
| `allowed_tools_only` | list | [] | 工具白名单（非空时仅允许列表内工具） |
| `can_delegate` | bool | True | 是否允许委派子 Agent |

### RolloutConfig

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_steps` | int | 50 | 最大执行步数 |
| `terminal_mode` | bool | True | 终端打印开关 |
| `storage_path` | str | None | 结果存储路径（JSONL） |
| `print_tool_calls` | bool | True | 打印工具调用 |
| `print_tool_results` | bool | True | 打印工具结果 |
| `max_consecutive_errors` | int | 3 | 连续错误容忍上限 |
| `checkpoint_interval` | int | 10 | 自动 checkpoint 间隔（0=禁用） |
| `enable_memory` | bool | True | 启用外部记忆 |
| `max_context_messages` | int | 50 | 触发压缩的消息数阈值 |
| `memory_keep_recent` | int | 20 | 压缩时保留最近消息数 |
| `memory_dir` | str | None | 记忆文件目录 |
| `enable_llm_summarise` | bool | True | 使用 LLM 生成摘要 |
| `enable_scaling_rules` | bool | True | 注入资源分配规则 |
| `scaling_rules` | str | None | 自定义 scaling rules 文本 |
| `enable_tracing` | bool | True | 启用执行追踪 |
| `trace_output_path` | str | None | Trace JSONL 输出路径 |

### SubRolloutConfig

继承 `RolloutConfig`，以下默认值不同：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_steps` | 20 | 子 Agent 步数更低 |
| `step_hint` | True | 在工具结果中显示步数提示 |
| `terminal_mode` | False | 默认静音 |
| `enable_scaling_rules` | False | 子 Agent 不需要 |
| `enable_memory` | False | 短期运行不需要 |
| `enable_tracing` | False | 默认不追踪 |
| `max_consecutive_errors` | 2 | 更低容忍度 |
| `checkpoint_interval` | 0 | 默认禁用 |

### TaskTool 配置

`TaskTool` 构造函数参数（v0.3.1 新增）：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `agent_registry` | dict | 必填 | Agent 配置注册表 |
| `max_steps` | int | 20 | 子 Agent 默认最大步数 |
| `category_registry` | CategoryRegistry | None | 任务类别注册表 |
| `task_store` | TaskStore | None | 持久化任务管理器 |
| `max_concurrent_subagents` | int | 10 | 最大并发子 Agent 数（Semaphore 限流） |
| `subtask_timeout` | float | None | 单个子任务超时秒数（None = 无限制） |

---

## 自定义工具

继承 `BaseTool`，实现 4 个属性/方法：

```python
from open_swarm import BaseTool, ToolResult

class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "获取指定城市的天气信息"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }

    async def execute(self, city: str) -> ToolResult:
        # 实际调用天气 API
        weather = await fetch_weather(city)
        return ToolResult(content=f"{city}: {weather}", success=True)
```

工具通过 `to_openai_schema()` 自动转换为 OpenAI Function Calling 格式，无需手动编写 JSON schema。

---

## 完整示例

```python
import asyncio
from open_swarm import (
    Agent, AgentConfig, MainRollout, RolloutConfig,
    SearchTool, VerifyTool, CreateSubagentTool, TaskTool,
    BackgroundOutputTool, BackgroundCancelTool,
    TaskCreateTool, TaskGetTool, TaskListTool, TaskUpdateTool,
    HandoffTool, CategoryRegistry, TaskStore, HandoffManager,
)

async def main():
    # 基础设施
    agent_registry = {}
    category_registry = CategoryRegistry(custom_categories={
        "research": {
            "description": "深度研究",
            "temperature": 0.5,
            "prompt_append": "你是一个严谨的研究员，交叉验证多个来源。",
        }
    })
    task_store = TaskStore(store_dir="result/tasks")
    handoff_manager = HandoffManager(handoff_dir="result/handoffs")

    # 工具
    search = SearchTool()
    verify = VerifyTool()
    create_sub = CreateSubagentTool(agent_registry)
    task_tool = TaskTool(
        agent_registry=agent_registry,
        max_steps=15,
        category_registry=category_registry,
        task_store=task_store,
        max_concurrent_subagents=10,     # v0.3.1: 并发限流
        subtask_timeout=300.0,           # v0.3.1: 子任务超时 5 分钟
    )
    bg_output = BackgroundOutputTool(task_tool)
    bg_cancel = BackgroundCancelTool(task_tool)
    handoff = HandoffTool(handoff_manager=handoff_manager, task_store=task_store)

    # 主 Agent
    config = AgentConfig(
        name="orchestrator",
        system_prompt="你是编排智能体，负责分析任务复杂度并分派子 Agent。",
        model_id="kimi-k2-0711-preview",
        temperature=0.7,
        can_delegate=True,
    )
    agent = Agent(config, tools=[
        search, verify, create_sub, task_tool,
        bg_output, bg_cancel,
        TaskCreateTool(task_store), TaskGetTool(task_store),
        TaskListTool(task_store), TaskUpdateTool(task_store),
        handoff,
    ])
    task_tool.set_parent_agent(agent)
    task_tool.set_parent_tools([search, verify])

    # 执行
    rollout = MainRollout(RolloutConfig(
        max_steps=30,
        enable_memory=True,
        enable_tracing=True,
        storage_path="result/output.jsonl",
        trace_output_path="result/trace.jsonl",
    ))
    result = await rollout.run(agent, "研究 AI Agent 的最新发展并撰写报告")

    print(f"状态: {result.status.value}")
    print(f"步数: {result.steps}")
    print(result.final_response)

asyncio.run(main())
```

---

## 存储格式

### 运行结果 (JSONL)

```json
{
  "main": [...],
  "subs": [
    {
      "agent": "researcher",
      "agent_id": "subagent_1",
      "category": "research",
      "prompt": "研究 AI Agent 最新发展",
      "messages": [...],
      "status": "completed",
      "steps": 8,
      "content": "研究结果...",
      "task_id": "T-a1b2c3d4"
    }
  ]
}
```

### 任务文件 (.agent_tasks/T-xxxx.json)

```json
{
  "id": "T-a1b2c3d4",
  "subject": "Build Frontend",
  "status": "completed",
  "blocked_by": [],
  "blocks": ["T-e5f6g7h8"],
  "owner": "orchestrator",
  "result": "前端构建完成...",
  "created_at": "2026-03-15T14:30:00",
  "updated_at": "2026-03-15T14:45:00"
}
```

### Handoff 文件 (.agent_handoffs/H-xxxx.json)

```json
{
  "id": "H-20260315_143022_a1b2c3",
  "agent_name": "orchestrator",
  "summary": "会话涉及 42 条消息和 8 次工具执行...",
  "completed_work": ["完成前端搜索", "验证数据源"],
  "remaining_work": ["后端整合未完成"],
  "key_files": ["src/app.py", "tests/test_api.py"],
  "context_messages": [...],
  "task_snapshot": [...]
}
```

---

## 执行流程全景图

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    MainRollout                        │
│  ┌────────────────────────────────────────────────┐  │
│  │  [Scaling Rules 注入] → System Prompt           │  │
│  │  [Memory 连接] → 上下文压缩准备                   │  │
│  │  [Fork/Handoff 引用] → 工具绑定                   │  │
│  └────────────────────────────────────────────────┘  │
│                        │                              │
│           ┌────────────┼────────────┐                 │
│           ▼            ▼            ▼                 │
│       LLM 推理   → 工具调用    → 完成检测             │
│           │            │            (含空回复/         │
│           │            │             错误响应检测)     │
│           │     ┌──────┼──────┬──────┐               │
│           │     ▼      ▼      ▼      ▼               │
│           │  search  verify  assign_task  task_create │
│           │                    │                      │
│           │        [消息快照 snapshot_parent_messages] │
│           │              ┌─────┴─────┐                │
│           │              ▼           ▼                │
│           │         同步执行    后台执行               │
│           │              │           │                │
│           │       [Semaphore 限流]   │                │
│           │       [wait_for 超时]    │                │
│           │              │     ┌─────┘                │
│           │      ┌───────┘     │                      │
│           │      ▼             ▼                      │
│           │  SubRollout   asyncio.Task               │
│           │      │         (Lock 保护结果)            │
│           │      ▼             ▼                      │
│           │   子 Agent      background_output         │
│           │   返回结果      获取结果                    │
│           │                                           │
│           ▼                                           │
│    ┌──────────────┐   ┌──────────────┐               │
│    │  Checkpoint   │   │   Memory     │               │
│    │  每 N 步保存   │   │  超限时压缩   │               │
│    │  (最多保留 3)  │   │              │               │
│    └──────────────┘   └──────────────┘               │
│                        │                              │
│                        ▼                              │
│      ┌──────────────────────────────────┐             │
│      │  Tracer 事件记录 + Token 用量统计  │             │
│      └──────────────────────────────────┘             │
│                        │                              │
│           [interrupt() → cancel_all_background()]     │
└──────────────────────────────────────────────────────┘
    │
    ▼
  RolloutResult (状态、消息、步数、子 Agent 记录)
    │
    ├── 保存到 JSONL (storage_path)
    ├── 导出 Trace (trace_output_path)
    └── 可选: Handoff 交接到新会话
```

---

## 已知限制与 TODO

| 编号 | 说明 | 状态 |
|:---:|------|:---:|
| #22 | **Handoff 消息历史注入**：`load` 操作目前只返回格式化文本，不会将 `context_messages` 真正注入到 Rollout 消息历史。需设计 tool → rollout 回调机制。 | TODO |
| #24 | **Agent Registry 持久化**：`CreateSubagentTool` 创建的 agent 配置仅存在内存中，跨会话 Handoff 恢复后丢失。需设计文件持久化格式或在 Handoff 中保存 registry 快照。 | TODO |

---

## 声明

本仓库是**个人性质的实验性项目**，仅用于 demo / 参考实现。v0.3.1 进行了全面的可靠性加固（并发控制、超时、原子写入、错误恢复等），但尚未经过大规模生产验证。成本控制、多租户隔离等企业级功能未系统性处理。项目不代表任何公司或官方产品的立场。

**License**: MIT
