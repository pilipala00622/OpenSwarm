# OpenSwarm — 多智能体编排框架 v0.4.1

> 一个轻量级、可扩展的多 AI 智能体协作框架，对齐 Oh-My-OpenCode 核心功能。  

---

## 目录

- [项目概览](#项目概览)
- [v0.4.1 更新摘要](#v041-更新摘要)
- [项目结构](#项目结构)
- [安装与运行](#安装与运行)
- [核心概念](#核心概念)
  - [Agent 智能体](#agent-智能体)
  - [Rollout 执行循环](#rollout-执行循环)
  - [Tool 工具系统](#tool-工具系统)
  - [执行环境层：Runtime 与 Sandbox](#执行环境层runtime-与-sandbox)
- [七大核心功能](#七大核心功能)
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

Open Swarm 是一个多智能体编排框架，主 Agent 通过创建子 Agent 并分派任务来协作完成复杂工作。`v0.3.0` 对齐了 Oh-My-OpenCode 的核心功能集，`v0.3.1` 重点完成可靠性加固；当前 `v0.4.1` 在此基础上引入了 `parent/team-lite` 双模式协作、lead 侧 team 管理工具，以及本地知识引擎层。

**七大核心功能**：

| 功能 | 说明 |
|---|---|
| **Category System** | 8 种内置任务类别，每种映射到不同的模型/温度/推理强度 |
| **Tool 权限控制** | 黑名单、白名单、禁止委派三层工具访问控制 |
| **后台任务** | 异步启动子 Agent，主 Agent 不阻塞，稍后获取结果 |
| **Task 依赖** | 任务间 blockedBy/blocks 依赖关系，自动并行调度，含环检测 |
| **Handoff** | 跨会话上下文交接，自动提取已完成/待完成工作 |
| **外部记忆** | Anthropic 风格的 LLM 摘要 + 文件持久化上下文压缩 |
| **知识引擎** | 本地代码/文档索引 + 结构化上下文检索，为 Agent 提供更准确的工程上下文 |

## v0.4.1 更新摘要

| 能力 | 说明 |
|---|---|
| **team-lite 协作** | 支持 `parent` / `team` 双模式子级协作，新增 lead/team 消息、状态和 cleanup 工具 |
| **知识引擎层** | 新增 `KnowledgeEngine` 和 `retrieve_context`，为 Agent 提供本地结构化上下文检索 |
| **失败恢复增强** | team-mode 任务失败、取消、cleanup 时自动释放 managed task，避免卡死在旧 owner |
| **示例与文档** | 新增 `parent/team` demo、live team demo、knowledge engine demo，并同步更新说明文档 |
| **可靠性基线继承** | 继续沿用 `v0.3.1` 的并发限流、超时、原子写入、错误恢复与 token 统计能力 |

---

## v0.3.1 可靠性里程碑

> 2026-03-15 — 全面可靠性加固，修复 22 个问题（3 Critical / 4 High / 6 Medium / 4 Low / 5 Arch），为后续 `v0.4.1` 的 team-lite 与知识引擎演进打下稳定基线。

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
│   ├── retrieve_context.py      # RetrieveContextTool — 本地知识引擎检索
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
│   ├── knowledge_engine.py      # KnowledgeEngine — 本地代码/文档索引与结构化检索
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
├── containers/                  # 容器化开发/运行时基础设施（app/dev/runtime/e2b-sandbox）
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
| `retrieve_context` | `RetrieveContextTool` | 使用本地知识引擎检索相关代码、符号、文档和关联文件 |
| `verify_result` | `VerifyTool` | 使用 LLM 交叉验证事实 |
| `create_subagent` | `CreateSubagentTool` | 动态创建子 Agent 配置（支持 `overwrite` 更新） |
| `assign_task` | `TaskTool` | 分派任务给子 Agent（核心） |
| `background_output` | `BackgroundOutputTool` | 获取后台任务结果 |
| `background_cancel` | `BackgroundCancelTool` | 取消后台任务 |
| `task_create` | `TaskCreateTool` | 创建受管理的任务 |
| `task_claim` | `TaskClaimTool` | team 模式下原子认领下一个可执行任务 |
| `task_get` | `TaskGetTool` | 按 ID 查询任务 |
| `task_list` | `TaskListTool` | 列出所有活跃任务 |
| `task_update` | `TaskUpdateTool` | 更新任务状态 |
| `team_message` | `TeamMessageTool` | 向指定 teammate 或全队发送消息 |
| `team_inbox` | `TeamInboxTool` | 拉取当前 teammate 的待处理消息 |
| `team_lead_inbox` | `TeamLeadInboxTool` | lead 读取所有 teammate 发来的待处理消息 |
| `team_lead_message` | `TeamLeadMessageTool` | lead 向指定 teammate 或全队发送消息 |
| `team_members` | `TeamMembersTool` | 查看当前 team 已注册的成员 |
| `team_status` | `TeamStatusTool` | 查看当前 team id、成员和仍在运行的 team 后台任务 |
| `team_cleanup` | `TeamCleanupTool` | 清理 team mailbox 状态；必要时可强制取消 team 后台任务 |
| `handoff` | `HandoffTool` | 创建/加载跨会话交接文档 |

---

### 执行环境层：Runtime 与 Sandbox

在这个仓库里，`runtime` 指的不是某个 Python 类，而是 **Agent 实际执行代码、命令、文件操作时所处的环境**。  
也就是说，模型负责“决定做什么”，`TaskTool` / `Rollout` 负责“组织谁去做”，而 `runtime` 负责“这些动作究竟在哪里做、以什么权限做、能访问什么资源”。

对多 Agent 框架来说，`runtime` 通常决定：

- 代码和 shell 命令在哪里运行
- 文件修改发生在哪个工作区
- 进程使用什么用户身份和权限
- 能否隔离不同任务、不同用户或不同 team 的执行副作用

因此，`runtime` 更接近**执行层基础设施**，而不是推理层能力。

#### 当前仓库里的状态

当前 `v0.4.1` 的主线仍然是 **单进程 orchestration runtime**：

- lead 和子 Agent 主要是同一 Python 进程中的对象/协程
- `team-lite` 扩展的是协作语义，而不是进程级隔离
- `CodeRunnerTool` 当前仍以本地子进程方式执行 Python / JavaScript

这意味着本仓库已经具备比较完整的 **控制面（control plane）**：

- Agent 配置与工具权限
- Rollout 执行循环
- 子 Agent 分派与 team-lite 协作
- 任务依赖、handoff、memory、知识引擎

但**执行面（execution plane）** 仍然偏轻，`containers/` 更像是把框架进一步升级为可隔离执行系统的基础设施入口。

#### `containers/` 在做什么

`containers/` 目录不是主编排逻辑，而是一组围绕运行环境的容器资产：

- `containers/app/`：主应用镜像，负责打包前后端与服务入口
- `containers/dev/`：开发容器，提供一致的 Docker 开发环境
- `containers/runtime/`：给 Agent 执行动作准备的运行时镜像/沙箱镜像
- `containers/e2b-sandbox/`：把沙箱运行环境迁移到 E2B 的模板配置
- `containers/build.sh`：镜像构建与发布脚本

可以把它理解成：

- `agent/`、`rollout/`、`swarm_tool/`、`utils/`：大脑、调度器和状态系统
- `containers/`：工作环境、隔离边界和部署载体

#### 什么是 Runtime

在 agent 系统里，`runtime` 就是“实际干活的环境”。

没有 `runtime`，Agent 只能做推理和规划；  
有了 `runtime`，Agent 才能真正去：

- 运行 Python / JavaScript
- 执行 shell 命令
- 读写工作区文件
- 安装依赖、跑测试、调用本地工具

这也是为什么 `runtime` 往往决定一个 agent 系统能否从 demo 走向真实可用：它关系到安全性、一致性、可复现性和资源控制。

#### 什么是 Sandbox

`sandbox` 是 `runtime` 的一种具体形态，强调**隔离**。

典型目标包括：

- 不直接在宿主机无边界执行 Agent 生成的代码
- 将文件系统、副作用、权限和依赖限制在一个可控环境里
- 便于任务结束后回收环境，降低污染和风险

所以可以简单理解为：

- `runtime` = 执行环境这个总概念
- `sandbox` = 带隔离边界的 runtime

#### 什么是 E2B Sandbox

`E2B sandbox` 是一种**云端托管的安全执行环境**。  
它不是本地 Docker 容器本身，而是由 [E2B](https://e2b.dev) 提供的远程 sandbox，适合运行 AI 生成代码和 agent 任务。

在这个目录下，`containers/e2b-sandbox/` 提供的是一个 E2B 模板定义，意味着未来可以：

- 先把自定义 Dockerfile 构建成 E2B template
- 再通过 Python 或 JS SDK 动态拉起远程 sandbox
- 把 agent 的代码执行、命令执行、文件操作放进云端隔离环境里完成

它和本地 runtime 的关系可以概括为：

- `runtime` 是概念层：执行环境
- `sandbox` 是实现形态：强调隔离
- `E2B sandbox` 是具体后端：由 E2B 托管的远程隔离 runtime

#### 为什么这层重要

从架构上看，`containers/runtime` 和 `containers/e2b-sandbox` 的价值不在于“让模型更聪明”，而在于让系统更像一个真正可运行的 agent 平台：

- 让代码执行从“宿主机直跑”升级为“在可控环境里跑”
- 让不同机器上的依赖和执行行为更一致
- 为每个任务、每个 team、甚至每个用户提供更强的隔离边界
- 为未来的远程执行、多租户部署和资源配额管理打基础

因此，在本仓库里，`containers/` 的重要性主要体现在**执行层与部署层**。  
它不是当前 `v0.4.1` 的主路径核心，但它是框架从“单进程多 Agent 编排 demo”走向“可隔离执行的 agent 系统”的关键基础设施方向。

---

### 为什么工具层是核心

在 `agent-swarm` 里，模型负责推理，但工具决定：

- agent 能看到什么
- agent 能修改什么
- 多个 agent 如何协作
- 系统是否可恢复、可观测、可评测

所以工具层不是附属能力，而是整个框架的**执行层、状态层和认知层**。

可以把当前工具简单分成三类：

**1. 执行工具**

- `assign_task`
- `background_output`
- `background_cancel`
- `task_create`
- `task_update`

这些工具让 Agent 不只是“描述计划”，还能真正执行任务和推进流程。

**2. 协调工具**

- `task_claim`
- `team_message`
- `team_inbox`
- `team_lead_message`
- `team_lead_inbox`
- `team_members`
- `team_status`
- `team_cleanup`

这些工具让 `team-lite` 不会退化成一组互不相干的 worker，而是成为一个带共享状态和消息流的协作系统。

**3. 上下文工具**

- `search`
- `verify_result`
- `retrieve_context`

这些工具决定 Agent 在行动前能拿到什么证据和上下文。  
其中 `retrieve_context` 代表新的知识引擎层，它让 Agent 可以在本地代码和文档中先做结构化检索，再进入执行或协作。

一个实用判断标准是：

- 没有执行工具，Agent 只能“想”
- 没有协调工具，多 Agent 只能“并行”，不能“协作”
- 没有上下文工具，Agent 容易在错误或不足的信息上推理

因此，`agent-swarm` 的上限不仅取决于模型能力，也取决于工具系统是否足够清晰、可控和可组合。

---

### 知识引擎层

`KnowledgeEngine` 是新增的一层本地上下文基础设施，用来让 Agent 在执行任务前先拿到**更贴近代码结构的上下文**，而不是只依赖用户 prompt 或简单文本搜索。

当前实现提供：

- 本地文件索引：扫描代码和文档文件
- Python AST 元信息：提取函数、类、import 关系
- Markdown heading 索引：让架构文档和说明文档也可参与召回
- 轻量结构扩展：在 primary hit 之外补充 import/被引用关系的相关文件

可直接通过 `RetrieveContextTool` 使用：

```python
from open_swarm import KnowledgeEngine, RetrieveContextTool

engine = KnowledgeEngine(root_dir=".")
tool = RetrieveContextTool(knowledge_engine=engine)
```

适合的典型问题：

- “这个模块的 claim/release 流程在哪？”
- “team cleanup 相关逻辑有哪些文件？”
- “handoff 机制和消息历史恢复相关的代码在哪？”

注意：当前知识引擎是本地轻量实现，主要用于给 Agent 提供**更好的工程上下文**。它还不是完整的向量数据库或全量代码图谱系统，但已经能作为 `parent/team-lite` 协作上的一层检索底座。

---

## 七大核心功能

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

> **`v0.3.1` 线程安全改进**：LLM 摘要不再临时修改共享 `LLMClient` 的 `model_id`，而是通过 `model=` 参数覆盖传递，避免并发协程间的状态污染。

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

`v0.3.1` 引入了一系列生产就绪能力，提升系统在真实部署中的稳定性；`v0.4.1` 则在这条基线上继续扩展协作与上下文能力。

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

**增强的错误检测**（`v0.3.1` 基线）：

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
| `subagent_mode` | str | "parent" | 子级协作模式：`parent` 为传统父子汇报，`team` 为共享任务/消息协作 |

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

`TaskTool` 构造函数参数（当前 `v0.4.1` 版本）：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `agent_registry` | dict | 必填 | Agent 配置注册表 |
| `max_steps` | int | 20 | 子 Agent 默认最大步数 |
| `category_registry` | CategoryRegistry | None | 任务类别注册表 |
| `task_store` | TaskStore | None | 持久化任务管理器 |
| `max_concurrent_subagents` | int | 10 | 最大并发子 Agent 数（Semaphore 限流） |
| `subtask_timeout` | float | None | 单个子任务超时秒数（None = 无限制） |
| `team_mailbox` | TeamMailbox | None | team 模式下的共享消息邮箱（默认写入 `.agent_team/`） |

### 子 Agent 协作模式

当前 `v0.4.1` 中，lead agent 之下的子级协作支持两种模式：

| 模式 | 说明 | 典型行为 |
|---|---|---|
| `parent` | 传统父子模式 | 子 Agent 执行完后直接把结果回传给 lead |
| `team` | team-lite 协作模式 | 子 Agent 可使用共享任务池和 mailbox，与其他 teammate 协作 |

**设置方式**：

```python
config = AgentConfig(
    name="main",
    system_prompt="You are an orchestrator.",
    subagent_mode="team",   # 默认对子 Agent 使用 team 模式
)
```

也可以在单次调用 `assign_task()` 时覆盖：

```python
await task_tool.execute(
    prompt="实现解析器并与其他 teammate 协作",
    subagent_mode="team",
)
```

在 `team` 模式下，spawn 出来的子 Agent 会额外获得：

- `task_claim`：原子认领下一个 ready task，避免多个 teammate 抢同一任务
- `team_message`：给指定 teammate 发消息或广播
- `team_inbox`：读取自己的待处理消息

lead 侧推荐额外挂载以下工具：

- `team_lead_message`：lead 向指定 teammate 或全队发送消息
- `team_lead_inbox`：读取 teammate 发给 lead 的消息
- `team_members`：查看当前有哪些 teammate 已经注册到 team
- `team_status`：查看 team id、成员和仍在运行的 team-mode 后台任务
- `team_cleanup`：清理当前 team 的 mailbox 目录；如果还有 team-mode 后台任务，可用 `force=true` 强制取消并清理

注意：当前实现是 **team-lite**，仍然运行在同一 Python 进程内，但已经把子级协作语义从“只向父级汇报”扩展为“可共享任务、可彼此通信”。

**当前实现额外考虑的边界条件**：

- direct message 只会投递给已注册 teammate；不存在的 recipient 会明确返回失败
- lead broadcast / teammate broadcast 若当前没有有效收件人，也会返回失败而不是静默成功
- team-mode 任务失败、超时或取消时，会自动把 managed task 释放回 `pending`，避免任务卡死在原 owner 名下
- `team_cleanup(force=true)` 在取消后台 team 任务时，会同步释放对应 managed task，便于后续重试
- `team_status` 会展示当前仍在运行的 team 后台任务及其关联的 managed task id

### 适用场景

`parent` 和 `team` 不是替代关系，而是适合不同类型的问题：

**更适合 `parent` 模式的场景**：

- 单次调研、验证、代码生成这类“做完就回报”的窄任务
- 子任务之间不需要互相交流，只需要把结果汇总给 lead
- 成本敏感，希望减少额外状态和协调开销

**更适合 `team` 模式的场景**：

- 多个子任务可以并行推进，但过程中需要交换发现
- lead 需要中途广播策略、修正方向或追加约束
- 任务带依赖，需要成员自领任务、失败后重试、持续观察团队状态
- 代码审查、方案拆分、排障、多角色研究这类“边做边协作”的工作

**典型 team-lite 用法**：

- 一个 teammate 负责实现思路，另一个负责测试/风险分析
- 多个 teammate 并行验证不同 bug 假设，并向 lead 回报中间发现
- lead 创建任务池后，teammates 自领 ready task，失败时任务自动释放等待重做

### 评测建议

如果你想为 `parent/team-lite` 建一套评测，建议至少从下面几个维度切入。

**1. 任务完成质量**

- 最终答案是否正确、完整、可执行
- team 模式是否真的比 parent 模式带来更好的覆盖度或更低遗漏率
- 多 teammate 输出汇总后，是否比单一路径更稳健

**2. 协作有效性**

- teammate 是否真的使用了 `team_message` / `team_lead_message`
- `team_lead_inbox` 中的中间消息是否对最终决策有帮助，而不是噪音
- 是否存在无效广播、重复广播、互相没有消费消息的问题

**3. 任务调度与恢复能力**

- `task_claim` 是否避免了重复认领
- 依赖任务是否在 blocker 完成后正确解锁
- 子任务失败、超时、取消后，任务是否能自动回到 `pending`
- `team_cleanup(force=true)` 后是否还能重新启动一轮干净 team

**4. 状态可观测性**

- `team_members`、`team_status`、`team_lead_inbox` 是否足够解释当前团队状态
- lead 是否能仅凭这些工具判断“谁在做什么、谁卡住了、是否该 cleanup”

**5. 成本与效率**

- 总步数、总 tool calls、子任务数量
- parent vs team 在同类任务上的完成时间和 token 消耗
- team 模式的额外协调成本是否被结果质量提升抵消

**6. 稳定性与边界条件**

- 向不存在 teammate 发消息是否正确失败
- 没有收件人时 broadcast 是否返回合理结果
- cleanup 时仍有后台任务运行是否被正确阻止
- 后台任务取消后是否留下脏状态或僵尸 task

**推荐的评测组织方式**：

- 先做 `parent` vs `team` A/B 对照，固定同一任务集
- 每类任务至少覆盖：独立任务、依赖任务、失败重试任务、需要中途重定向任务
- 同时记录：最终质量、消息数量、任务状态流转、时间、token、人工主观评分

**一套最小评测集可以从这 4 类开始**：

1. 双人并行研究：两个 teammate 分别从不同角度分析同一问题
2. 依赖流水线：A 完成后 B 才能做，验证 task unblock
3. 中途改方向：lead 在执行中用 `team_lead_message` 改变要求
4. 失败恢复：故意让一个 teammate 超时或取消，观察 task 是否释放并可重做

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
        max_concurrent_subagents=10,     # v0.4.1: 并发限流
        subtask_timeout=300.0,           # v0.4.1: 子任务超时 5 分钟
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

本仓库是**个人性质的实验性项目**，仅用于 demo / 参考实现。当前 `v0.4.1` 已具备较完整的可靠性与协作能力（并发控制、超时、原子写入、错误恢复、team-lite、知识引擎等），但尚未经过大规模生产验证。成本控制、多租户隔离等企业级功能未系统性处理。项目不代表任何公司或官方产品的立场。

**License**: MIT


## 0316 补充和Claude Team的差别
现有框架：主 Agent 通过工具调用拉起子 Agent，子 Agent 是同一 Python 运行时里的对象/协程，主要向父级回传结果。

Claude teams：lead 和 teammates 都是独立 Claude Code 会话，成员之间可直接通信，并通过共享任务板自协调。


Claude teams 的 task list 则更接近一个轻量调度器：

有 pending / in progress / completed
有依赖阻塞
有 self-claim
有 lead assign
有 file locking 防止多人抢同一个 task

Claude teams 的 teammate 是独立会话，但你当前框架是单进程 runtime。
第一版完全可以接受：

仍然是 in-process agent
但在协作语义上模拟 team
也就是先做 team-lite。

## 0317 如何提高基线模型的效果，baseline的提升

## 0325 创新点：单agent训练，多agent环境生成
1. 环境变得动态：原版 Env LLM 更像一个单点采样器。多智能体后，环境变成一个有内部分工的系统，可以围绕“边界难度”主动构造课程，而不是单次采样后直接喂给训练。

2. 质量控制内生环境中，以前通常是 生成 -> 粗过滤 -> 入库。多智能体后可以变成 生成 -> 试做 -> 反驳 -> 验证 -> 去重 -> 定级 -> 入库。这意味着环境本身带有“自我质检”能力，这在方法上是一个很实在的提升。

3. GenEnv 的核心目标本来就是边界任务。多智能体环境的创新点在于：不只是事后挑出边界题，而是让环境团队在生成阶段就围绕“卡住但非无解”这个目标协作。

【如果人工标注】核心是：人工校验的核心作用，不是替模型逐题打补丁，而是验证整个多智能体环境生成流程是否真的在产生“正确、边界、有效”的训练分布。

### 用 CreativeBench 风格做环境生成模板

这是一个中等强度、但更接近研究创新的结合方式。核心思想不是直接把 `CreativeBench` benchmark 样本当作训练集，而是把它当作“任务构型来源”与“风格锚点”，让多智能体环境去学习其中的组合式创造模式，再自动生成**同风格但不重合**的新任务；最终评测时，仍然使用原始 `CreativeBench` 作为 held-out benchmark。

这样做的关键约束是：

- 不直接拿 benchmark 样本做监督训练，避免数据污染和 benchmark 泄漏。
- 让环境团队学习任务的结构模式，而不是记忆题面或 canonical solution。
- 新生成任务必须与原始 benchmark 在题面、约束组合、参考实现层面保持非重合。
- 最终仍然在原始 `CreativeBench` 上测试，验证收益是否来自“学会构造创造性训练分布”，而不是“看过类似题”。

这个版本最能证明的点是：**multi-agent environment 学到的不是单个题目的答案，而是如何系统性地构造创造性、可验证、且对单 Agent 有训练价值的任务分布。**

### 为什么方案 B 比直接拿 bench 训练更合理

- benchmark 的作用应该是检验泛化，而不是直接参与监督。
- `CreativeBench` 更适合提供“任务风格模板”，例如多约束组合、实现空间开放、允许多种正确解、同时可执行验证。
- 多智能体环境比单一 Env LLM 更适合学习这种风格，因为它可以把“出题、试做、审题、定级、去重”拆成多个角色并行完成。

### 方案 B 的环境团队分工

- `generator`：基于 `CreativeBench` 的任务风格生成候选题，重点学习“组合式创造”而不是复写题面。
- `composer`：重新组合约束、数据结构、边界条件和接口要求，形成同风格新题。
- `solver`：先尝试给出一个可执行参考解，过滤无解题、脏题和描述不完整的题。
- `critic`：检查任务是否清晰、是否可验证、是否与已有题目过于相似。
- `difficulty calibrator`：判断任务对当前单 Agent 是过易、边界还是过难，只保留最有学习价值的样本。

环境侧可以形成这样一条流水线：

`CreativeBench 风格归纳 -> 候选任务生成 -> 参考解试做 -> 相似度/重复度过滤 -> 难度校准 -> 入训练池`

### 研究假设

如果方案 B 成立，那么即使不直接使用 benchmark 样本做训练，单 Agent 在原始 `CreativeBench` 上依然应该表现出：

- 更高的 `pass@1`
- 更高的有效创造性分数（例如“执行成功前提下的新颖度”）
- 更好的组合泛化能力，而不是只会复述参考解

### 方案 B 的方法贡献可以这样表述

1. **Single-Agent Policy, Multi-Agent Environment**：保持训练对象仍为单 Agent，避免多智能体 RL 带来的 credit assignment 爆炸；创新集中在环境生成侧。
2. **Creative Style Distillation Without Benchmark Leakage**：不直接训练 benchmark 样本，而是蒸馏任务构型与风格特征，用于生成同风格但不重合的新任务。
3. **Self-Verified Curriculum Construction**：新样本必须经过多角色协作验证后才能进入训练集，从而提高课程质量与样本有效性。
4. **Held-Out Creative Generalization**：最终仍在原始 `CreativeBench` 上评测，以验证环境是否真正学会了构造创造性训练分布。

### 实验上如何讲清楚

最自然的对比方式是：

- baseline A：原始 `GenEnv`，单一 Env LLM 生成训练任务
- baseline B：直接用 `CreativeBench` 风格 prompt 做单模型改写生成
- method：multi-agent environment 学习 `CreativeBench` 风格并生成非重合新任务

最终统一在原始 `CreativeBench` 上评测。如果 method 优于 baseline，且训练阶段未直接使用 benchmark 样本，就可以更有力地说明：提升来自环境团队学会了构造高价值的创造性课程，而不是数据记忆。