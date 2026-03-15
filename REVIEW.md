# Agent-Swarm v0.3.0 — 深度审阅报告

> 审阅日期：2026-03-15
> 范围：全部源码（agent/, rollout/, swarm_tool/, tool/, utils/）
> 修复状态：2026-03-15 全部修复完成

---

## 一、Critical 级问题

### 1. 子 Agent 的 Category 模型配置形同虚设

**文件**：`swarm_tool/task.py` 第 269-272 行

```python
subagent = Agent(
    config=subagent_config,   # model_id = "gpt-5.4"（来自 category）
    tools=subagent_tools,
    llm_client=self.parent_agent.llm_client if self.parent_agent else None,
    #          ↑ 这个 client 的 model_id 是父 agent 的 "kimi-k2.5"
)
```

`Agent.__init__` 中，如果传入了 `llm_client`，直接赋值使用，不会根据 `config.model_id` 创建新 client。而 `process_message()` 调用的是 `self.llm_client.chat()`，其中 `model=self.llm_client.model_id`——永远是父 agent 的模型。

**后果**：Category 系统精心配置的 8 种模型（ultrabrain → gpt-5.4, quick → claude-haiku-4-5 等）**全部不生效**，所有子 agent 实际都用父 agent 的模型。

**修复方案**：

```python
# 在 _run_subtask 中，判断模型是否与父 agent 不同
if (self.parent_agent and
    (model_id != self.parent_agent.config.model_id or
     api_key != self.parent_agent.config.api_key or
     base_url != self.parent_agent.config.api_base_url)):
    # 模型不同，创建独立 LLM client
    from ..utils.llm_client import LLMClient
    llm_client = LLMClient(model_id=model_id, api_key=api_key, base_url=base_url)
else:
    llm_client = self.parent_agent.llm_client if self.parent_agent else None

subagent = Agent(config=subagent_config, tools=subagent_tools, llm_client=llm_client)
```

---

### 2. LLM 全部重试失败 → 被错误地当作正常完成

**文件**：`utils/llm_client.py` 第 67-72 行 + `rollout/base.py` 第 49-53 行

当 `LLMClient.chat()` 三次重试全部失败时：

```python
return {
    "content": f"Error: {str(last_error)}",
    "finish_reason": "error",
}
```

Rollout 的 `_is_complete()` 判断：

```python
def _is_complete(self, response):
    if response.get("content") and not response.get("tool_calls"):
        return True   # ← "Error: ..." 有 content，无 tool_calls → True
```

**后果**：LLM 的网络错误、API key 过期等严重问题被当作"正常回答"返回给用户。Rollout 认为任务成功完成，`RolloutResult.status = COMPLETED`。

**修复方案**：

```python
def _is_complete(self, response):
    if response.get("finish_reason") == "error":
        return False  # 不要把错误当完成
    if response.get("content") and not response.get("tool_calls"):
        return True
    if response.get("finish_reason") == "stop":
        return True
    return False
```

同时在 MainRollout/SubRollout 的循环中加上对 `finish_reason == "error"` 的检测，触发错误重试逻辑。

---

### 3. 空回复导致无限循环

**文件**：`rollout/base.py` + `rollout/main_rollout.py`

如果 LLM 返回 `{"content": "", "tool_calls": null, "finish_reason": "length"}` 或者任何 content 为空串（falsy）且无 tool_calls 的情况：

- `_is_complete()` → `content` 是空串（falsy） → 第一个条件 False
- `finish_reason != "stop"` → 第二个条件 False
- 返回 `False`

Rollout 会 `current_step += 1` 然后继续循环，每步都重新把完全相同的消息列表发给 LLM，很可能得到一模一样的空回复——直到 `max_steps` 耗尽。

**修复方案**：

```python
# 在 rollout 循环中，检测空回复
if not response.get("content") and not response.get("tool_calls"):
    self.consecutive_errors += 1
    self.messages.append({
        "role": "user",
        "content": "[System] Received empty response. Please provide an answer or use a tool."
    })
    self.current_step += 1
    continue
```

---

## 二、High 级问题

### 4. 并发 assign_task 导致 subagent_counter 竞态

**文件**：`swarm_tool/task.py` 第 366-367 行

```python
self.subagent_counter += 1
subagent_id = f"subagent_{self.subagent_counter}"
```

`Agent.execute_tool_calls()` 用 `asyncio.gather()` 并行执行所有 tool calls。当主 agent 在一个 turn 中调用了 3 个 `assign_task`，三个协程几乎同时执行 `self.subagent_counter += 1`。虽然 asyncio 是单线程的，`+=` 本身是原子的（因为 GIL + 事件循环），**但问题出在 background 模式**下：`_on_complete` 回调访问 `self.sub_results` 和 `self._background_results` 时，与正在执行的其他协程存在竞态。

**修复方案**：使用 `asyncio.Lock` 保护关键段，或使用 `itertools.count()` 作为计数器。

---

### 5. Memory 压缩时直接修改共享 LLM Client 的 model_id

**文件**：`utils/memory.py` 第 89-97 行

```python
original_model = self._llm_client.model_id
self._llm_client.model_id = model_override   # ← 修改共享对象
response = await self._llm_client.chat(...)   # ← await 期间其他协程可能用到这个 client
self._llm_client.model_id = original_model    # ← 恢复
```

如果在 `await` 挂起期间，另一个协程（比如主 rollout 的 LLM 调用）也使用同一个 `llm_client`，它会用到被临时修改的 model_id。

**修复方案**：让 `LLMClient.chat()` 接受可选的 `model` 参数覆盖，而不是修改实例状态：

```python
async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096, model=None):
    kwargs = {"model": model or self.model_id, ...}
```

---

### 6. TaskStore 文件写入无原子性保证

**文件**：`utils/task_store.py`

`_save_task()` 直接 `open(filepath, "w")` 然后 `json.dump()`。如果进程在写入中途崩溃（或多个协程并发写同一个文件），JSON 文件可能被截断或损坏。

**修复方案**：write-to-temp-then-rename 模式：

```python
def _save_task(self, task):
    filepath = self._task_path(task.id)
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, filepath)  # 原子替换
```

---

### 7. Task 依赖图没有环检测

**文件**：`utils/task_store.py` `create()`

可以创建 A → blockedBy B → blockedBy A 的循环依赖，导致两个 task 永远无法变成 ready。

**修复方案**：在 `create()` 中做 DFS/BFS 检测：

```python
def _has_cycle(self, task_id, blocked_by):
    visited = set()
    stack = list(blocked_by)
    while stack:
        current = stack.pop()
        if current == task_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        blocker = self._tasks.get(current)
        if blocker:
            stack.extend(blocker.blocked_by)
    return False
```

---

## 三、Medium 级问题

### 8. AgentConfig 有多个配置字段从未传递给 LLM

| 字段 | 在 `process_message` 中使用? | 在 `LLMClient.chat` 中支持? |
|------|:---:|:---:|
| `temperature` | Yes | Yes |
| `max_tokens` | Yes | Yes |
| `model_id` | No（用的是 llm_client.model_id） | N/A |
| `top_p` | **No** | **No** |
| `reasoning_effort` | **No** | **No** |
| `thinking_budget` | **No** | **No** |

这些字段在 Category 配置中也有设置（如 `ultrabrain` 设了 `thinking_budget=32000`），但完全无效。

**修复方案**：在 `LLMClient.chat()` 中加入这些参数的透传，并在 `Agent.process_message()` 中传递。

---

### 9. fork_context 的读取时机不确定

**文件**：`swarm_tool/task.py` `_build_forked_context()`

`_parent_messages_ref` 是 `MainRollout.messages` 列表的直接引用。当 `asyncio.gather` 并行执行多个 `assign_task` 时，每个子任务读取 fork context 的时机不同——先执行的子 agent 可能看不到后面 tool call 的结果，而后执行的可能看到部分结果。

**修复方案**：在 MainRollout 的 `execute_tool_calls` 之前做一次消息快照，传递给 TaskTool。

---

### 10. Background task 的 CancelledError 在 Python 3.9+ 无法被捕获

**文件**：`swarm_tool/task.py` 第 385-397 行

```python
def _on_complete(future, sid=subagent_id):
    try:
        result_dict = future.result()
    except Exception as e:  # CancelledError 继承 BaseException，不在这里
        ...
```

Python 3.9 以后 `CancelledError` 继承自 `BaseException` 而非 `Exception`。当 background task 被 cancel 时，`_on_complete` 会抛出未捕获的 `CancelledError`。

**修复方案**：

```python
except (Exception, asyncio.CancelledError) as e:
```

---

### 11. Handoff 的 completed_work 和 remaining_work 永远为空

**文件**：`utils/handoff.py` 第 101-104 行

```python
completed = []
remaining = []
# ... 没有任何代码填充这两个列表
```

HandoffDocument 有专门的字段 `completed_work` 和 `remaining_work`，`to_context_message()` 也会渲染它们，但创建逻辑中从未填充。

**修复方案**：结合 task_snapshot 和消息内容自动推断，或让 LLM summary 来提取。

---

### 12. JSON 解析失败时静默使用空参数 {}

**文件**：`agent/agent.py` 第 108-112 行

```python
try:
    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
except json.JSONDecodeError:
    arguments = {}
```

LLM 偶尔返回格式有误的 JSON（特别是小模型），工具会收到空参数调用，可能执行意外操作。用户和 LLM 都无法感知参数解析失败了。

**修复方案**：

```python
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
    return {
        "tool_call_id": tc.get("id", tool_name),
        "role": "tool",
        "content": f"Error: Could not parse arguments. Raw: {args_str[:200]}",
    }
```

---

### 13. CreateSubagentTool 不支持更新已有 agent

**文件**：`swarm_tool/create_subagent.py` 第 71-75 行

```python
if name in self.agent_registry:
    return ToolResult(
        content=f"Agent '{name}' already exists. Use it directly...",
        success=True
    )
```

如果 LLM 想修改已有 agent 的 system prompt，只能得到一个"已存在"的提示，没有 update/overwrite 机制。

**修复方案**：增加可选的 `overwrite` 参数，或新增 `update_subagent` 工具。

---

## 四、Low 级问题

### 14. Checkpoint 列表无上限

**文件**：`rollout/base.py`

`self.checkpoints.append(copy.deepcopy(self.messages))` — 对于 50 步的 rollout（checkpoint_interval=10），会保存 5 个消息快照。如果每个快照 50+ 条消息、每条包含大量 tool output，内存占用显著。

**建议**：保留最近 2-3 个 checkpoint 即可，类似 ring buffer。

---

### 15. `_save_result` 对相对路径的边缘处理

**文件**：`rollout/main_rollout.py` 第 178-183 行

```python
if storage_path.endswith(".jsonl"):
    filepath = storage_path
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
```

如果 `storage_path = "result.jsonl"`（无目录前缀），`os.path.dirname` 返回空串 `""`，`os.makedirs("", exist_ok=True)` 在某些平台会报错。

---

### 16. datetime.utcnow() 已被 Python 3.12 弃用

**文件**：`utils/task_store.py`, `utils/handoff.py`

应迁移到 `datetime.now(datetime.UTC).isoformat()`。

---

### 17. 直接访问私有属性 `memory._llm_client`

**文件**：`rollout/main_rollout.py` 第 51 行

```python
self.memory._llm_client = agent.llm_client
```

违反封装原则。`AgentMemory` 应提供 `set_llm_client()` 公开方法。

---

## 五、架构层面改进建议

### 18. 缺少 Timeout / 超时控制

当前没有对单个子 agent 执行的总时长做限制。如果某个 LLM 调用卡住（超过 httpx 的 120s timeout 之外的网络层问题），或子 agent 陷入某种工具调用死循环，没有机制能打断它。

**建议**：在 `_run_subtask` 中加 `asyncio.wait_for(coro, timeout=total_timeout)` 包装。

---

### 19. 缺少 Token/Cost 统计

当前 `RolloutTracer` 记录了 LLM 调用次数和工具执行次数，但没有记录 token 消耗。对于一个可能动态分配 5-10 个子 agent 的系统，token 消耗是关键运维指标。

**建议**：在 `LLMClient.chat()` 中提取 `response.usage`（`prompt_tokens`, `completion_tokens`），通过 tracer 或返回值向上汇报。

---

### 20. 缺少全局 Agent 并发上限

Scaling Rules 是一段文本 prompt，靠 LLM 自觉遵守。如果 LLM 不听劝、在一个 turn 里发起 20 个 assign_task，系统会全部并行执行——20 个并发 LLM 调用，可能触发 rate limit 或 OOM。

**建议**：在 `TaskTool` 中加一个 `max_concurrent_subagents` 参数，用 `asyncio.Semaphore` 限流：

```python
self._semaphore = asyncio.Semaphore(max_concurrent_subagents)

async def _run_subtask(self, ...):
    async with self._semaphore:
        # ... 原有逻辑
```

---

### 21. 缺少子 Agent 结果的质量校验

子 agent 返回的 content 直接作为 tool result 传回主 agent，没有任何校验。如果子 agent 超步返回的是"我步数用完了，还没结论"，主 agent 可能把这当作有效结果。

**建议**：对于 `max_steps_reached` 的结果，在返回内容中加上明确的标记，或者让主 agent 的 system prompt 中说明如何处理这种情况。

---

### 22. Handoff 的 load 操作只返回文本，不注入到消息历史

`HandoffTool._load_handoff()` 返回的是一个 ToolResult（文本），被追加为 tool response 消息。但 handoff 文档中的 `context_messages` 并没有被真正注入到 rollout 的消息历史中——只是被序列化成文本展示。

如果目标是"跨会话恢复上下文"，应该把 `context_messages` 作为真正的消息历史注入，而不是扁平化成一段文字。

---

### 23. 没有 Graceful Shutdown 机制

当 `interrupt()` 被调用时，只是设 `self.interrupted = True`，然后在下一个循环迭代中检查。但如果 agent 正在执行一个长时间的工具调用（比如子 agent 的同步执行），这个 interrupt 要等到工具返回后才能生效。

**建议**：对于 background tasks，interrupt 时主动 cancel 所有正在运行的 background tasks。

---

### 24. 缺少 Agent Registry 的持久化

`CreateSubagentTool` 创建的 agent 配置存在内存 dict 中。如果与 Handoff 配合使用（跨会话恢复），恢复后之前创建的 subagent 配置全部丢失。

**建议**：agent_registry 也做文件持久化，或在 handoff 中保存 registry 快照。

---

## 六、汇总表

| 序号 | 严重度 | 问题 | 文件 | 修复状态 |
|:---:|:---:|------|------|:---:|
| 1 | **Critical** | 子 agent 共享父 LLM client，category 模型不生效 | `swarm_tool/task.py` | **已修复** |
| 2 | **Critical** | LLM 全部重试失败被误判为正常完成 | `llm_client.py` + `base.py` | **已修复** |
| 3 | **Critical** | 空回复导致无限循环直至 max_steps | `rollout/base.py` | **已修复** |
| 4 | **High** | subagent_counter / sub_results 并发竞态 | `swarm_tool/task.py` | **已修复** |
| 5 | **High** | Memory 压缩修改共享 LLM client 的 model_id | `utils/memory.py` | **已修复** |
| 6 | **High** | TaskStore 文件写入无原子性 | `utils/task_store.py` | **已修复** |
| 7 | **High** | Task 依赖图无环检测 | `utils/task_store.py` | **已修复** |
| 8 | **Medium** | reasoning_effort / thinking_budget / top_p 配置无效 | `agent/agent.py` | **已修复** |
| 9 | **Medium** | fork_context 读取时机不确定 | `swarm_tool/task.py` | **已修复** |
| 10 | **Medium** | CancelledError 在 Py3.9+ 无法被捕获 | `swarm_tool/task.py` | **已修复** |
| 11 | **Medium** | Handoff completed/remaining_work 始终为空 | `utils/handoff.py` | **已修复** |
| 12 | **Medium** | JSON 解析失败静默传空参数 | `agent/agent.py` | **已修复** |
| 13 | **Medium** | CreateSubagent 不支持更新已有 agent | `create_subagent.py` | **已修复** |
| 14 | **Low** | Checkpoint 列表无上限 | `rollout/base.py` | **已修复** |
| 15 | **Low** | `_save_result` 相对路径边缘 | `main_rollout.py` | **已修复** |
| 16 | **Low** | `datetime.utcnow()` deprecated | 多个文件 | **已修复** |
| 17 | **Low** | 直接访问 `_llm_client` 私有属性 | `main_rollout.py` | **已修复** |
| 18 | **Arch** | 缺少子 agent 总超时控制 | 架构 | **已修复** |
| 19 | **Arch** | 缺少 token/cost 统计 | 架构 | **已修复** |
| 20 | **Arch** | 缺少全局 agent 并发上限 | 架构 | **已修复** |
| 21 | **Arch** | 缺少子 agent 结果质量校验 | 架构 | **已修复** |
| 22 | **Arch** | Handoff load 只返回文本，不注入消息历史 | 架构 | TODO |
| 23 | **Arch** | 缺少 Graceful Shutdown 机制 | 架构 | **已修复** |
| 24 | **Arch** | Agent Registry 无持久化 | 架构 | TODO |

---

## 七、优先修复建议

**第一轮（必须修）**：#1、#2、#3 — 这三个是会导致用户直接感知到系统出错的 bug。

**第二轮（尽快修）**：#5、#7、#8 — Memory 竞态、依赖环、配置字段失效。

**第三轮（完善）**：#18、#19、#20 — Timeout、Token 统计、并发限流，上生产前必须有。
