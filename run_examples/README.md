# Run Examples

Example scripts for running the multi-agent rollout framework.

## Environment Variables

Set the following environment variables before running:

```bash
# Required for kimi models
export KIMI_API_KEY="your-kimi-api-key"

# Required for qwen models 
export OPENAI_API_KEY="your-openai-api-key"

# Optional: for web search functionality
export SERPER_API_KEY="your-serper-api-key"
```

## Examples

### 1. Kimi + Kimi (Main and Sub both use kimi-k2.5)

```bash
export KIMI_API_KEY="sk-xxx"
export SERPER_API_KEY="xxx"  # optional
python run_kimi_kimi.py
```

This configuration uses kimi-k2.5 for both the main agent and sub-agents.
- More thorough results with multiple iterations
- Supports reasoning_content (thinking mode)

### 2. Eval API 适配（用 llm.py 的 data_eval/混元 接口）

不设 `KIMI_API_KEY`，改用项目内 `Agent_swarm/llm.py` 的 `get_model_answer` 跑单智能体对话（仅测试 **kimi-k2.5**；当前 eval 接口不支持 tools，仅纯文本一问一答）：

```bash
cd Agent_swarm
python swarm_example/run_examples/run_eval_api.py
```

需保证能 `import llm`（在 `Agent_swarm` 目录下执行，或把 `Agent_swarm` 加入 `PYTHONPATH`）。

**多智能体（create_subagent + assign_task）**：当前 eval API 不支持 tools，多智能体只会退化为单轮文本回复。可跑 `run_eval_api_multiagent.py` 观察该行为；要真正跑多智能体请用支持 function calling 的接口（见下方 Kimi + Kimi / Kimi + Qwen）。

### 3. Kimi + Qwen (Main uses kimi-k2.5, Sub uses qwen2.5-72b)

```bash
export KIMI_API_KEY="sk-xxx"
export OPENAI_API_KEY="sk-xxx"
export SERPER_API_KEY="xxx"  # optional
python run_kimi_qwen.py
```

This configuration uses kimi-k2.5 for the main agent and qwen2.5-72b-instruct for sub-agents.
- Faster execution with fewer steps
- Good for tasks requiring quick parallel processing

### 4. Parent vs Team demo (no external API needed)

```bash
python3 run_parent_team_demo.py
```

This demo uses a tiny in-memory fake LLM client and shows:
- `parent` mode: sub-agent finishes and reports directly to the lead
- `team` mode: teammates broadcast progress, lead can inspect `team_members` / `team_status`,
  then read `team_lead_inbox` and run `team_cleanup`

### 5. Team live demo (real function-calling model)

```bash
export OPENAI_API_KEY="sk-..."
export OPEN_SWARM_MODEL="gpt-4o-mini"   # optional
python3 run_team_live_demo.py
```

This demo uses the real `LLMClient` and asks the lead agent to:
- create managed tasks
- spawn teammates with `assign_task(..., subagent_mode="team")`
- send guidance via `team_lead_message`
- inspect coordination state with `team_members` and `team_status`
- read teammate updates with `team_lead_inbox`
- finish with `team_cleanup`

### 6. Knowledge engine demo (no external API needed)

```bash
python3 run_knowledge_engine_demo.py
```

This demo shows the new local knowledge engine layer:
- indexes local code and markdown files
- extracts Python defs/imports and markdown headings
- returns ranked context hits plus related files
- useful for architecture questions before spawning agents

## Why Tools Matter

These examples are intentionally organized around tools rather than only around models.

In `agent-swarm`, the model is responsible for reasoning, but tools determine:

- what the agent can observe
- what the agent can change
- how multiple agents coordinate
- whether the system is recoverable and debuggable

You can think of the current tool layer in three parts:

### 1. Execution tools

These let the agent take action:

- `assign_task`
- `background_output`
- `background_cancel`
- `task_create`
- `task_update`

Without them, the agent can only describe a plan, not execute one.

### 2. Coordination tools

These make multi-agent workflows possible:

- `task_claim`
- `team_message`
- `team_inbox`
- `team_lead_message`
- `team_lead_inbox`
- `team_members`
- `team_status`
- `team_cleanup`

Without them, `team` mode would degrade into a set of isolated workers with no shared state.

### 3. Context tools

These improve the quality of reasoning before execution:

- `search`
- `verify_result`
- `retrieve_context`

`retrieve_context` is especially important because it adds a lightweight knowledge engine layer:
before an agent acts, it can first retrieve relevant code, symbols, docs, and related files from the local repository.

In practice, better tools usually matter more than simply adding more agents.  
More agents without task, status, and context tools often just means more noise.  
Well-designed tools give the lead and teammates a reliable execution surface.

## Output

Results are saved to the `result/` directory as JSONL files:
- `kimi_kimi_result.jsonl` - Results from kimi+kimi configuration
- `kimi_qwen_result.jsonl` - Results from kimi+qwen configuration

Each file contains:
```json
{
  "main": [...],  // Main agent conversation messages
  "subs": [...]   // Sub-agent conversation records
}
```

## Customization

To use your own query, modify the `query` variable in the scripts:

```python
query = "Your custom task description here"
```

To change the model or API endpoint, modify the `AgentConfig`:

```python
config = AgentConfig(
    model_id="your-model",
    api_key=os.environ.get("YOUR_API_KEY"),
    api_base_url="https://your-api-endpoint/v1",
    subagent_model_id="sub-model",
    subagent_api_key=os.environ.get("SUB_API_KEY"),
    subagent_api_base_url="https://sub-api-endpoint/v1",
)
```
