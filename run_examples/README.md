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
