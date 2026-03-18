# Agent Teams 核心总结

> 来源：[Claude Code Agent Teams 文档](https://code.claude.com/docs/en/agent-teams)

## 1. Agent Teams 是什么

`Agent teams` 是 Claude Code 里的多智能体协作机制，用来协调多个 Claude Code 实例一起完成复杂任务。

它的基本结构包括：

- `team lead`：主会话，负责建队、分工、协调和汇总
- `teammates`：多个独立 Claude Code 会话，各自处理任务
- `task list`：共享任务列表，支持依赖和认领
- `mailbox`：成员之间的消息系统

它和普通 `subagents` 的最大区别是：

- `subagents` 只向主 agent 回报
- `agent teams` 的成员之间可以直接沟通，并围绕共享任务列表协作

## 2. 什么时候适合用

文档推荐的高价值场景有四类：

- 研究与评审：多个成员从不同角度并行分析同一问题
- 新模块或新功能：不同 teammate 各自负责一块，减少相互干扰
- 多假设并行调试：不同 teammate 验证不同根因，并互相质疑
- 跨层协作：前端、后端、测试等不同层面的任务并行推进

不适合的典型情况：

- 顺序依赖很强的任务
- 多人容易改同一文件的任务
- 很小的任务
- 仅需要“做完就回报”的窄任务

## 3. 与 Subagents 的区别

| 维度 | Subagents | Agent teams |
|---|---|---|
| 上下文 | 独立上下文窗口，但结果回到主 agent | 独立上下文窗口，成员彼此独立 |
| 通信 | 只向主 agent 回报 | 成员之间可直接消息通信 |
| 协调 | 主 agent 全权管理 | 共享任务列表，自协调 |
| 更适合 | 结果导向的窄任务 | 需要讨论、协作、挑战的复杂任务 |
| 成本 | 较低 | 较高 |

可以简单理解为：

- `subagents` 更像短工 worker
- `agent teams` 更像带共享任务板和消息系统的项目组

## 4. 如何启动

Agent teams 默认关闭，需要开启实验开关：

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

然后可以直接用自然语言要求 Claude 创建 team，例如：

```text
Create an agent team to explore this from different angles:
one teammate on UX, one on architecture, one as devil's advocate.
```

Claude 会：

1. 创建 team
2. 生成 teammates
3. 建立共享 task list
4. 并行推进任务
5. 汇总结果
6. 最后尝试 cleanup

## 5. 控制方式

### 显示模式

支持两种模式：

- `in-process`：所有 teammates 在当前主终端里运行
- `split panes`：每个 teammate 一个 pane，需要 `tmux` 或 iTerm2

### 指定人数和模型

可以要求：

- 建几个 teammate
- 每个 teammate 用什么模型

### 要求先 plan 再执行

对于高风险任务，可以要求 teammate 先做 plan，lead 审批通过后再执行。

### 直接和 teammate 对话

用户可以直接与某个 teammate 交互，而不必永远通过 lead 中转。

### 任务分配方式

任务列表支持：

- `lead assigns`：lead 显式派发任务
- `self-claim`：teammate 自己认领未阻塞任务

任务支持依赖关系，被阻塞的任务在前置任务完成前不能认领。

## 6. 核心机制

### 架构

一个 team 由以下几部分组成：

- `lead`
- `teammates`
- `task list`
- `mailbox`

### 上下文

每个 teammate 有独立上下文窗口。

teammate 会加载：

- 项目里的 `CLAUDE.md`
- MCP servers
- skills
- lead 生成时给它的 spawn prompt

但 **不会继承 lead 的对话历史**。

### 通信

消息机制包括：

- `message`：发给指定 teammate
- `broadcast`：发给所有 teammates

消息会自动投递，不需要轮询。

### 任务协调

共享任务列表负责：

- 维护 `pending / in_progress / completed`
- 处理任务依赖
- 支持成员自领任务
- 通过文件锁避免多人同时 claim 同一个任务

## 7. 成本与权衡

Agent teams 的代价比单会话或 subagent 高得多：

- 每个 teammate 都是独立 Claude 实例
- 每个实例都有自己的上下文窗口
- token 消耗会随 active teammates 增长

因此文档建议：

- 简单任务不要上 team
- 真正需要并行探索和协作收敛时再用

## 8. 最佳实践

文档建议的重点做法包括：

- 给 teammates 足够具体的 spawn prompt
- 从 `3-5` 个 teammates 起步
- 每个 teammate 保持 `5-6` 个任务规模较合适
- 优先从 research/review 这类边界清楚的任务开始
- 避免多个 teammate 改同一个文件
- 持续监控和纠偏，不要长时间完全放养

## 9. 常见问题

常见故障点包括：

- teammate 没出现：任务可能不够复杂，或显示模式环境不满足
- 权限弹窗太多：需要提前配置权限
- teammate 出错后停住：需要直接干预或重新生成 replacement teammate
- lead 过早认为任务完成：需要明确要求它继续等待 teammates
- cleanup 失败：通常因为仍有 active teammates

## 10. 当前限制

文档明确指出这是 experimental 功能，主要限制包括：

- 不支持嵌套 teams
- 一个 session 同时只能管理一个 team
- lead 身份不能转移
- teammate 启动时继承 lead 权限，不能在 spawn 时逐个细分
- shutdown 可能较慢
- task status 可能滞后
- `in-process` teammates 不能随 `/resume` 恢复
- split-pane 依赖 `tmux` 或 iTerm2，兼容环境有限

## 11. 关键结论

理解 `agent teams`，最重要的是抓住下面这句：

**它不是“多开几个 subagent”，而是“一个 lead + 多个独立 teammate + 共享任务板 + 消息系统”的协作框架。**

所以：

- 如果你只需要快速分派几个窄任务，用 `subagents`
- 如果你需要多成员并行探索、交换发现、共同收敛结果，用 `agent teams`
