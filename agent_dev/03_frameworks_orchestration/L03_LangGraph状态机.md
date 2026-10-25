# 第 3 课：LangGraph 状态机——把 while 循环画成图

> 🎬 开场钩子：设想你的 Agent 跑到第 7 步，需要经理批准才能发那封邮件。经理在开会，三小时后才回复。现在问题来了：你的 `while True` 循环怎么办？进程挂着干等三小时？杀掉进程，三小时后从哪一步继续？审批记录存在哪个变量里？——这三个问题，裸写循环的人最终都会撞上，而它们共同指向同一个答案：你需要的不是更聪明的循环，是**状态机**。

⏱ 建议学时：3 小时

## 本课目标

- 说出 `while` 循环式 Agent 的三个结构性缺陷，并解释状态机如何逐一解决
- 用三要素（State / Node / Edge）写出最小 LangGraph 图，解释 reducer 的作用
- 用条件边实现"agent ↔ tools"循环与轮数上限强制收敛
- 解释 human-in-the-loop 的实现原理：checkpointer（状态存档）+ 中断恢复
- 完成：读懂 `code/L03_langgraph_demo.py` 并跑通（默认 mock 模式，不需要 API key）

## 一、为什么是图：循环的三宗罪

阶段 1 你写的 Agent 骨架大致是：

```python
# 阶段 1 的裸写骨架（示意，非完整程序）
messages = [{"role": "user", "content": question}]
while True:
    resp = call_llm(messages)
    if not resp.tool_calls:
        break
    for tc in resp.tool_calls:
        messages.append(execute(tc))
```

三个结构性缺陷：

1. **状态是隐式的**。对话历史在局部变量 `messages` 里，轮数在循环计数里，"下一步做什么"藏在代码控制流里。进程一死，全部蒸发。
2. **控制流是命令式的**。想加"高危操作需人工审批"这种分支，你得改循环体、加标志位，逻辑越插越乱。
3. **没有原生的暂停点**。`while` 要么在跑要么死了，没有"跑到第 N 步、存档、等人、恢复"的概念。

LangGraph 的答案：把 Agent 显式建模为**图**——状态（State）是全局唯一数据，节点（Node）是纯函数（进状态、出增量），边（Edge）决定走向，条件边（conditional edge）在运行时决定分支。三宗罪分别对应：状态显式化、控制流声明化、节点边界即天然存档点。

> ☕ Java 视角：你可能用过 Spring StateMachine 或工作流引擎（Activiti/Camunda 那类 BPMN 引擎）。LangGraph 就是 LLM 时代的轻量工作流引擎：State 对应流程变量，Node 对应服务任务，条件边对应排他网关（exclusive gateway）。区别是它的"任务执行器"是大模型，且为"循环到模型满意为止"这种 BPMN 里很别扭的模式做了原生设计。

## 二、三要素与最小图

### 2.1 State：一张共享的表

```python
# 依赖 langgraph（uv pip install langgraph），版本以官方文档为准
import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict, total=False):
    # reducer：节点返回该字段的增量时，用 operator.add 合并而非整体覆盖
    messages: Annotated[list, operator.add]
    rounds: int          # 无 reducer：节点返回什么就是什么（整体覆盖）
    next_action: str
```

**reducer 是第一个必须理解的概念**：`messages` 带 `operator.add`，意味着每个节点只需返回"新增的消息"，框架自动拼进全量历史；而 `rounds` 没有标注 reducer，节点返回 `{"rounds": 3}` 就是直接覆盖成 3。这决定了"哪些字段是累加型（历史、日志），哪些是快照型（计数、路由决定）"。

### 2.2 Node：纯函数，进状态出增量

```python
def agent_node(state: AgentState) -> dict:
    """决策节点：看历史，决定调工具还是收尾。返回增量，不修改入参。"""
    rounds = state.get("rounds", 0) + 1
    # ... 调用模型或 mock 决策逻辑（完整版见 code/L03_langgraph_demo.py）
    return {"rounds": rounds, "next_action": "tools"}  # 增量字典
```

节点必须是**可重入的纯函数**（同样的输入状态产生同样的输出增量）——因为带 checkpointer 的图可能在中断后重放这个节点。这是从"随手改全局变量"到"工作流引擎思维"最大的转变。

### 2.3 Edge 与条件边

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")            # 入口
builder.add_edge("tools", "agent")          # 固定边：工具执行完回到决策
builder.add_conditional_edges(              # 条件边：运行时选路
    "agent",                                # 从哪个节点出发
    route_after_agent,                      # 路由函数：读状态返回下一站名
    {"tools": "tools", "review": "review", END: END},  # 返回值 → 目标节点映射
)
graph = builder.compile()
```

注意条件边实现循环的方式：`tools → agent` 的固定边提供回边，`agent` 的条件边决定"继续调工具还是结束"。**while 循环消失了，变成了图上肉眼可见的一个环。** 轮数上限也可以放进路由函数：`rounds >= MAX` 时强制走 `END`——防御性终止从"try/except 里数数"变成路由规则。

> 🍿 彩蛋｜ReAct 与那张著名的图：LangGraph 官方教程反复出现的"agent ↔ tools 循环"图，形状正是 ReAct 论文（2022 年，Princeton，Yao 等，arXiv 2210.03629）提出的 Reason+Act 交替结构——模型想一步（reason）、做一步（act）、看一眼结果再想。你阶段 1 裸写的循环，不知不觉复现了这篇论文的核心想法；现在你只是给它换了张更体面的皮。

## 三、human-in-the-loop：中断即存档

回看开场钩子的三小时审批。LangGraph 的解法分两层：

1. **checkpointer（存档器）**：每执行完一个节点，把当前 State 存进检查点库（内存版、SQLite 版等，以官方文档为准）。图从任意检查点恢复。
2. **中断（interrupt）**：把某个节点声明为中断点（或提供中断配置），图运行到那里就暂停并返回；审批意见回来后，从检查点恢复执行，审批结果作为输入注入。

于是"等经理三小时"变成了：跑到 `review` 节点 → 暂停并存档 → 进程该干嘛干嘛（甚至重启）→ 审批事件到达 → 从检查点恢复 → 继续。**等待不再占用你的进程**，这和"HTTP 请求挂在 Tomcat 线程上等回调"到"消息队列 + 状态持久化"的架构演进是同一个思想。

> ☕ Java 视角：对照你熟悉的两种等待：同步阻塞（线程挂着等）vs 事件驱动（状态落库、事件唤醒）。LangGraph 的 checkpointer+interrupt 就是后者，检查点 ≈ 流程引擎的流程实例快照表。

`code/L03_langgraph_demo.py` 里的 `human_review` 节点用**控制台 `input()` 模拟审批**（这样无需数据库就能跑通"高危工具先过审"的形状）；真实项目的跨进程中断恢复要配 checkpointer 与 interrupt API，接口细节以官方文档（python.langchain.com 的 LangGraph 章节）为准。

## 四、裸写 vs 图：一张对照表

| 维度 | while 循环（阶段 1） | LangGraph 图（本课） |
|---|---|---|
| 状态 | 隐式（局部变量） | 显式（TypedDict，可序列化存档） |
| 分支/审批 | 改循环体，标志位越积越多 | 条件边 + 中断节点，加在图上 |
| 循环上限 | 手动计数 + break | 路由函数规则 |
| 暂停恢复 | 无原生支持 | checkpointer 检查点 |
| 可视化 | 无 | 图结构可导出查看（以官方文档为准） |
| 上手成本 | 你已经会了 | 一个新心智模型 + 一个依赖 |

结论先行的判断：**流程一旦包含"循环 + 分支 + 人"，图编排的收益开始大于学习成本**；固定线性 pipeline 则未必值得。

## 五、新手常见的四个坑

1. **在节点里写 `while`**：图上的循环靠边表达，节点里再写循环等于双保险互相打架——轮数控制一律交给路由函数
2. **reducer 用错**：`messages` 忘了加 `operator.add`，节点返回的消息会**整体替换**历史而不是追加，表现为"模型失忆"
3. **节点里做不可逆副作用**：发了邮件才中断等审批——节点边界要想清楚"这一步做完再存档，回放安全吗"（对照阶段 4 L03 的高危分级）
4. **状态里塞大对象**：State 会随 checkpointer 持久化，塞进整个向量库引用会让存档膨胀——放 ID，取数在节点内做

> 🍿 彩蛋｜checkpoint 的黑话血统：这个词来自游戏存档——玩家随手存一盘，死了从存档点复活。分布式系统借走它之后含义变成"系统状态的持久化快照"（Flink、Kafka Streams 里都有），LangGraph 延续了这个用法。你会在阶段 4 L01 看到，它同时也是调试 Agent 的救命稻草：从检查点回放，比翻日志猜强十倍。

## 练习

1. 跑通 `code/L03_langgraph_demo.py`（mock 模式），然后画出实际执行路径（哪些节点、走了几轮）。
   提示：在 demo 里打开或添加结构化打印（讲义阶段 4 L01 的做法），观察 `messages` 的增量合并。
2. 给 demo 加第二个工具（如 `calculator`），让 mock 模型在用户消息包含数字时改调它。
   提示：改动集中在 mock 决策函数与 `tools_node` 的分发逻辑，图结构不用动——体会这一点。
3. 把轮数上限从 5 改成 2，构造一个会触发强制收敛的输入，观察收尾消息。
   提示：看看路由函数和 agent 节点谁先检查 `rounds`。
4. 思考题：如果要把 demo 的控制台审批换成"HTTP 回调 + 8 小时后恢复"，除了 checkpointer 还需要什么？
   提示：想想谁负责"存"与"唤醒"，对照你做过的异步订单状态机。

## 小结与自测

- Q1：State 里 `Annotated[list, operator.add]` 与普通字段的区别？
  一句话：带 reducer 的字段节点返回增量会自动合并，普通字段被整体覆盖。
- Q2：条件边如何形成循环？
  一句话：固定边 `tools → agent` 提供回边，agent 的条件边运行时决定继续（tools）或终止（END）。
- Q3：human-in-the-loop 的两个机制是什么？
  一句话：checkpointer 负责每个节点后的状态存档，interrupt 负责在指定点暂停并支持恢复。
- Q4：什么时候值得从裸写迁到图？
  一句话：流程包含循环+分支+人工介入时收益超过学习成本；线性流程未必值得。
- Q5：为什么节点要写成纯函数？
  一句话：带检查点的图可能在中断恢复时重放节点，纯函数保证重放结果一致。

## 下一课预告

第 4 课跳出"自己写工具"的世界：MCP 协议——为什么全行业需要一个统一的工具协议，以及你的 Agent 如何像插 USB 设备一样接入现成工具生态。
