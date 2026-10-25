"""
L03 演示：用 LangGraph 把阶段 1 的 while 循环重构成显式状态机。

依赖（版本以官方文档为准，版本可能更新）：
    uv pip install langgraph
    文档：python.langchain.com 的 LangGraph 章节

运行：
    python L03_langgraph_demo.py

说明：
    - 默认 MOCK 模式：不调用真实 LLM、不需要任何 API key，
      用规则函数模拟"模型决策"，便于本地跑通图结构本身：
      条件边、agent<->tools 循环、轮数上限、高危工具人工审批。
    - human_review 节点用控制台 input() 模拟审批；
      真实项目的跨进程中断恢复请使用 checkpointer + interrupt（以官方文档为准）。
    - 想接真实模型：把 agent_node 里的 mock 决策逻辑换成 ChatModel 调用即可，
      图结构完全不用动——这正是"编排与实现分离"的意义。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

MOCK = True                     # 保持 True 即可跑通演示
MAX_ROUNDS = 5                  # 循环上限：防止失控循环
DANGEROUS_TOOLS = {"send_email"}  # 高危工具白名单：必须经人工审批


# ---------- 1. State：全局唯一的状态表 ----------

class AgentState(TypedDict, total=False):
    """messages 带 reducer：节点只返回增量，框架自动合并；
    其余字段是快照型：节点返回什么就覆盖成什么。"""

    messages: Annotated[list[dict], operator.add]
    rounds: int        # agent 节点被访问的次数（快照型）
    next_action: str   # agent 节点做出的路由决定（快照型）
    approved: bool     # 人工审批结果（快照型）


# ---------- 2. 工具层（mock 实现） ----------

def run_tool(name: str, args: dict) -> str:
    """真实项目里这里替换成你的工具函数（查库、调 API 等）。"""
    if name == "search":
        return f"（mock 搜索结果）关于「{args.get('query', '')}」的 3 条摘要……"
    if name == "send_email":
        return f"（mock）邮件已发送给 {args.get('to', '')}"
    return f"未知工具：{name}"


# ---------- 3. 节点：纯函数，进状态、出增量 ----------

def agent_node(state: AgentState) -> dict:
    """决策节点：看历史决定「调工具 / 去审批 / 收尾」。mock 用规则代替模型。"""
    rounds = state.get("rounds", 0) + 1
    messages = state.get("messages", [])
    user_text = next(
        (m["content"] for m in messages if m.get("role") == "user"), ""
    )

    # 防御性终止：超过轮数上限强制收敛（对应裸写版里的手动计数 + break）
    if rounds > MAX_ROUNDS:
        return {
            "rounds": rounds,
            "next_action": "end",
            "messages": [{"role": "assistant",
                          "content": "已达轮数上限，强制收敛：请换个更具体的问题。"}],
        }

    last = messages[-1] if messages else {}
    if last.get("role") == "tool":
        # 刚拿到工具结果 → 收尾作答（真实项目：把全量 messages 发给模型生成）
        answer = f"根据工具结果（{last.get('name', '')}）：{last['content']}"
        return {"rounds": rounds, "next_action": "end",
                "messages": [{"role": "assistant", "content": answer}]}

    # mock 决策规则：提到"邮件"走高危工具，否则走普通搜索
    if "邮件" in user_text:
        tool_call = {"name": "send_email", "args": {"to": "boss@example.com"}}
        next_action = "review"          # 高危 → 先过人工审批
    else:
        tool_call = {"name": "search", "args": {"query": user_text[:20]}}
        next_action = "tools"
    print(f"[agent] 第 {rounds} 轮决策：调工具 {tool_call['name']} -> {next_action}")
    return {"rounds": rounds, "next_action": next_action,
            "messages": [{"role": "assistant", "content": "",
                          "tool_call": tool_call}]}


def tools_node(state: AgentState) -> dict:
    """执行节点：取出最后一条 tool_call 并执行，结果以 tool 消息回填。"""
    tc = state["messages"][-1]["tool_call"]
    print(f"[tools] 执行 {tc['name']}({tc['args']})")
    return {"messages": [{"role": "tool", "name": tc["name"],
                          "content": run_tool(tc["name"], tc["args"])}]}


def human_review_node(state: AgentState) -> dict:
    """审批节点：高危工具的人工确认点（演示用控制台 input 代替中断恢复）。
    真实项目：声明为中断点 + checkpointer 存档，审批事件到达后从检查点恢复。"""
    tc = state["messages"][-1]["tool_call"]
    print(f"[review] ⚠️ 高危工具需要人工确认：{tc['name']}({tc['args']})")
    approved = input("允许执行吗？(y/n) > ").strip().lower() == "y"
    if not approved:
        return {"approved": False, "next_action": "agent",
                "messages": [{"role": "tool", "name": tc["name"],
                              "content": "用户拒绝了本次执行"}]}
    return {"approved": True, "next_action": "tools"}


# ---------- 4. 路由函数：条件边的运行时决策 ----------

def route_after_agent(state: AgentState) -> str:
    """读 agent 节点写进状态的 next_action，决定下一站。"""
    return state.get("next_action", "end")


def route_after_review(state: AgentState) -> str:
    """审批通过 → 去执行；拒绝 → 回到 agent 重新决策。"""
    return "tools" if state.get("approved") else "agent"


# ---------- 5. 组图：while 循环在这里变成看得见的环 ----------

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("review", human_review_node)

    builder.add_edge(START, "agent")   # 入口
    builder.add_edge("tools", "agent") # 回边：工具执行完回到决策节点（循环的"环"）

    builder.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "review": "review", "end": END},
    )
    builder.add_conditional_edges(
        "review", route_after_review,
        {"tools": "tools", "agent": "agent"},
    )
    return builder.compile()


def main() -> None:
    graph = build_graph()
    # 第 1 个问题走 tools 循环；第 2 个触发人工审批（终端输入 y 或 n 试试两种走向）
    questions = [
        "LangGraph 和 LangChain 有什么区别？",
        "给老板发封邮件说我下周休假",
    ]
    for question in questions:
        print("=" * 60)
        print("用户：", question)
        result = graph.invoke({"messages": [{"role": "user", "content": question}]})
        final = result["messages"][-1]
        print(f"Agent（共 {result['rounds']} 轮）：{final.get('content')}")


if __name__ == "__main__":
    main()
