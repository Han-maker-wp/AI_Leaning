"""L03_mini_agent.py —— 多工具迷你 Agent（阶段 1 · 第 3 课配套代码，checkpoint 起点）

不用任何框架的 Agent loop：工具注册表 + 循环 + 终止条件 + 错误文本化 + 轮数上限。
工具：calculate（AST 安全计算器雏形）/ read_file / write_file（workspace jail 简版，完整安全版见 L05）。
默认离线运行：mock 模型按消息历史用条件分支决策，剧本为"读 data.txt → 算乘积 → 写 result.txt"。
设了 LLM_API_KEY 则切换真实调用（可选 LLM_BASE_URL / LLM_MODEL）。

协议字段以 platform.openai.com/docs 官方文档为准，版本可能更新。
"""

import ast
import json
import operator
import os
import sys
from pathlib import Path

MAX_ROUNDS = 5
WORKSPACE = Path(__file__).resolve().parent / "agent_workspace"

# ---------- 工具实现（含错误文本化：错误也是给模型的输出） ----------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_path(relative: str) -> Path:
    """L05 的目录 jail 简版：resolve 后必须仍在 WORKSPACE 内。"""
    p = (WORKSPACE / relative).resolve()
    if p != WORKSPACE and WORKSPACE not in p.parents:
        raise ValueError(f"路径越界: {relative}")
    return p


def calculate(a: float, b: float, op: str) -> str:
    """演示用安全计算器（AST 完整版见 L05 讲义）。"""
    fn = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
          "div": operator.truediv}.get(op)
    if fn is None:
        return json.dumps({"ok": False, "error": "BAD_OP", "message": f"未知运算: {op}",
                           "hint": "op 只支持 add/sub/mul/div"}, ensure_ascii=False)
    return json.dumps({"ok": True, "result": fn(a, b)}, ensure_ascii=False)


def read_file(path: str) -> str:
    try:
        return json.dumps({"ok": True, "content": _safe_path(path).read_text("utf-8")},
                          ensure_ascii=False)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"ok": False, "error": type(e).__name__, "message": str(e)},
                          ensure_ascii=False)


def write_file(path: str, content: str) -> str:
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "written": path}, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"ok": False, "error": "PATH_ESCAPED", "message": str(e)},
                          ensure_ascii=False)


# ---------- 注册表：schema 与实现单一事实源，加工具零改循环 ----------
TOOL_IMPLS = {"calculate": calculate, "read_file": read_file, "write_file": write_file}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculate", "description": "对两个数做算术运算，返回结果。",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "number"}, "b": {"type": "number"},
            "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]}},
            "required": ["a", "b", "op"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "读取工作目录内指定文本文件的内容。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对工作目录的路径"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "把文本写入工作目录内的指定文件。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]


def dispatch(name: str, args: dict) -> str:
    func = TOOL_IMPLS.get(name)
    if func is None:
        return json.dumps({"ok": False, "error": "UNKNOWN_TOOL",
                           "message": f"未注册的工具: {name}"}, ensure_ascii=False)
    return func(**args)  # 工具内部已把异常转成 error 文本，循环不死


# ---------- mock 模型：按消息历史条件分支，剧本=读文件→计算→写文件→收尾 ----------
def _tool_msg(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def mock_model_reply(messages: list[dict]) -> dict:
    """按已回填的 tool 消息数推进剧本。判定依据只有 messages——这正是真模型的工作方式。"""
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    n = len(tool_msgs)
    if n == 0:                                # 第 1 步：读文件
        return {"role": "assistant", "content": None,
                "tool_calls": [_tool_msg("read_file", {"path": "data.txt"}, "call_1")]}
    read_result = json.loads(tool_msgs[0]["content"])
    if not read_result.get("ok"):
        return {"role": "assistant", "content": "data.txt 读取失败，请确认文件存在。",
                "tool_calls": None}
    if n == 1:                                # 第 2 步：从文件内容解析两个数并计算
        a, b = (int(t) for t in read_result["content"].split())
        return {"role": "assistant", "content": None,
                "tool_calls": [_tool_msg("calculate", {"a": a, "b": b, "op": "mul"}, "call_2")]}
    calc_result = json.loads(tool_msgs[1]["content"])
    if not calc_result.get("ok"):
        return {"role": "assistant", "content": "计算失败，请检查参数。", "tool_calls": None}
    value = calc_result["result"]
    if n == 2:                                # 第 3 步：把结果写入文件
        return {"role": "assistant", "content": None,
                "tool_calls": [_tool_msg("write_file",
                                         {"path": "result.txt", "content": str(value)}, "call_3")]}
    # 第 4 步：收尾回答（不再请求工具，循环终止）
    write_ok = json.loads(tool_msgs[2]["content"]).get("ok")
    final = f"已计算并写入 result.txt：{value}" if write_ok else f"计算完成（{value}），但写文件失败"
    return {"role": "assistant", "content": final, "tool_calls": None}


def real_model_reply(client, model: str, messages: list[dict]) -> dict:
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
    msg = resp.choices[0].message
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [{"id": tc.id, "type": "function",
                       "function": {"name": tc.function.name,
                                    "arguments": tc.function.arguments}}
                      for tc in msg.tool_calls]
    return {"role": "assistant", "content": msg.content, "tool_calls": tool_calls}


# ---------- Agent 主循环（与 L01 同骨架，注册表让多工具零成本接入） ----------
def run_agent(messages: list[dict], reply_fn) -> str:
    for round_no in range(1, MAX_ROUNDS + 1):
        msg = reply_fn(messages)
        messages.append({k: v for k, v in msg.items() if v is not None or k != "tool_calls"})
        if not msg["tool_calls"]:
            return msg["content"]
        for tc in msg["tool_calls"]:
            name, args = tc["function"]["name"], json.loads(tc["function"]["arguments"])
            result = dispatch(name, args)
            print(f"  [第{round_no}轮] 执行 {name}({args}) -> {result[:60]}")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return f"已达最大轮数 {MAX_ROUNDS}，强制收敛。"


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "data.txt").write_text("12 34", encoding="utf-8")  # 准备剧本数据
    messages = [{"role": "user", "content": "读 data.txt 里的两个数，算乘积后写入 result.txt"}]
    if os.environ.get("LLM_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                        base_url=os.environ.get("LLM_BASE_URL"))
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        reply_fn = lambda msgs: real_model_reply(client, model, msgs)  # noqa: E731
        print("[模式] 真实 API\n")
    else:
        reply_fn = mock_model_reply
        print("[模式] mock（剧本：读文件 → 计算 → 写文件）\n")
    print("最终回答:", run_agent(messages, reply_fn))
    print("result.txt 内容:", (WORKSPACE / "result.txt").read_text("utf-8"))
    sys.exit(0)


if __name__ == "__main__":
    main()
