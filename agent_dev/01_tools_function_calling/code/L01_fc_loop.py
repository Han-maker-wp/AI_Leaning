"""L01_fc_loop.py —— 手写 function calling 循环（阶段 1 · 第 1 课配套代码）

完整实现"模型请求工具 → 代码执行 → 结果回填 → 再请求 → 终止"的循环。
默认离线运行：用条件分支模拟模型响应（mock），协议数据结构与真实 API 完全同构；
设了 LLM_API_KEY 则切换为真实调用（可选 LLM_BASE_URL / LLM_MODEL）。

协议字段（tools / tool_calls / tool_call_id / role="tool"）以
platform.openai.com/docs 官方文档为准，版本可能更新。
"""

import json
import os
import sys

MAX_ROUNDS = 5  # 循环安全阀：防止模型无限要求调工具

# ---------- 1. tool schema：给模型看的接口文档 ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前天气。仅支持北京、上海；温度单位为摄氏度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如「北京」"},
                },
                "required": ["city"],
            },
        },
    }
]

# ---------- 2. 工具实现（你的代码真正干活的地方） ----------
WEATHER_DB = {"北京": 18, "上海": 24}


def get_weather(city: str) -> dict:
    """真实实现。错误按 L02 的原则返回结构化文本，不抛异常炸循环。"""
    if city not in WEATHER_DB:
        return {"ok": False, "error": "CITY_NOT_FOUND",
                "message": f"不支持的城市: {city}",
                "hint": "仅支持北京、上海，请换城市或向用户澄清"}
    return {"ok": True, "city": city, "temp_c": WEATHER_DB[city]}


TOOL_IMPLS = {"get_weather": get_weather}


# ---------- 3. mock 模型：用条件分支模拟"模型决策" ----------
def mock_model_reply(messages: list[dict]) -> dict:
    """模拟模型响应，返回与真实 API 同构的消息 dict。
    决策依据只有 messages——这正是真模型的工作方式（它看不到别的）。"""
    last = messages[-1]
    if last["role"] == "tool":
        # 看到工具结果 → 生成最终回答（不再包含 tool_calls，循环将终止）
        data = json.loads(last["content"])
        answer = f"{data['city']}当前气温 {data['temp_c']} 摄氏度。" if data.get("ok") \
            else f"暂时查不到（{data.get('message')}），换个城市试试？"
        return {"role": "assistant", "content": answer, "tool_calls": None}
    # 否则（用户刚提问）→ 请求调用工具；arguments 是 JSON 字符串，与真实协议一致
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_001",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "北京"})},
        }],
    }


def real_model_reply(client, model: str, messages: list[dict]) -> dict:
    """真实模式：调用兼容端点，把 SDK 消息对象转成与 mock 同构的 dict。"""
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto")
    msg = resp.choices[0].message
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [{"id": tc.id, "type": "function",
                       "function": {"name": tc.function.name,
                                    "arguments": tc.function.arguments}}
                      for tc in msg.tool_calls]
    return {"role": "assistant", "content": msg.content, "tool_calls": tool_calls}


# ---------- 4. 主循环：本课的主角 ----------
def run_agent(messages: list[dict], reply_fn) -> str:
    for round_no in range(1, MAX_ROUNDS + 1):
        msg = reply_fn(messages)
        print(f"── 第 {round_no} 轮 · 模型返回 "
              f"{'[工具请求]' if msg['tool_calls'] else '[最终回答]'}")

        # ① 回填 assistant 消息（含 tool_calls）——漏掉这步 API 会直接报错
        to_append = {k: v for k, v in msg.items() if v is not None or k != "tool_calls"}
        messages.append(to_append)

        if not msg["tool_calls"]:
            return msg["content"]  # 终止条件：模型不再请求工具

        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])  # ② arguments 是字符串！
            print(f"   模型请求: {name}({args})")
            result = TOOL_IMPLS[name](**args)              # ③ 你的代码真正执行
            print(f"   执行结果: {result}")
            messages.append({                              # ④ 结果以 role=tool 回填，靠 id 配对
                "role": "tool", "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False)})
    return f"已达最大轮数 {MAX_ROUNDS}，强制停止。完整历史已保留在 messages 中。"


def main() -> None:
    messages = [{"role": "user", "content": "北京今天多少度？"}]
    if os.environ.get("LLM_API_KEY"):
        from openai import OpenAI  # 有 key 才导入；离线模式零第三方依赖
        client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                        base_url=os.environ.get("LLM_BASE_URL"))
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        reply_fn = lambda msgs: real_model_reply(client, model, msgs)  # noqa: E731
        print("[模式] 真实 API\n")
    else:
        reply_fn = mock_model_reply
        print("[模式] mock（未检测到 LLM_API_KEY，用条件分支模拟模型）\n")
    answer = run_agent(messages, reply_fn)
    print(f"\n最终回答: {answer}")
    print(f"messages 共 {len(messages)} 条——多轮工具调用的全部状态就是它")
    sys.exit(0)


if __name__ == "__main__":
    main()
