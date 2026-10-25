"""L04_structured_output.py —— 结构化输出三层防御（阶段 0 · 第 4 课配套代码）

三层：约定层（prompt 带 schema 与示例）→ 校验层（Pydantic）→ 修正层（错误回传重试）。
容错解析负责"只修格式伤"（剥围栏/截主体/去尾逗号），语义伤一律拒绝并触发重试。

运行模式：
  - 无 LLM_API_KEY：自动进入 mock 模式，用内置的"花式翻车"模型回复演示全流程（离线可跑）
  - 有 LLM_API_KEY（可选 LLM_BASE_URL / LLM_MODEL）：走真实 API
依赖安装：uv add pydantic openai
"""

import json
import os
import random
import re
import sys
from typing import Literal

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:  # 依赖未装时给出可操作的提示，而不是抛裸栈
    sys.exit("缺少 pydantic 依赖：请先运行 uv add pydantic（或 pip install pydantic）")

MAX_RETRIES = 2  # 自我修正重试上限：保证程序必然终止


class Ticket(BaseModel):
    """工单分类的输出契约：字段、类型、约束一次声明。"""
    category: Literal["咨询", "投诉", "合作", "unknown"] = Field(description="工单类别")
    priority: int = Field(ge=1, le=5, description="优先级，1 最低 5 最高")
    summary: str = Field(min_length=5, max_length=60, description="一句话摘要")


SYSTEM_PROMPT = (
    "你是客服工单分类器。只输出一个 JSON 对象，不要 markdown 代码块，不要解释文字。\n"
    'schema: {"category": "咨询|投诉|合作|unknown", "priority": 1-5 的整数, "summary": "5-60字摘要"}\n'
    "信息不足以判断时 category 填 unknown、priority 填 3，不要编造。\n"
    '示例——输入：你们系统又崩了，我要退款！输出：{"category": "投诉", "priority": 4, '
    '"summary": "系统崩溃用户要求退款"}'
)

USER_EMAIL = "您好，我们想咨询一下贵司 API 服务的企业版报价与合作流程，麻烦回复。"

# ---- mock 模式：内置三份"花式翻车"的模型回复，演示容错与修正路径 ----
MOCK_RAW_OUTPUTS = [
    '好的，以下是分类结果：\n```json\n{"category": "咨询", "priority": 4, '
    '"summary": "咨询API企业版报价与合作流程",}\n```',  # 围栏 + 前置废话 + 尾逗号
    '{"category": "咨询", "priority": 9, "summary": "咨询API企业版报价与合作流程"}',  # 语义伤：越界
    '{"category": "咨询", "priority": 4, "summary": "咨询API企业版报价与合作流程"}',  # 修正后的合法输出
]
MOCK_REPAIRED_OUTPUT = MOCK_RAW_OUTPUTS[2]  # 收到错误说明书后，模型通常一轮修好


def tolerant_json_loads(text: str) -> dict:
    """容错解析：只修格式伤（围栏/前后杂质/尾逗号）；修不好抛异常交给重试。"""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("输出中找不到 JSON 对象")
    cleaned = cleaned[start:end + 1]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # 去掉 } ] 前的尾逗号
    return json.loads(cleaned)


def mock_model_reply(retry_error: str | None) -> str:
    """用条件分支模拟模型响应（离线演示用）：首轮顺序吐翻车样例，修正轮给合法输出。"""
    if retry_error is None:
        idx = random.randrange(len(MOCK_RAW_OUTPUTS) - 1)  # 前两份随机翻车
        print(f"    [mock 模型原始输出] {MOCK_RAW_OUTPUTS[idx][:50]}...")
        return MOCK_RAW_OUTPUTS[idx]
    print(f"    [mock 收到修正请求，返回合法输出]")
    return MOCK_REPAIRED_OUTPUT


def real_model_reply(client, model: str, history: list[dict], retry_error: str | None) -> str:
    """真实模式：把校验错误作为 assistant/用户回执拼回 messages，请求模型修正。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)
    if retry_error:
        messages.append({"role": "user", "content": f"你的上一次输出未通过校验：{retry_error}。请只输出修正后的 JSON。"})
    else:
        messages.append({"role": "user", "content": f"输入：{USER_EMAIL}\n输出："})
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content or ""


def parse_ticket(raw: str) -> Ticket:
    """容错解析 + Pydantic 校验；失败时把人类可读的错误说明书抛给上层。"""
    try:
        data = tolerant_json_loads(raw)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"无法解析出 JSON：{e}") from e
    try:
        return Ticket.model_validate(data)
    except ValidationError as e:
        detail = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise ValueError(detail) from e  # 字段级明细 = 回传模型的修复说明书


def main() -> None:
    api_key = os.environ.get("LLM_API_KEY")
    client = None
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if api_key:
        from openai import OpenAI  # 有 key 才导入，离线模式零依赖
        client = OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL"))
        print("[模式] 真实 API")
    else:
        print("[模式] mock（未检测到 LLM_API_KEY，用内置翻车样例演示）")

    history: list[dict] = []
    retry_error: str | None = None
    for attempt in range(1 + MAX_RETRIES):
        if client:
            raw = real_model_reply(client, model, history, retry_error)
            history.append({"role": "assistant", "content": raw})
        else:
            raw = mock_model_reply(retry_error)
        try:
            ticket = parse_ticket(raw)
            print(f"  ✔ 第 {attempt + 1} 次通过校验: {ticket.model_dump()}")
            return
        except ValueError as e:
            retry_error = str(e)
            print(f"  ✘ 第 {attempt + 1} 次被拒: {retry_error}")
    print(f"  ✘ 重试 {MAX_RETRIES} 次仍失败，放弃（上层应记日志并走降级路径）")
    sys.exit(1)


if __name__ == "__main__":
    main()
