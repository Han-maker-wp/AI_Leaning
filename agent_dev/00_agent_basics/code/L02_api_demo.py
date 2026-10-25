"""L02_api_demo.py —— LLM API 实战演示（阶段 0 · 第 2 课配套代码）

功能：非流式调用 / 多轮对话 / 流式输出（打字机 + 首 token 延迟）/ usage 统计。

运行前提（依赖外部 API，用环境变量占位）：
  - LLM_API_KEY   必填：你的 API key（OpenAI 官方或任意兼容端点）
  - LLM_BASE_URL  可选：兼容端点地址（如 https://某平台/v1）；不设则连 OpenAI 官方
  - LLM_MODEL     可选：模型名，默认 gpt-4o-mini
依赖安装：uv add openai
API 参数以 platform.openai.com/docs 官方文档为准，版本可能更新。
"""

import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:  # 依赖未装时给出可操作的提示，而不是抛裸栈（讲义 2.7 的原则）
    OpenAI = None  # type: ignore[assignment]

SYSTEM_PROMPT = "你是一个简洁的中文技术助手，回答不超过两句话。"


def build_client():
    """从环境变量构建客户端：key 必须有，base_url 可选（兼容端点）。"""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        sys.exit("未检测到 LLM_API_KEY。请先: export LLM_API_KEY=你的key "
                 "（可选: export LLM_BASE_URL=兼容端点地址  LLM_MODEL=模型名）")
    if OpenAI is None:
        sys.exit("缺少 openai 依赖：请先运行 uv add openai（或 pip install openai）")
    return OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL"))


def demo_single(client: OpenAI, model: str) -> None:
    """演示一：单次调用——content / finish_reason / usage 三件套。"""
    print("=" * 50, "\n[演示一] 非流式单次调用")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "用一句话解释什么是 REST"},
        ],
    )
    msg = resp.choices[0].message
    print("回答:", msg.content)
    print("finish_reason:", resp.choices[0].finish_reason)
    u = resp.usage  # 精确账单永远以 usage 为准，不要自己估算
    print(f"token: 输入={u.prompt_tokens} 输出={u.completion_tokens} 总计={u.total_tokens}")


def demo_multi_turn(client: OpenAI, model: str) -> None:
    """演示二：多轮对话——API 无状态，历史靠客户端全量重发。"""
    print("=" * 50, "\n[演示二] 多轮对话（观察 messages 列表如何生长）")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    questions = ["什么是 SSE？", "它和 WebSocket 有什么区别？"]  # 第二问依赖第一问上下文
    for q in questions:
        messages.append({"role": "user", "content": q})
        resp = client.chat.completions.create(model=model, messages=messages)
        answer = resp.choices[0].message.content
        print(f"问: {q}\n答: {answer}\n")
        # 关键一步：把 assistant 历史也追加进去，否则下一轮模型不记得自己说过什么
        messages.append({"role": "assistant", "content": answer})
    print(f"当前 messages 共 {len(messages)} 条——第 2 轮实际重发了全部历史")


def demo_stream(client: OpenAI, model: str) -> None:
    """演示三：流式输出——打字机效果 + 首 token 延迟（TTFT）测量。"""
    print("=" * 50, "\n[演示三] 流式输出")
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用三句话介绍 token 与计费的关系"}],
        stream=True,  # 流式开关：生成一个 token 推一个 chunk
    )
    first = True
    pieces = []
    for chunk in stream:
        piece = chunk.choices[0].delta.content  # 增量片段，可能为 None
        if not piece:
            continue
        if first:
            print(f"（首 token 延迟 {time.perf_counter() - t0:.2f}s）")
            first = False
        pieces.append(piece)
        print(piece, end="", flush=True)  # 打字机效果
    print(f"\n（总耗时 {time.perf_counter() - t0:.2f}s）")


def main() -> None:
    client = build_client()
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    try:
        demo_single(client, model)
        demo_multi_turn(client, model)
        demo_stream(client, model)
    except Exception as e:  # 网络/鉴权/限流等异常的友好出口，不裸抛栈
        print(f"\n调用失败: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
