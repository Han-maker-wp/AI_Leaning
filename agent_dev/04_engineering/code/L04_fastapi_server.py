"""
L04 演示：用 FastAPI 把 Agent 封装成 HTTP 服务（含 SSE 流式端点）。

依赖（版本以官方文档为准，版本可能更新）：
    uv pip install fastapi uvicorn
    文档：fastapi.tiangolo.com

运行：
    python L04_fastapi_server.py        # 启动在 http://127.0.0.1:8000
    然后打开 http://127.0.0.1:8000/docs 体验自动生成的交互文档

说明：
    - 默认 MOCK 模式：agent_stream() 用规则函数生成假回答，无需任何 API key，
      便于先跑通"契约 + 流式 + 健康检查"的服务形状。
    - 接真实模型：把 agent_stream() 里标注 [LLM 接入点] 的部分换成真实调用，
      并设置环境变量 OPENAI_API_KEY（本文件不内置任何厂商 SDK）。
    - 会话记忆用内存 dict 演示：生产环境必须外置（Redis/DB），见讲义第四节。
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MOCK = True  # True = 假回答；False 需自行接入真实模型（见 agent_stream 内注释）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # 接真实模型时必须设置

app = FastAPI(title="agent-service", version="0.1.0")

# ---------- 会话记忆（内存演示版：生产环境换 Redis/DB，按 session_id 存取） ----------

HISTORY: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY = 20  # 每会话保留的最大轮数（防上下文无限膨胀）


# ---------- 请求/响应契约（≈ DTO + Bean Validation） ----------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="用户输入")
    session_id: str | None = Field(default=None, description="会话 ID，多轮对话必带")


class ChatChunk(BaseModel):
    """流式协议的最小单位：新增文本 + 是否结束"""
    delta: str
    done: bool


# ---------- Agent 本体（mock） ----------

async def agent_stream(message: str, session_id: str) -> str:
    """异步生成器逐块产出回答。

    [LLM 接入点] 真实实现：把 message 与 HISTORY[session_id] 拼成上下文，
    调用模型 API 并逐块 yield（各家 SDK 的流式接口以官方文档为准，
    需要环境变量 OPENAI_API_KEY）。等待网络 IO 用 await，不要同步阻塞。
    """
    if MOCK or not OPENAI_API_KEY:
        reply = (f"（mock 回答）已收到：{message!r}。"
                 f"真实模式下这里会调模型并逐字流出。会话 {session_id} 共 "
                 f"{len(HISTORY[session_id])} 轮历史。")
    else:
        reply = "[未实现真实模型调用] 请回到本文件的 [LLM 接入点] 注释处补全。"
    # 模拟流式：把回答切块，每块间隔一点时间（模拟网络到达的节奏）
    for i in range(0, len(reply), 4):
        await asyncio.sleep(0.05)  # 必须用 await 版：阻塞版会饿死事件循环（练习 4）
        yield reply[i:i + 4]


def remember(session_id: str, user_msg: str, reply: str) -> None:
    """写回会话记忆并截断到上限（最简单的滑动窗口策略）"""
    HISTORY[session_id].append({"user": user_msg, "assistant": reply})
    if len(HISTORY[session_id]) > MAX_HISTORY:
        HISTORY[session_id] = HISTORY[session_id][-MAX_HISTORY:]


# ---------- 端点 ----------

@app.get("/health")
async def health() -> dict:
    """健康检查：负载均衡探活用（顺手带上一点运行信息）"""
    return {"status": "ok", "mock": MOCK, "sessions": len(HISTORY)}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """非流式：一次性返回完整回答（适合后端互调、测试）"""
    session_id = req.session_id or "default"
    start = time.perf_counter()
    chunks: list[str] = []
    async for delta in agent_stream(req.message, session_id):
        chunks.append(delta)
    reply = "".join(chunks)
    remember(session_id, req.message, reply)
    return {"reply": reply, "session_id": session_id,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1)}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式：SSE 逐块推送（curl -N 或 Java HttpClient ofLines 消费）。

    注意：浏览器原生 EventSource 只支持 GET；服务端互调用 POST + 流式读
    响应体即可（主流大模型 API 也是这个组合）。
    """
    session_id = req.session_id or "default"
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def produce() -> None:
        """生产者：驱动 agent_stream 并收集全文写回记忆"""
        collected: list[str] = []
        async for delta in agent_stream(req.message, session_id):
            collected.append(delta)
            await queue.put(delta)
        remember(session_id, req.message, "".join(collected))
        await queue.put(None)  # 结束哨兵

    async def sse_events():
        producer = asyncio.create_task(produce())
        while True:
            item = await queue.get()
            if item is None:
                break
            # SSE 格式：每条消息以 "data: " 开头、空行结尾（协议细节以规范为准）
            yield f"data: {ChatChunk(delta=item, done=False).model_dump_json()}\n\n"
        yield f"data: {ChatChunk(delta='', done=True).model_dump_json()}\n\n"
        await producer  # 确保记忆写回完成后再收尾

    return StreamingResponse(sse_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # 开发自跑入口；生产多 worker 配置以 uvicorn 官方文档为准
    uvicorn.run(app, host="127.0.0.1", port=8000)
