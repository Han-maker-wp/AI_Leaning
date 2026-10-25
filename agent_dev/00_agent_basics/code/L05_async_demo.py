"""L05_async_demo.py —— 批量并发调用 + 限流 + 指数退避重试（阶段 0 · 第 5 课配套代码）

演示：asyncio.gather 并发 / Semaphore 限并发 / 429 模拟与指数退避（带抖动）/ 可重试分类。

运行模式：
  - 无 LLM_API_KEY：模拟模式（asyncio.sleep 模拟 IO，随机模拟 429），离线可跑
  - 有 LLM_API_KEY（可选 LLM_BASE_URL / LLM_MODEL）：真实批量调用
依赖安装：uv add openai
可调参数：CONCURRENCY（并发度）、N_TASKS（任务数）、FAIL_RATE（模拟 429 概率）
"""

import asyncio
import os
import random
import sys
import time

CONCURRENCY = 3      # 信号量大小：并发天花板（练习里改成 1 / 10 观察耗时）
N_TASKS = 10         # 批量任务数
FAIL_RATE = 0.3      # 模拟模式下每次请求触发 429 的概率
LATENCY = 0.5        # 模拟单次调用耗时（秒）


class RateLimited(Exception):
    """模拟的 429 限流异常（可重试）。"""


class BadRequest(Exception):
    """模拟的 400 参数错误（不可重试，重试只会白白烧配额）。"""


async def mock_llm_call(task_id: int) -> str:
    """离线模拟的一次调用：随机延迟 + 随机 429。"""
    await asyncio.sleep(LATENCY * (0.5 + random.random()))
    if random.random() < FAIL_RATE:
        raise RateLimited(f"task {task_id} 被限流")  # 429: 排队，稍后再来
    return f"task {task_id} 的结果"


async def real_llm_call(client, model: str, task_id: int) -> str:
    """真实模式调用（AsyncOpenAI）。429 会以 SDK 异常形式抛出，此处统一演示框架。"""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"用一句话给任务 {task_id} 起个代号"}],
    )
    return resp.choices[0].message.content or ""


async def call_with_retry(coro_factory, max_retries: int = 4, base: float = 0.3):
    """带指数退避+抖动的重试包装。coro_factory 是"无参且返回协程"的工厂，
    保证每次重试能重新发起请求（协程对象只能 await 一次）。"""
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except BadRequest:
            raise  # 不可重试：确定性失败，立即上抛
        except RateLimited:
            if attempt == max_retries:
                raise  # 额度用尽，如实失败
            delay = base * (2 ** attempt) + random.uniform(0, base)  # 指数退避 + 抖动
            print(f"    收到 429，第 {attempt + 1} 次退避 {delay:.2f}s 后重试")
            await asyncio.sleep(delay)


async def worker(sem: asyncio.Semaphore, call) -> str:
    """拿不到信号量名额就在此等待——并发闸门在 async with 这一行。"""
    async with sem:
        return await call_with_retry(call)


async def main() -> None:
    random.seed(42)  # 固定随机种子，保证离线演示可复现
    api_key = os.environ.get("LLM_API_KEY")
    client = None
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if api_key:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL"))
        print("[模式] 真实 API")
    else:
        print("[模式] 模拟 IO（未检测到 LLM_API_KEY）")
    print(f"任务数={N_TASKS} 并发度={CONCURRENCY}\n")

    sem = asyncio.Semaphore(CONCURRENCY)

    def make_call(i: int):
        # 工厂：每次调用生成新协程——重试时用
        if client:
            return lambda: real_llm_call(client, model, i)
        return lambda: mock_llm_call(i)

    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(worker(sem, make_call(i)) for i in range(1, N_TASKS + 1)),
        return_exceptions=True,  # 单个失败不拖垮整批，最后统一汇报
    )
    elapsed = time.perf_counter() - t0

    ok = [r for r in results if not isinstance(r, Exception)]
    bad = [r for r in results if isinstance(r, Exception)]
    print(f"\n完成: {len(ok)}/{N_TASKS}，失败 {len(bad)}，总耗时 {elapsed:.2f}s")
    print(f"参考：串行预计约 {N_TASKS * LATENCY:.1f}s（并发收益来自等待重叠，不是并行计算）")
    for r in bad:
        print("  失败任务:", type(r).__name__, r)


if __name__ == "__main__":
    asyncio.run(main())
