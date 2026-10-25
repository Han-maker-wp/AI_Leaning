# L05 配套：vLLM 快速上手（启动 + curl 调用）

> vLLM 版本迭代较快，启动参数与命令形式以 vLLM GitHub 仓库 README / 官方文档为准；
> 本文为「典型示例」。

## 0. 适用前提

- 需要 NVIDIA GPU（vLLM 不面向 CPU 场景）。
- 显存按阶段 4 L04 的账：7B bf16 ≈ 14GB 权重 + KV cache 预留 → 建议 24GB 卡；
  也可直接加载 AWQ/GPTQ 量化版模型降到 8~12GB。

## 1. 安装

```bash
# 建议 Python 3.10+ 虚拟环境中安装；版本以官方文档为准
pip install vllm
```

## 2. 启动 OpenAI 兼容服务

```bash
# 典型启动命令（新版本为 `vllm serve`；旧版本为 python -m vllm.entrypoints.openai.api_server）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192
```

参数说明（均为典型值，完整列表以官方文档为准）：

| 参数 | 含义 | 调整直觉 |
|---|---|---|
| `--gpu-memory-utilization` | 允许使用的显存比例 | 预留 0.05~0.1 给波动；与其他程序共卡时调低 |
| `--max-model-len` | 最大上下文长度 | 直接决定 KV cache 预留（L03/L04），能小则小 |
| `--host/--port` | 监听地址 | `0.0.0.0` 供局域网/容器外访问 |

首次启动会自动从 HF 下载模型；国内网络先 `export HF_ENDPOINT=https://hf-mirror.com`。
启动完成的标志：日志出现 Uvicorn/OpenAI 兼容 server 的监听信息。

## 3. curl 调用示例

### 3.1 查看已加载模型

```bash
curl http://localhost:8000/v1/models
```

### 3.2 Chat Completions（主接口）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个严谨的 Java 编程助手。"},
      {"role": "user", "content": "用一句话解释 Spring 的依赖注入"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 256
  }'
```

响应结构与 OpenAI 完全一致：`choices[0].message.content` 为答案，`usage` 含 token 统计。

### 3.3 流式输出（SSE）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "写一个 Java 快速排序"}],
    "stream": true
  }'
# 响应为 text/event-stream：每行 data: {...} 是一个增量片段，最后以 data: [DONE] 结束
```

### 3.4 用 OpenAI 官方 SDK 直连（换 base_url 即可）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",   # vLLM 默认不校验，任意非空值即可
)
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

## 4. 并发体验（感受 continuous batching）

```bash
# 用一个简单循环同时发 8 个请求，观察吞吐远高于逐个串行（vLLM 的核心价值）
for i in $(seq 1 8); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-7B-Instruct",
         "messages": [{"role": "user", "content": "写 100 字介绍你自己"}],
         "max_tokens": 128}' > resp_$i.json &
done
wait
```

进一步压测可使用专业工具（如开源的 LLM 基准/压测项目，任选其一），关注指标：
吞吐（requests/s、tokens/s）、首 token 延迟（TTFT）、P95 延迟。

## 5. 常见问题速查

| 现象 | 处理 |
|---|---|
| 启动报 CUDA OOM | 降 `--gpu-memory-utilization` 或 `--max-model-len`；或换量化版模型 |
| 下载模型超时 | `export HF_ENDPOINT=https://hf-mirror.com` 后重启服务 |
| 请求 404 model not found | `model` 字段必须与启动时的模型名/`--served-model-name` 完全一致 |
| 想服务自己微调的模型 | 传本地权重目录路径启动（阶段 5 L09 演示） |

## 6. 与 Java 侧的衔接

vLLM 的 OpenAI 兼容接口即阶段 5 `code/java_vllm_client.md` 的对接目标：
Spring Boot 后端通过 HttpClient 调 `/v1/chat/completions`，即可把自托管模型接入你的
Java 技术栈——那是毕业项目的最后一公里。
