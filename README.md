# AI 学习工作区

个人 AI 学习仓库，包含两条并行轨道，全部讲义、示例代码与进度记录都在这里。

> 起点：Java Web 基础扎实 / AI Agent 使用熟练 / 数学扎实 / Python 零基础
> 终点：独立完成大模型微调全流程（数据构建 → LoRA/QLoRA 训练 → 评估 → 部署），并能以开发者视角使用与构建 Agent

## 两条轨道

| 轨道 | 周期 | 目录 | 目标 |
|------|------|------|------|
| 主线 | 约 15 周 | `00_python/` ~ `06_frontier/` | Python → ML → PyTorch → Transformer → LLM 生态 → 微调全流程 |
| Agent 开发 | 约 8-10 周（可并行） | `agent_dev/` | API → 手写 function calling → RAG → 框架/MCP → 工程化 + 毕业项目 |

## 目录导航

- `ROADMAP.md` / `PROGRESS.md` — 主线路线图与学习进度
- `agent_dev/ROADMAP.md` / `agent_dev/PROGRESS.md` — Agent 开发轨道路线与进度
- `00_python/` ~ `05_finetuning/` — 主线各阶段课程：`L*.md` 讲义（主要学习资料）、`README.md` 阶段大纲、`code/` 可运行示例
- `agent_dev/00_agent_basics/` ~ `agent_dev/04_engineering/` — Agent 开发各阶段，结构同主线
- `06_frontier/` — 前沿追踪指南
- `STORIES.md` / `GLOSSARY.md` — 下饭故事集与术语速查（`agent_dev/` 内有 Agent 专用版）
- `notes/` — 学习笔记区

## 快速开始

```bash
# Python 3.10+；推荐用 uv 管理环境
uv venv && source .venv/bin/activate
uv pip install <课程所需依赖>
```

示例代码均为课程配套演示，LLM 相关示例通过环境变量读取密钥：

```bash
export LLM_API_KEY=你的key      # 必填
export LLM_BASE_URL=兼容端点地址  # 可选
export LLM_MODEL=gpt-4o-mini     # 可选
```

详细协作方式见 `HOW_TO_LEARN.md`（`agent_dev/` 内有对应版）。
