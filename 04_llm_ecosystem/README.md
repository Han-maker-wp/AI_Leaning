# 阶段 4：LLM 生态与推理实战

> 建议总学时：15~18 小时（7 课 + Checkpoint 实操）
> 前置：Python / PyTorch / Transformer 课程已完结（能独立写 MNIST 训练循环、用 HF 微调过 BERT 分类器）

## 这个阶段解决什么问题

前置课程里你微调的对象是 BERT——一个「判别式」编码器。从这个阶段开始，我们把主角换成
decoder-only 生成式大模型（GPT 架构家族）。你会先搞懂三件事，再动手：

1. **懂原理**：一个开源 LLM 从预训练到「会聊天」经历了什么；推理时 token 是怎么一个个吐出来的。
2. **会跑推理**：transformers / Ollama / vLLM 三条路线都跑通，知道什么时候用哪条。
3. **会算账**：给定模型和精度，口算出需要多少显存——这是后面所有工程决策的地基。
4. **定方案**：本地 GPU 还是云 GPU？本阶段最后一课给出决策框架。

这个阶段**不训练任何模型**，但每一课都在为阶段 5（微调实战）铺路：你算不清显存就没法定算力，
不懂采样参数就没法评估微调效果，不懂 chat template 就会踩微调最经典的坑。

## 阶段目标（学完应能）

- [ ] 画出 LLM 三阶段（预训练 → SFT → RLHF/DPO）流程图并解释每阶段的监督信号
- [ ] 在自己的机器上跑通至少一条 Qwen2.5 推理路线（CPU 或 GPU 均可）
- [ ] 口算「7B 模型在 int4 / fp16 / QLoRA 微调下各需要多少显存」
- [ ] 用 Java 写一个调用本地 LLM 的小程序（发挥你的 Java Web 优势）
- [ ] 产出一份属于自己的算力决策清单（本地 vs 云）

## 7 课大纲

| 课次 | 主题 | 关键词 | 动手产出 |
|------|------|--------|----------|
| L01 | LLM 三阶段全景 | 预训练 / SFT / RLHF / InstructGPT / base vs chat | 流程图笔记 |
| L02 | 开源模型生态与 HF Hub | Llama / Qwen / DeepSeek / 模型卡 / 国内镜像 | 下载一个模型到本地 |
| L03 | 推理原理与采样参数 | 自回归 / KV Cache / temperature / top-p / chat template | 跑 `code/L03_generate_params.py` |
| L04 | 量化与显存估算 | fp16 / bf16 / int4 / GGUF / 显存公式 | 完成口算练习 |
| L05 | 本地推理实战 | transformers / Ollama / vLLM / Java 集成 | 三条路线各跑一次 |
| L06 | 算力决策课 | 显存优先 / AutoDL / 成本测算 / nvidia-smi | 个人算力决策清单 |
| L07 | Function Calling 与 RAG 概览 | 结构化输出 / chunking / embedding / RAG vs 微调选型表 | 选型判断练习 |

## Checkpoint（阶段验收）

完成下面两项才算通过本阶段：

**任务 1：跑通 Qwen 推理**
- 在本地（CPU 或 GPU）用任意路线跑通 Qwen2.5 系列（推荐 `Qwen/Qwen2.5-0.5B-Instruct` 或
  `qwen2.5:0.5b`）的多轮对话，能解释你设置的每个采样参数的含义。
- 参考 L03 / L05 的实操步骤。

**任务 2：搭一个小应用**
- 用 Java（HttpClient / Spring Boot 均可）调用 Ollama 或 vLLM 的 HTTP API，实现一个命令行
  或 Web 小工具，功能不限（如：中文摘要器、代码注释生成器）。
- 参考素材：`code/L05_ollama_java_client.md`、`code/L05_vllm_quickstart.md`。

**验收自问**：
1. 我的程序里 temperature 设了多少？为什么？
2. 如果换成 7B 模型，我的机器/方案还跑得动吗？显存够吗？（用 L04 的公式回答）

## 学习建议

- 命令与参数以官方文档为准（版本迭代很快），讲义中命令均为「典型示例」。
- L04 的显存估算是本阶段最重要的交付，务必亲手算几遍，不要只看表。
- 如果你现在没有任何 GPU：完全没问题。L01~L04 纯理论 + 小模型 CPU 实验，L06 会帮你决策。
