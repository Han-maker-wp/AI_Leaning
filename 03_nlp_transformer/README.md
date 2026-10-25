# 阶段 3：NLP + Transformer

> 前置：阶段 2（PyTorch 训练循环八股文、反向传播理解）
> 环境：Linux + Python 3.10+；L06/L07 需联网下载模型（国内建议配 HF 镜像，见 L06）；CPU 可完成全部课程（微调课约 1~5 分钟，下载模型约 400MB）
> 预计总学时：约 18 小时（8 课 + Checkpoint 实战）

## 阶段目标

1. 吃透 Attention 与 Transformer：能白板推导注意力公式，解释每个组件（√d_k、掩码、多头、位置编码、残差/LayerNorm）存在的理由；
2. 理解两大预训练范式（BERT 与 GPT）的目标函数差异，以及目标如何决定能力边界；
3. 入门 HuggingFace 生态：tokenizer、模型、Hub 的标准用法与国内镜像配置；
4. 完成人生第一次微调：中文 BERT 文本分类器（端到端脚本，可复用到任何分类任务）。

与最终目标的关系：微调 LLM = 本阶段的微调流程 + 更大的模型 + 更省显存的微调方法（LoRA / 量化）。训练循环完全复用阶段 2 的八股文——本阶段结束时你会亲手验证这一点。

## 课程大纲

| 课次 | 主题 | 核心内容 | 学时 |
|---|---|---|---|
| L01 | 分词与词向量 | 分词难点、词袋→TF-IDF→嵌入演进、word2vec、nn.Embedding | 2h |
| L02 | RNN 与 LSTM 概览 | 隐藏状态、梯度消失连乘直觉、门控、为何被取代 | 1.5h |
| L03 | Attention 精讲 | **本阶段核心课**：Q/K/V、公式拆解、掩码、多头、O(n²)、20 行实现 | 3h |
| L04 | Transformer 论文精读 | arXiv 1706.03762 逐段精读、架构、位置编码、残差+LN | 3h |
| L05 | BERT 与 GPT 预训练范式 | MLM vs 自回归、特殊 token、目标决定能力、GPT 为何胜出 | 2h |
| L06 | HuggingFace 实战入门 | pipeline、AutoTokenizer、AutoModel、镜像配置、Hub 用法 | 2h |
| L07 | BERT 文本分类微调 | **第一次微调**：完整训练脚本、评估、保存与推理 | 3h |
| L08 | 字符级语言模型实战 | bigram → makemore 式神经 LM → 采样生成，看清 GPT 雏形 | 2.5h |

配套可运行代码在 `code/` 目录：`L03_attention_demo.py`、`L06_hf_pipeline_demo.py`、`L07_bert_finetune.py`、`L08_char_lm.py`。

## Checkpoint（阶段通关检验）

1. **白板推导**：不看资料写出注意力公式，并解释 √d_k、softmax、因果掩码各自的作用；
2. **完成微调**：跑通 L07 的中文情感分类微调，评估结果显著优于随机基线（50%），并能指出 BertForSequenceClassification 里哪些参数是预训练加载的、哪些是新初始化的；
3. **换数据复用**：把 L07 脚本接到你自己找的任一二分类数据集（Hub 或本地 CSV，几百条起步）跑通，体会「微调脚本 = 模板 + 数据」；
4. **口头复述**：向空气讲解「为什么 BERT 不能直接做生成、GPT 可以」。

## 资源清单

- HuggingFace NLP Course（huggingface.co/learn）：L06/L07 的官方配套，强烈建议完成第 1~3 章。
- Andrej Karpathy "Neural Networks: Zero to Hero" 中的 makemore 系列（YouTube，有搬运）：L08 配套；进阶看 GitHub 上的 nanoGPT。
- 论文：Attention Is All You Need（arXiv 1706.03762）、BERT（arXiv 1810.04805）、GPT-3（arXiv 2005.14165）。
- 《动手学深度学习》（zh.d2l.ai）：注意力机制等章节用于查漏补缺。

## 学习建议

- L03、L04 是本阶段的硬核两课，允许慢，但公式必须亲手推一遍、demo 必须亲手跑一遍；
- L07 的训练脚本建议放进 git——它就是你毕业课题微调脚本的直系祖先；
- 每课的「用自己的话复述」比做笔记有效：能讲清楚才算懂。
