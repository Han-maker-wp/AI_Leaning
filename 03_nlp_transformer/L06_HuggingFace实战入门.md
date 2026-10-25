# 第 6 课：HuggingFace 实战入门

> 🎬 开场钩子：Hugging Face 的 logo 是一个拥抱的笑脸 🤗，名字来自一个滑板术语。这家公司最初做的是聊天机器人产品，后来转型做开源模型平台，意外长成了「AI 界的 GitHub」——今天海量模型与数据集都托管在这里。这节课学习怎么在其中取用模型，它是你后续微调与部署的主战场。

⏱ 建议学时：2 小时

## 本课目标

1. 能画出 HuggingFace 生态地图（transformers / datasets / tokenizers / Hub / accelerate）；
2. 用 pipeline 三行代码完成一次推理；
3. 掌握 AutoTokenizer：encode / decode、input_ids 与 attention_mask；
4. 掌握 AutoModel：说清输出 [batch, seq, hidden] 每一维的含义；
5. 配置国内镜像加速下载，学会在 Hub 上找模型与读模型卡。

## 6.1 生态地图

HuggingFace 是「深度学习的 GitHub + Maven 中央仓库」——对你这个 Java 背景的学员，这个类比几乎不用解释：

| 组件 | 作用 | Java 世界对应 |
|---|---|---|
| transformers | 模型 / 分词器库（PyTorch 之上的高层 API） | 框架层（如 Spring） |
| datasets | 数据集加载与处理工具 | —— |
| tokenizers | 高性能分词（Rust 实现） | —— |
| Hub | 百万级模型 / 数据集仓库 | Maven Central + GitHub |
| accelerate | 多卡 / 混合精度训练适配 | —— |

学习资源：huggingface.co/learn 的 NLP Course（免费，有中文翻译），第 1~3 章与本课和下一课内容重合，强烈建议配套完成。

> 🍿 彩蛋｜性能件都是 Rust 芯：表里 tokenizers 那格的「Rust 实现」不是孤例——如今 Python 生态的性能关键件流行 Rust 内核：数据处理有 polars，代码工具有 ruff，分词有 tokenizers。Python 做胶水、Rust 做发动机，Java 老兵看着应该眼熟：和当年「C++ 内核 + Java 框架」的 JVM 生态是同一种分层哲学。

## 6.2 安装与国内镜像

```bash
uv pip install transformers datasets
# transformers 依赖 torch，确认已装好阶段 2 的 CPU 版
```

模型权重托管在 huggingface.co，国内直连不稳，配置镜像（写进 ~/.bashrc）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

镜像站由社区维护（hf-mirror.com，站内有详细使用说明）。下载的模型缓存在 `~/.cache/huggingface/`，同一模型只下载一次——第一次慢，之后离线可用。

## 6.3 pipeline：三行代码推理

```python
from transformers import pipeline

clf = pipeline("text-classification")            # 默认加载英文情感模型（首次运行会下载）
print(clf("I love this course!"))
# [{'label': 'POSITIVE', 'score': 0.9998...}]
```

pipeline 是「全家桶」封装：自动完成 下载模型 → tokenize → 前向 → 后处理，返回人话结果。常用任务名还有 "ner"（命名实体识别）、"question-answering"、"translation"、"summarization" 等，完整列表见官方文档（huggingface.co）。

理解它「背后做了什么」更重要：pipeline = AutoTokenizer + AutoModel（带任务头）+ 后处理。接下来两节把这三样拆开手动组装一遍——第 7 课的微调就需要这种手动级的理解。

## 6.4 AutoTokenizer：文本 → id

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-chinese")

enc = tok("今天天气真好")
print(enc["input_ids"])        # 长度 8：[CLS] + 6 个字的 id + [SEP]
print(enc["attention_mask"])   # 长度 8 的全 1：1=真 token，0=padding
print(tok.convert_ids_to_tokens(enc["input_ids"]))
# ['[CLS]', '今', '天', '天', '气', '真', '好', '[SEP]']

print(tok.decode(enc["input_ids"]))   # id -> 文本：[CLS] 今天天气真好 [SEP]
```

三个认知点：

1. **中文按字切分**：bert-base-chinese 是字级（每字一个 id）；英文 BERT 是 WordPiece 子词（unhappiness → un / ##hap / ##piness）。子词的动机回看第 1 课的分词讨论；
2. **特殊 token 自动添加**：[CLS] / [SEP] 呼应第 5 课 BERT 的约定——[CLS] 是分类槽位；
3. **attention_mask 是给 padding 用的**：一批句子长短不齐要补齐（padding），补的位置 attention_mask=0，告诉模型「这些位置不是内容，别注意它们」。单句无 padding 时全 1。

批量与补齐/截断（下一课微调的标准姿势）：

```python
enc = tok(
    ["这句话很短", "这句话特别特别特别特别特别长长长长长长长长"],
    padding=True,           # 补齐到批内最长
    truncation=True,        # 超过 max_length 截断
    max_length=16,
    return_tensors="pt",    # 直接返回 torch tensor
)
print(enc["input_ids"].shape)          # [2, 16]：短句被 pad、长句被截
print(enc["attention_mask"])           # 短句那行末尾出现 0（padding 位置）
```

## 6.5 AutoModel：id → 上下文向量

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-chinese")   # 12 层 encoder，768 维
model.eval()

with torch.no_grad():
    out = model(**enc)                   # enc 来自上一节（含 input_ids / attention_mask）

print(out.last_hidden_state.shape)       # [2, 16, 768]
#      ↑batch ↑seq  ↑hidden
```

`[batch, seq_len, hidden_size=768]`：每个位置一个 768 维向量，已经过 12 层双向注意力的加工——**每个字的表示都融合了全句语义**（第 3 课自注意力的成果）。取 [CLS]：`out.last_hidden_state[:, 0]` → [2, 768]，正是第 5 课预告的分类接口。

模型家族速记：`AutoModel` 返回骨架隐状态；`AutoModelForSequenceClassification` = 骨架 + 分类头（下一课主角）；从类名后缀能推断任务头。`model.config` 里藏着超参（hidden_size、num_hidden_layers、num_attention_heads……），`model.num_parameters()` 看参数量。

device 惯例照用阶段 2 的写法：`model.to(device)`、输入张量 `.to(device)`。CPU 下 bert-base 单批前向约百毫秒到秒级，教学够用。

## 6.6 Hub：找模型、读模型卡

Hub（huggingface.co）是选型入口：

1. 搜关键词（如 chinese bert sentiment），按下载量排序——下载量是社区用脚投票的结果；
2. 打开**模型卡（Model Card）**读三样：训练数据、评测结果、Usage 代码块（输入输出格式、能不能直接跑）。模型卡质量参差，没有模型卡的谨慎使用；
3. 看许可证（Apache / MIT 可商用，部分仅限研究）；
4. `Auto*.from_pretrained("作者/模型名")` 即可加载任意 Hub 模型——名字就是坐标。

选型提示：教学与 demo 用官方基座（bert-base-chinese）；真实项目优先找已在你任务上微调过的模型——也许根本不用自己微调。

## 6.7 完整 demo

整合本课内容（对应 `code/L06_hf_pipeline_demo.py`）：

```python
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

if __name__ == "__main__":
    # 0. 首次运行前先：export HF_ENDPOINT=https://hf-mirror.com

    # 1. pipeline 三行体验（背后 = tokenizer + model + 后处理）
    clf = pipeline("text-classification")
    print(clf("I love this course!"))

    # 2. 手动拆解：tokenizer
    name = "bert-base-chinese"
    tok = AutoTokenizer.from_pretrained(name)
    enc = tok(["今天天气真好", "深度学习很有意思"], padding=True, return_tensors="pt")
    print("input_ids 形状:", enc["input_ids"].shape)        # [2, 10]
    print("还原第 0 句:", tok.decode(enc["input_ids"][0]))

    # 3. 手动拆解：model
    model = AutoModel.from_pretrained(name).eval()
    with torch.no_grad():
        out = model(**enc)
    print("last_hidden_state:", out.last_hidden_state.shape)   # [2, 10, 768]
    print("[CLS] 向量形状:", out.last_hidden_state[:, 0].shape)  # [2, 768]
```

## 练习

1. 用 `pipeline("ner")` 跑一段英文，观察实体标注的输出格式。
   提示：返回是逐 token 的列表，留意词首 B-/I- 前缀。
2. 对比 `tok("Hello world")` 分别在 bert-base-chinese 与 bert-base-uncased 下的切分结果。
   提示：英文模型切出子词；中文模型对英文按字符/词片处理，观察 id 数量。
3. 打印 `model.config` 中 hidden_size、num_hidden_layers、num_attention_heads 三个值。
   提示：bert-base 是 12 层 × 768 维 × 12 头。
4. 在 Hub 搜一个中文情感分类模型，读模型卡，记下 Usage 代码与许可证。
   提示：注意它要求的输入输出格式。

## 小结与自测

1. pipeline 内部四步？
   答：下载并加载模型 → tokenize → 前向 → 后处理成可读结果。
2. attention_mask 为 0 的位置表示什么？
   答：padding 补齐位置，非真实 token，注意力应忽略。
3. last_hidden_state 的三个维度？
   答：[batch, seq_len, hidden_size]，每 token 一个上下文相关的向量。
4. 国内镜像怎么配？
   答：export HF_ENDPOINT=https://hf-mirror.com。
5. AutoModel 与 ForSequenceClassification 的区别？
   答：前者返回骨架隐状态；后者是骨架 + 分类头，直接输出 logits。

## 下一课预告

万事俱备：第 7 课把 tokenizer、BERT、分类头、阶段 2 的训练八股文组装起来——完成你人生第一次模型微调（中文情感分类）。
