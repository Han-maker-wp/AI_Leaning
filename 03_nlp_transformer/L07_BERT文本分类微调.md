# 第 7 课：BERT 文本分类微调——人生第一次微调

> 🎬 开场钩子：值得记一笔的日子——这是你人生第一次微调一个大模型。从今天起，「预训练模型」对你不再是 API 背后的黑盒，而是一份可以上手改造的权重。多年 Java 经验早就告诉你：能在别人的代码上安全地打补丁，才算真正读懂了它。今天，你给 BERT 打上第一个补丁。

⏱ 建议学时：3 小时（脚本 CPU 运行约 1~5 分钟，首次下载模型约 400MB）

## 本课目标

1. 完成端到端微调：数据 → tokenizer → 模型 → 训练 → 评估 → 保存 → 推理；
2. 说清 BertForSequenceClassification 的结构（预训练主体 + 新初始化分类头）与两者参数比例；
3. 用阶段 2 的训练八股文驱动 BERT（体会「模板复用」）；
4. 会算准确率与混淆矩阵，并解释结果；
5. 能定义「微调」，说清它与从零训练的成本差异，以及它与毕业课题的关系。

## 7.1 任务设定：中文评论情感二分类

输入一句中文评论，输出 0（负面）/ 1（正面）。这是 NLP 的 Hello World，也是电商、外卖、电影评分场景的真实需求。

教学数据：讲义与代码内置 48 条人工小样本（24 正 24 负），目的是**跑通全流程**——样本太少，指标本身没有统计意义（还可能轻微过拟合），请把注意力放在流程与代码结构上。替换为真实数据的方法在 7.2 说明。

## 7.2 数据准备：自定义 Dataset

延续阶段 2 的 Dataset 协议（`__len__` + `__getitem__`，像实现 Java 的 List 接口）。设计决策：`__getitem__` 返回**原始文本与标签**，tokenize 推迟到批级别（collate_fn）做——原因：padding 长度取决于**批内最长句**，必须看到整批才能确定。

```python
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate_fn(tokenizer, max_length=64):
    """把一批 (text, label) 变成模型需要的 tensor 字典。"""
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return collate
```

**如何换成真实数据集**：任何「两列（text, label）」的 CSV / jsonl 都能直接接上——

```python
import csv

with open("my_data.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
texts = [r["text"] for r in rows]
labels = [int(r["label"]) for r in rows]
```

规模建议：几千到几万条时微调才开始有意义（中文情感任务几万条可达 90%+ 准确率量级）。也可以用 HF 的 datasets 库 `load_dataset` 从 Hub 拉公开数据集（如英文 IMDB；中文情感可在 Hub 搜 ChnSentiCorp）。内置 48 条是为了 CPU 上分钟级跑完全流程的教学权衡。

## 7.3 tokenizer 批处理细节

`collate` 里的三个参数（上一课讲过，这里是实战）：

- `padding=True`：批内补齐到最长句，attention_mask 自动标 0，BERT 忽略这些位置；
- `truncation=True, max_length=64`：评论再长也只留 64 token——教学数据都短，真实数据用于防 OOM / 拖慢；
- `return_tensors="pt"`：直接产出 torch tensor，可 `model(**enc)`。

## 7.4 模型：BertForSequenceClassification

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2,        # 二分类
)
```

它就是第 5 课讲的接口的官方实现：

```
BERT encoder（12 层，约 1.1 亿参数，预训练加载）
    ↓ last_hidden_state[:, 0]      # 取 [CLS]
Dropout(0.1)
    ↓
Linear(768 → 2)                    # 分类头：768*2 + 2 = 1538 个参数，随机初始化
```

震撼的数字：**约 1.1 亿个预训练参数 vs 1538 个新参数**。微调时两者一起训练——新头从随机开始快速学会「读」CLS 向量，预训练主体用小学习率轻微适应任务。控制台会警告分类头是 newly initialized——正常，且正是预期。

预训练权重给了什么？一个从没见过你这些评论的模型，已经懂中文的字词组合、语法结构、常见情感表达——48 条样本也能学出个样子，靠的就是这 1.1 亿参数里的先验。从零训练同结构模型需要百万级标注 + 天级 GPU，这就是微调的意义（第 5 课范式的实例化）。

> 🍿 彩蛋｜微调的姿势：像极了在成熟开源框架上做定制——底座（BERT）不 fork 不重写，只挂一个自己的扩展点（分类头）小步适配。Java 老兵最懂这套哲学：能扩展就别改动，能注入就别硬编码。

## 7.5 完整训练脚本

主体 = 阶段 2 八股文 + 换模型 + 换数据管道 + 换超参。完整可运行版在 `code/L07_bert_finetune.py`，此处逐段。

超参选择（微调 BERT 的社区共识起点）：

```python
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5          # 比从零训练小两个数量级！预训练权重已是好解，大步会破坏它
WEIGHT_DECAY = 0.01
```

训练循环（对比 L06：骨架一字不差，只是 loss 由模型内部计算——传入 labels 时 BertForSequenceClassification 直接返回 loss）：

```python
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for enc in loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}   # 整个输入字典搬设备
        optimizer.zero_grad()                             # ① 清梯度
        out = model(**enc)                                # ② 前向（含 labels → 自动算 loss）
        loss = out.loss                                   # ③ 损失
        loss.backward()                                   # ④ 反向
        optimizer.step()                                  # ⑤ 更新

        total_loss += loss.item() * enc["labels"].size(0)
        correct += (out.logits.argmax(-1) == enc["labels"]).sum().item()
        n += enc["labels"].size(0)
    return total_loss / n, correct / n
```

优化器用 AdamW（第 5 阶段 2 课的结论：Transformer 微调标配）：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
```

CPU 时间预算：48 样本 × 3 epoch ÷ batch 8 = 18 个 step，每 step 约 1~5 秒——总共一两分钟。换成几千条数据 CPU 要数小时（这正是真项目需要 GPU 的地方：云 GPU 按小时租，脚本不用改，device 惯例已兼容）。

## 7.6 评估：准确率 + 混淆矩阵

```python
@torch.no_grad()
def evaluate_with_cm(model, loader, num_labels=2):
    model.eval()
    cm = torch.zeros(num_labels, num_labels, dtype=torch.long)
    for enc in loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        preds = model(**enc).logits.argmax(-1)
        for p, t in zip(preds, enc["labels"]):
            cm[t, p] += 1                 # 行 = 真实，列 = 预测
    acc = cm.diag().sum() / cm.sum()
    return acc.item(), cm
```

混淆矩阵读法（二分类）：

```
                 预测负面    预测正面
真实负面          TN          FP（把差评判成好评——电商最贵的一种错）
真实正面          FN（把好评判成差评）  TP
```

它比单看准确率多回答一个问题：**错都错在哪边**。类别不平衡时准确率会说谎（90% 样本是正面时，全猜正面也有 90% 准确率），混淆矩阵与 precision / recall 是照妖镜——概念与机器学习课一致，这里是实现。

## 7.7 保存与推理

```python
# 保存（HF 风格：模型 + tokenizer 存进一个目录）
model.save_pretrained("bert_sentiment")
tokenizer.save_pretrained("bert_sentiment")

# 加载与推理
from transformers import AutoTokenizer, BertForSequenceClassification

model2 = BertForSequenceClassification.from_pretrained("bert_sentiment")
tok2 = AutoTokenizer.from_pretrained("bert_sentiment")

def predict(text: str) -> dict:
    enc = tok2(text, return_tensors="pt", truncation=True, max_length=64)
    model2.eval()
    with torch.no_grad():
        logits = model2(**enc).logits
    prob = torch.softmax(logits, dim=-1)[0]
    return {"负面": prob[0].item(), "正面": prob[1].item()}
```

与阶段 2 的 `state_dict` 方式思想等价（存数据、不存代码）；`save_pretrained` 是 HF 生态的封装（连带 config 与 tokenizer），跨机器部署更顺。想统一用 `torch.save(model.state_dict(), ...)` 管理也完全可以。

## 7.8 点题：这就是「微调」

回头看这次实验，以及它在你路线图上的位置：

```
预训练（别人做了：GPU 集群、月级、无标注海量文本）
        ↓
微调（你刚做的：笔记本 CPU、分钟级、几十条标注）   ← 你在这里
        ↓
推理部署（predict 函数）
```

毕业课题相对本课只是两个替换：

1. **换基座**：BERT（1.1 亿参数、encoder）→ LLM（70 亿+、decoder-only）。数据变成对话/指令格式，但训练循环八股文**不变**；
2. **换微调方式**：全参数微调（本课）→ LoRA / QLoRA（冻结主体、只训练插入的小 adapter）——因为阶段 2 L08 算过：7B 模型全参微调仅参数+梯度+Adam 就要约 112GB 显存，单卡放不下。训练循环八股文**依然不变**，变的只是哪些参数可训。

所以本课的价值不在那 48 条小数据，而在于：**你已经完整走过一遍工业界微调的最小闭环**。

> 🍿 彩蛋｜小火慢炖：从零训练像猛火爆炒，微调像小火慢炖——汤底（预训练权重）是别人用大代价熬好的，你只负责把自己那味料融进去。火太大会把汤熬糊：这就是为什么微调要压着学习率走。

## 练习

1. 打印 `list(model.named_parameters())` 的前几项，找出分类头的参数名。
   提示：classifier.weight / classifier.bias。
2. 把 LR 改成 2e-3 重跑，观察训练 loss 的行为并解释。
   提示：大 lr 会破坏预训练权重，loss 可能先降后乱甚至 nan。
3. 冻结 BERT 主体（`for p in model.bert.parameters(): p.requires_grad = False`）只训分类头，对比效果与单步耗时。
   提示：这是「特征提取」用法；小数据下有时反而更稳。
4. 换成你找的真实数据集（几百条起步）重跑，报告准确率与混淆矩阵。
   提示：两列 CSV 即可接上 7.2 的代码。

## 小结与自测

1. BertForSequenceClassification 的结构？
   答：预训练 BERT encoder + Dropout + Linear(768, num_labels)（新初始化）。
2. 微调为什么用小学习率（2e-5）？
   答：预训练权重已在良好区域，大步长会破坏已学到的知识。
3. tokenize 为什么放在 collate 而不是 __getitem__？
   答：padding 长度依赖批内最长句，必须批级别处理。
4. 混淆矩阵比准确率多告诉你什么？
   答：错误的方向分布（哪类被误判成哪类），暴露类别不平衡下的假象。
5. 毕业课题相对本课的两个替换是什么？
   答：换基座（BERT→LLM）、换方式（全参→LoRA）；训练循环不变。

## 下一课预告

最后一课换个视角：不用任何预训练模型，从数字符开始写一个字符级语言模型，亲手体验「预测下一词」目标的魔力，并看清它与你将微调的 GPT 系模型之间只差「注意力」这一步。
