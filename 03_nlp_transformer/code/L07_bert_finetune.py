"""L07：BERT 中文情感分类微调（CPU 可跑的教学版）——人生第一次微调。

首次运行前：
    1) 安装依赖：uv pip install torch transformers
    2) 配置国内镜像：export HF_ENDPOINT=https://hf-mirror.com
    3) 首次会下载 bert-base-chinese（约 400MB），缓存在 ~/.cache/huggingface/

运行：python L07_bert_finetune.py
预计耗时（CPU）：训练约 1~5 分钟（48 条样本、3 个 epoch）
产物：bert_sentiment/ 目录（微调后的模型与 tokenizer）
对应讲义：03_nlp_transformer/L07_BERT文本分类微调.md

说明：内置 48 条人工样本只为跑通全流程，指标无统计意义；
     替换真实数据只需把 POSITIVE/NEGATIVE 换成你的 (text, label) 列表或 CSV。
"""
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification

# ---------- 超参（微调 BERT 的社区共识起点） ----------
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5              # 比从零训练小两个数量级：保护预训练权重
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------- 教学数据：48 条内置样本（24 正 / 24 负） ----------
POSITIVE = [
    "这部电影太精彩了，强烈推荐", "物流很快，包装完好", "味道非常好，下次还来",
    "客服态度特别好，很有耐心", "性价比超高，值得购买", "用了两周，体验很棒",
    "画质清晰，音质出色", "孩子很喜欢这个玩具", "这本书写得很透彻",
    "酒店位置很好，房间干净", "老师讲课生动有趣", "软件运行流畅，没有广告",
    "做工精细，材质很舒服", "快递员送货上门，服务周到", "屏幕显示效果好",
    "续航时间长，充电很快", "这款耳机音质超出预期", "环境优雅，菜品新鲜",
    "课程内容干货满满", "鞋子轻便舒适，尺码标准", "安装简单，说明书清楚",
    "游戏画面精美，可玩性高", "咖啡香气浓郁，口感顺滑", "售后响应及时，问题解决了",
]
NEGATIVE = [
    "太失望了，和描述完全不符", "物流太慢，等了半个月", "味道一般，不会再买了",
    "客服态度很差，爱答不理", "质量太差，用两天就坏了", "拍照效果模糊不清",
    "尺寸不对，穿着不合身", "内容空洞，浪费钱", "房间隔音很差，一夜没睡好",
    "讲课枯燥，全程念PPT", "广告太多，体验极差", "材质粗糙，有线头",
    "包装破损，商品被压坏", "屏幕有坏点，申请退货", "电池不耐用，发热严重",
    "耳机戴着不舒服，夹耳朵", "菜品不新鲜，服务怠慢", "课程更新太慢，内容过时",
    "鞋子磨脚，走不了远路", "安装困难，说明书混乱", "游戏经常闪退，优化差",
    "咖啡太苦，像是隔夜的", "售后推诿，一直不处理", "性价比低，后悔买了",
]


class SentimentDataset(Dataset):
    """阶段 2 Dataset 协议：__len__ + __getitem__（像实现 Java 的 List 接口）。

    __getitem__ 返回原始文本；tokenize 推迟到 collate 批级别做（padding 长度
    依赖批内最长句）。
    """

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate_fn(tokenizer, max_length=MAX_LENGTH):
    """把一批 (text, label) 变成模型需要的 tensor 字典。"""
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)   # BertForSequenceClassification 直接吃 labels
        return enc
    return collate


def train_one_epoch(model, loader, optimizer):
    """训练循环八股文（与阶段 2 L06 骨架完全一致）。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for enc in loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}   # 整个输入字典搬设备
        optimizer.zero_grad()                             # ① 清梯度
        out = model(**enc)                                # ② 前向（含 labels -> 自动算 loss）
        loss = out.loss                                   # ③ 损失
        loss.backward()                                   # ④ 反向
        optimizer.step()                                  # ⑤ 更新
        total_loss += loss.item() * enc["labels"].size(0)
        correct += (out.logits.argmax(-1) == enc["labels"]).sum().item()
        n += enc["labels"].size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_with_cm(model, loader, num_labels=2):
    """评估：准确率 + 混淆矩阵（行=真实，列=预测）。"""
    model.eval()
    cm = torch.zeros(num_labels, num_labels, dtype=torch.long)
    for enc in loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        preds = model(**enc).logits.argmax(-1)
        for p, t in zip(preds, enc["labels"]):
            cm[t, p] += 1
    acc = (cm.diag().sum() / cm.sum()).item()
    return acc, cm


def predict(model, tokenizer, text):
    """对单条文本做推理，返回各类别概率。"""
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    prob = torch.softmax(logits, dim=-1)[0]
    return {"负面": round(prob[0].item(), 4), "正面": round(prob[1].item(), 4)}


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------- 数据：48 条 -> 40 训练 / 8 验证 ----------
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
    pairs = list(zip(texts, labels))
    random.shuffle(pairs)
    texts, labels = zip(*pairs)
    train_texts, train_labels = list(texts[:40]), list(labels[:40])
    val_texts, val_labels = list(texts[40:]), list(labels[40:])
    print(f"训练 {len(train_texts)} 条 / 验证 {len(val_texts)} 条")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate_fn(tokenizer)
    train_loader = DataLoader(SentimentDataset(train_texts, train_labels),
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SentimentDataset(val_texts, val_labels),
                            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # ---------- 模型：预训练 BERT + 新初始化分类头 ----------
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    total = model.num_parameters()
    head = sum(p.numel() for p in model.classifier.parameters())
    print(f"总参数 {total:,}，其中新初始化分类头仅 {head:,} 个（这就是微调）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---------- 训练（八股文原样服役） ----------
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        val_acc, cm = evaluate_with_cm(model, val_loader)
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_acc={val_acc:.4f}")
        print(f"  混淆矩阵（行=真实[负,正] 列=预测[负,正]）:\n{cm}")

    # ---------- 保存与推理 ----------
    model.save_pretrained("bert_sentiment")
    tokenizer.save_pretrained("bert_sentiment")
    print("模型已保存到 ./bert_sentiment/")

    for text in ["味道好极了，下次还来", "太差了，再也不买"]:
        print(f"预测 {text!r} -> {predict(model, tokenizer, text)}")


if __name__ == "__main__":
    main()
