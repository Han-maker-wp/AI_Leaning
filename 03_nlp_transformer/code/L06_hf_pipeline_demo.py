"""L06：HuggingFace 三件套体验——pipeline / AutoTokenizer / AutoModel。

首次运行前配置国内镜像（写进 ~/.bashrc 更好）：
    export HF_ENDPOINT=https://hf-mirror.com
首次运行会下载模型：英文情感模型（约 250MB）+ bert-base-chinese（约 400MB）。

运行：python L06_hf_pipeline_demo.py
对应讲义：03_nlp_transformer/L06_HuggingFace实战入门.md
"""
import torch
from transformers import AutoModel, AutoTokenizer, pipeline


def main():
    # ---------- 1. pipeline：三行代码完成推理 ----------
    # 背后 = 下载模型 + tokenize + 前向 + 后处理；默认任务模型是英文情感分类
    clf = pipeline("text-classification")
    print("pipeline 情感分析:", clf("I love this course!"))

    # ---------- 2. AutoTokenizer：文本 -> token id ----------
    name = "bert-base-chinese"
    tok = AutoTokenizer.from_pretrained(name)

    single = tok("今天天气真好")
    print("单句 input_ids:", single["input_ids"])          # [CLS] + 6 个字 + [SEP]，共 8 个
    print("单句 attention_mask:", single["attention_mask"])  # 全 1（无 padding）
    print("切成 token:", tok.convert_ids_to_tokens(single["input_ids"]))
    print("decode 还原:", tok.decode(single["input_ids"]))

    # 批量 + 补齐 + 截断（微调时的标准姿势）
    enc = tok(
        ["这句话很短", "这句话特别特别特别特别特别长长长长长长长长"],
        padding=True,           # 补齐到批内最长
        truncation=True,        # 超过 max_length 截断
        max_length=16,
        return_tensors="pt",
    )
    print("批量 input_ids 形状:", enc["input_ids"].shape)   # [2, 16]
    print("批量 attention_mask:\n", enc["attention_mask"])  # 短句那行末尾出现 0

    # ---------- 3. AutoModel：token id -> 上下文向量 ----------
    model = AutoModel.from_pretrained(name).eval()
    with torch.no_grad():                    # 推理不建图（阶段 2 的习惯）
        out = model(**enc)
    print("last_hidden_state 形状:", out.last_hidden_state.shape)   # [2, 16, 768]
    print("[CLS] 向量形状:", out.last_hidden_state[:, 0].shape)      # [2, 768]
    print("模型参数量:", model.num_parameters())
    print("层数:", model.config.num_hidden_layers,
          "| 隐层维度:", model.config.hidden_size,
          "| 注意力头数:", model.config.num_attention_heads)


if __name__ == "__main__":
    main()
