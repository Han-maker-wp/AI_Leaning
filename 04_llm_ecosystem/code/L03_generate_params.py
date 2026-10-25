# -*- coding: utf-8 -*-
"""
L03 采样参数对比实验（阶段 4 · 第 3 课配套代码）

用途：直观对比 greedy / temperature / top-k / top-p / repetition penalty
      对同一 prompt 生成结果的影响。

运行方式：
    pip install -U transformers torch
    # 国内网络先配置镜像（取消下一行注释）：
    # export HF_ENDPOINT=https://hf-mirror.com
    python L03_generate_params.py

说明：
- 使用 Qwen2.5-0.5B-Instruct，体量小，无 GPU 的机器（CPU）也能在可接受时间内跑完。
- transformers 接口与 generate 参数名可能随版本更新，以官方文档为准：
  https://huggingface.co/docs/transformers
"""

import os

# 国内镜像加速（如已 export 过环境变量可省略）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- 配置 ----------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # 小模型：实验首选；有 GPU 可换 7B
MAX_NEW_TOKENS = 80                        # 每组实验生成的最大新 token 数，控制实验时长

# 普通开放性问题：观察多样性与稳定性
PROMPT_NORMAL = "用两句话介绍你自己，再写一句关于秋天的诗。"
# 诱导复读的问题：观察 repetition penalty 的效果
PROMPT_REPEAT = "从 1 数到 30，只用逗号分隔数字：1, 2, 3,"


def build_chat_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    """把用户输入渲染成模型约定的 chat template（L03 第 5 节：特殊 token 约定）。"""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()  # 推理不需要梯度，省显存
def generate(model, tokenizer, prompt: str, seed: int = 42, **gen_kwargs) -> str:
    """按给定采样参数生成一次，返回新生成部分（不含 prompt）的文本。"""
    torch.manual_seed(seed)  # 固定随机种子，保证采样实验可复现（greedy 本身确定）
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # pad_token_id 显式给定，避免部分 tokenizer 未设置时产生警告
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    # 只解码 prompt 之后的新增 token
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def show(title: str, text: str) -> None:
    """统一的结果打印格式，方便对比阅读。"""
    print("\n" + "=" * 60)
    print(f"[{title}]")
    print("-" * 60)
    print(text.strip())
    print("=" * 60)


def main():
    print(f"加载模型：{MODEL_ID}（首次运行会自动下载，约 1GB）")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",   # 按模型卡精度加载；参数名以官方文档为准
        device_map="auto",    # 有 GPU 上 GPU，没有则 CPU
    )

    # ---------- 实验组 0：greedy vs 采样（对照基准） ----------
    prompt = build_chat_prompt(tokenizer, PROMPT_NORMAL)
    show("greedy（贪心，确定性）", generate(model, tokenizer, prompt, do_sample=False))
    show(
        "采样 T=0.7, top_p=0.9（对话常用默认）",
        generate(model, tokenizer, prompt, do_sample=True, temperature=0.7, top_p=0.9),
    )

    # ---------- 实验组 1：temperature 三档对比 ----------
    # 预期：T=0.1 接近确定/保守；T=1.0 正常；T=1.8 明显放飞（可能不通顺）
    for t in (0.1, 1.0, 1.8):
        show(
            f"temperature={t}（do_sample=True, top_k=0, top_p=1.0 关闭其他干预）",
            generate(
                model, tokenizer, prompt,
                do_sample=True, temperature=t, top_k=0, top_p=1.0,
            ),
        )

    # ---------- 实验组 2：top-k 对比 ----------
    # 预期：k=1 等价于贪心；k=5 保守多样；k=50 候选很宽（结合 T=1 观察差异）
    for k in (1, 5, 50):
        show(
            f"top_k={k}（do_sample=True, T=1.0）",
            generate(
                model, tokenizer, prompt,
                do_sample=True, temperature=1.0, top_k=k, top_p=1.0,
            ),
        )

    # ---------- 实验组 3：top-p 对比 ----------
    # 预期：p=0.1 极保守；p=0.95 常用默认；对比同为“收缩候选”，与固定 k 的差别在于候选数是动态的
    for p in (0.1, 0.95):
        show(
            f"top_p={p}（do_sample=True, T=1.0, top_k=0）",
            generate(
                model, tokenizer, prompt,
                do_sample=True, temperature=1.0, top_k=0, top_p=p,
            ),
        )

    # ---------- 实验组 4：repetition penalty（用诱导复读的 prompt） ----------
    prompt_rep = build_chat_prompt(tokenizer, PROMPT_REPEAT)
    show(
        "复读诱导 prompt · 无惩罚（T=0.3, top_p=0.9）",
        generate(model, tokenizer, prompt_rep,
                 do_sample=True, temperature=0.3, top_p=0.9),
    )
    show(
        "复读诱导 prompt · repetition_penalty=1.3",
        generate(model, tokenizer, prompt_rep,
                 do_sample=True, temperature=0.3, top_p=0.9,
                 repetition_penalty=1.3),
    )

    print(
        "\n实验完成后，请回到讲义 L03 第 6 节，为每组实验写一句话结论。\n"
        "注意观察两点：1) temperature=0.1 与 greedy 的输出是否几乎一致；\n"
        "2) 高 temperature 下事实类内容是否开始出错。"
    )


if __name__ == "__main__":
    main()
