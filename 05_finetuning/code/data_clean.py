# -*- coding: utf-8 -*-
"""
data_clean.py —— 指令数据集（alpaca 格式）清洗脚本
阶段 5 · L02 配套代码

功能：
  1. 格式校验：字段完整、类型正确、output 非空、剔除特殊 token 字符串
  2. 长度过滤：instruction/output 过短（敷衍）或过长（失控）的样本
  3. 精确去重：规范化后按 instruction+input 哈希去重
  4. 近似去重：基于字符 n-gram 的 Jaccard 相似度（小数据集两两比对够用）

用法：
    python data_clean.py --in alpaca_sample.json --out cleaned.json
    python data_clean.py --in raw.json --out cleaned.json \
        --min-output 10 --dup-threshold 0.8

说明：
    - 近似去重为 O(n^2)，适合几千~几万条；更大规模建议换 MinHash/LSH 思路。
    - 参数默认值为常见起点，请按你的数据分布调整。
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter

# ---------- 可调阈值（默认值） ----------
DEFAULT_MIN_INSTRUCTION = 5      # instruction 少于该字符数视为可疑（太短任务不明）
DEFAULT_MIN_OUTPUT = 10          # output 少于该字符数视为敷衍（如“好的”“略”）
DEFAULT_MAX_OUTPUT = 8000        # output 超长视为失控样本（按任务调整）
DEFAULT_DUP_THRESHOLD = 0.85     # Jaccard 相似度超过该值判为近似重复

# 常见 LLM 特殊 token 的检测正则（不同家族不同，按需补充）
SPECIAL_TOKEN_PAT = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|</s>|<s>|<|system|>"
)


def normalize_text(text: str) -> str:
    """规范化文本：去首尾空白、压缩内部连续空白、统一小写——精确去重前的标准动作。"""
    return re.sub(r"\s+", " ", text.strip()).lower()


def shingles(text: str, n: int = 3) -> set:
    """字符 n-gram 集合，用于近似去重的 Jaccard 相似度计算。"""
    text = normalize_text(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """两个集合的 Jaccard 相似度 = 交集 / 并集。"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate(item: dict, idx: int, min_instruction: int, min_output: int,
             max_output: int) -> tuple[bool, str]:
    """格式与长度校验：返回 (是否通过, 不通过原因)。"""
    # 字段完整性：instruction 与 output 必须存在且为字符串（input 可选）
    for field in ("instruction", "output"):
        if field not in item or not isinstance(item[field], str):
            return False, f"缺少字段或类型错误: {field}"
    instruction = item["instruction"]
    output = item["output"]
    inp = item.get("input", "") or ""   # input 缺省视为空串

    # 长度规则
    if len(instruction.strip()) < min_instruction:
        return False, "instruction 过短"
    if len(output.strip()) < min_output:
        return False, "output 过短（疑似敷衍）"
    if len(output) > max_output:
        return False, "output 超长"

    # 内容规则：混入特殊 token 会导致训练后格式崩坏（阶段5 L10 坑4）
    if SPECIAL_TOKEN_PAT.search(output) or SPECIAL_TOKEN_PAT.search(instruction):
        return False, "包含特殊 token 字符串"
    # 敷衍占位模式（可按需扩充）
    if re.fullmatch(r"(好的|嗯|略|此处省略|pass)[。.!！]?", output.strip()):
        return False, "占位符式敷衍输出"
    return True, ""


def clean(data: list, min_instruction: int, min_output: int,
          max_output: int, dup_threshold: float) -> tuple[list, Counter]:
    """主清洗流程：格式校验 -> 精确去重 -> 近似去重。返回 (清洗后数据, 统计报告)。"""
    stats = Counter()
    passed = []

    # ---------- 第 1 步：格式与长度校验 ----------
    for idx, item in enumerate(data):
        ok, reason = validate(item, idx, min_instruction, min_output, max_output)
        if ok:
            passed.append(item)
        else:
            stats[f"过滤:{reason}"] += 1

    # ---------- 第 2 步：精确去重 ----------
    seen_hashes = set()
    exact_unique = []
    for item in passed:
        key = normalize_text(item["instruction"] + "\n" + (item.get("input", "") or ""))
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            stats["去重:精确重复"] += 1
            continue
        seen_hashes.add(h)
        exact_unique.append(item)

    # ---------- 第 3 步：近似去重（O(n^2)，小数据集适用） ----------
    result = []
    kept_shingle_sets = []   # 只对 instruction+input 算相似度：同一问题不同答案才是重复
    for item in exact_unique:
        s = shingles(item["instruction"] + "\n" + (item.get("input", "") or ""))
        is_dup = False
        for prev in kept_shingle_sets:
            if jaccard(s, prev) >= dup_threshold:
                is_dup = True
                break
        if is_dup:
            stats["去重:近似重复"] += 1
        else:
            kept_shingle_sets.append(s)
            result.append(item)

    stats["保留样本"] = len(result)
    return result, stats


def main():
    parser = argparse.ArgumentParser(description="alpaca 格式数据清洗脚本")
    parser.add_argument("--in", dest="in_file", required=True, help="输入 json 文件（对象数组）")
    parser.add_argument("--out", dest="out_file", required=True, help="输出清洗后 json")
    parser.add_argument("--min-instruction", type=int, default=DEFAULT_MIN_INSTRUCTION)
    parser.add_argument("--min-output", type=int, default=DEFAULT_MIN_OUTPUT)
    parser.add_argument("--max-output", type=int, default=DEFAULT_MAX_OUTPUT)
    parser.add_argument("--dup-threshold", type=float, default=DEFAULT_DUP_THRESHOLD,
                        help="近似重复的 Jaccard 阈值（0~1，越小越严格）")
    args = parser.parse_args()

    # 读入：要求顶层是 JSON 数组
    with open(args.in_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit("错误：输入文件顶层应为 JSON 数组")

    result, stats = clean(data, args.min_instruction, args.min_output,
                          args.max_output, args.dup_threshold)

    # 写出：剔除 _note 等以下划线开头的辅助字段（教学瑕疵标注等，不该进入训练）
    cleaned = [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in result
    ]
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    # 统计报告：每一项被去掉的原因与数量
    print("=" * 46)
    print(f"清洗报告：{args.in_file}")
    print(f"  输入样本数 : {len(data)}")
    for key, val in sorted(stats.items()):
        print(f"  {key:<14}: {val}")
    print(f"  已写入     : {args.out_file}")
    print("=" * 46)


if __name__ == "__main__":
    main()
