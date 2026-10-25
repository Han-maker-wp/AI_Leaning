"""L02_chunk_demo.py —— 文档分块策略对比（阶段 2 · 第 2 课配套代码）

实现两种分块并排对比：
  1. 固定长度分块：按字符数硬切（含 overlap）
  2. 递归字符分块：按分隔符优先级（段落→行→句号→逗号→空格）从粗到细切
纯标准库，离线可跑。观察点：固定切割如何"腰斩"指代，递归切割如何沿自然边界下刀。
"""

CHUNK_SIZE = 120   # 每块目标字符数（练习里改成 60 / 240 对比）
OVERLAP = 24       # 相邻块重叠字符数（约 10%-20%）

SAMPLE_DOC = """# 员工报销制度

## 报销范围

差旅费、培训费与办公用品采购均可报销。报销单需在消费发生后 30 天内提交，
逾期系统将自动关闭入口。该期限适用于以下三类人员：正式员工、实习生与经审批的外部顾问。

## 退款与冲销

已提交的报销单如填写错误，可在财务审核前撤回修改。若款项已发放，
需在 5 个工作日内发起冲销流程，逾期将由财务统一处理。

## 审批流程

直属主管审批后自动流转至财务复核，金额超过 5000 元需部门总监加签。
审批平均耗时 2 个工作日，旺季可能延长至 5 个工作日。"""


def fixed_size_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """固定长度分块：按字符硬切，最简单也最暴力——边界质量全凭运气。"""
    chunks = []
    step = max(1, size - overlap)      # 步长 = 块长 - 重叠
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece.strip())
        if start + size >= len(text):
            break
    return chunks


def recursive_chunks(text: str, size: int = CHUNK_SIZE,
                     separators: list[str] | None = None) -> list[str]:
    """递归字符分块：优先用大分隔符（段落）切，块超长再降级到更细的分隔符。"""
    if separators is None:
        separators = ["\n\n", "\n", "。", "，", " "]
    # 收集所有不超过 size 的完整片段；超长的片段递归降级处理
    pieces: list[str] = []

    def split_by(sep_idx: int, s: str) -> None:
        if len(s) <= size:                       # 已经够小：直接收
            if s.strip():
                pieces.append(s.strip())
            return
        if sep_idx >= len(separators):           # 没有更细的分隔符了：硬切兜底
            for i in range(0, len(s), size):
                if s[i:i + size].strip():
                    pieces.append(s[i:i + size].strip())
            return
        sep = separators[sep_idx]
        parts = s.split(sep) if sep else list(s)
        kept = ""
        for part in parts:
            candidate = (kept + sep + part) if kept else part
            if len(candidate) <= size:
                kept = candidate                 # 还装得下：继续装（尽量沿自然边界成块）
            else:
                if kept.strip():
                    pieces.append(kept.strip())
                if len(part) > size:             # 单个片段仍超长：降级到更细的分隔符
                    split_by(sep_idx + 1, part)
                    kept = ""
                else:
                    kept = part
        if kept.strip():
            pieces.append(kept.strip())

    split_by(0, text)
    return pieces


def show(title: str, chunks: list[str]) -> None:
    print(f"\n{'=' * 20} {title}（共 {len(chunks)} 块）{'=' * 20}")
    for i, c in enumerate(chunks):
        preview = c if len(c) <= 70 else c[:70] + "…"
        print(f"  [{i}] ({len(c):3d}字) {preview}")


def main() -> None:
    print(f"文档总长 {len(SAMPLE_DOC)} 字；chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")

    fixed = fixed_size_chunks(SAMPLE_DOC)
    rec = recursive_chunks(SAMPLE_DOC)
    show("固定长度分块", fixed)
    show("递归字符分块", rec)

    # 观察点：固定切割常把"该期限适用于"和它后面的说明切进两块；
    # 递归切割沿段落/句子边界下刀，指代关系基本保留在一个块内。
    print("\n观察：对比两种切法中「该期限」「该流程」等指代词与所指对象的距离。")
    print("练习：把 CHUNK_SIZE 改成 60 / 240 再跑，感受块形态与语义完整度的变化。")


if __name__ == "__main__":
    main()
