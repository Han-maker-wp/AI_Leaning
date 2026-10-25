"""L07：NumPy 核心演示——创建/索引切片/广播/矩阵乘/向量化性能对比。

运行前：uv pip install numpy
运行：python3 code/L07_numpy_demo.py
"""
import time

import numpy as np


def main() -> None:
    # ---------- 1. 创建与属性：先看 shape 与 dtype ----------
    a = np.array([1, 2, 3])                  # 从 list 创建，自动推断 dtype=int64
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # 3 行 2 列
    print(f"a: shape={a.shape}, dtype={a.dtype}")
    print(f"X: shape={X.shape}, dtype={X.dtype}")        # (3, 2) float64

    zeros = np.zeros((2, 3))                 # 全 0，常用作累加器/占位
    eye = np.eye(2)                          # 单位矩阵 I
    rng = np.random.default_rng(42)          # 固定种子，结果可复现
    w = rng.normal(0, 0.1, size=(2,))        # N(0, 0.1^2) 的权重
    print(f"zeros=\n{zeros}\neye=\n{eye}\nw={w}")
    print(f"linspace(0,1,5) = {np.linspace(0, 1, 5)}")   # 画图造 x 轴常用

    # ---------- 2. 索引、切片、布尔掩码 ----------
    M = np.arange(12).reshape(3, 4)          # 0..11 重排成 3x4
    print(f"M=\n{M}")
    print(f"第 0 行：{M[0]}")
    print(f"第 2 行第 3 列：{M[2, 3]}")       # 多维索引一次写完
    print(f"前两行：\n{M[:2]}")
    print(f"所有行的第 1 列：{M[:, 1]}")      # 「批量取特征列」就是这么来的
    print(f"最后两列：\n{M[:, -2:]}")
    print(f"偶数元素（布尔掩码）：{M[M % 2 == 0]}")   # 筛样本的常用手法

    # ---------- 3. 广播：形状从末位对齐 ----------
    row_mean = M.mean(axis=0)                # 按列求均值 → shape (4,)
    centered = M - row_mean                  # (3,4) - (4,)：均值被广播到每一行
    print(f"按列均值：{row_mean}")
    print(f"去均值后：\n{centered}")

    u = np.array([[10.0], [20.0], [30.0]])   # (3,1)
    v = np.array([1.0, 2.0, 3.0, 4.0])       # (4,)
    print(f"(3,1)+(4,) 广播成 (3,4)：\n{u + v}")   # 双向拉伸，外积式加法

    # ---------- 4. 矩阵乘与点积：@ ----------
    y_pred = X @ w                           # (3,2) @ (2,) → (3,)：一次算完 3 个样本
    print(f"X @ w = {y_pred}")
    W2 = rng.normal(0, 0.1, size=(2, 5))     # 2 维输入 → 5 维输出
    H = X @ W2                               # (3,2) @ (2,5) → (3,5)
    print(f"X @ W2 的 shape = {H.shape}")    # 内维相等，外维留任

    # ---------- 5. 向量化 vs for 循环：实测性能 ----------
    n = 1_000_000
    p = rng.standard_normal(n)
    q = rng.standard_normal(n)

    t0 = time.perf_counter()
    s = 0.0
    for x, y in zip(p, q):                   # Python 层逐元素循环
        s += x * y
    t1 = time.perf_counter()
    print(f"for 循环点积: {s:.2f}，耗时 {t1 - t0:.3f}s")

    t0 = time.perf_counter()
    s2 = float(p @ q)                        # NumPy 向量化
    t1 = time.perf_counter()
    print(f"向量化点积:   {s2:.2f}，耗时 {t1 - t0:.3f}s  ← 通常快一到两个数量级")


if __name__ == "__main__":
    main()
