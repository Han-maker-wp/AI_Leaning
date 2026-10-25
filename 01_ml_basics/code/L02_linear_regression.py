"""L02：线性回归——生成数据、手写 MSE、闭式解与拟合线可视化。

运行前：uv pip install numpy matplotlib
运行：python3 code/L02_linear_regression.py
输出：L02_fit.png
"""
import numpy as np
import matplotlib.pyplot as plt


def mse_loss(y_pred: np.ndarray, y: np.ndarray) -> float:
    """均方误差 MSE = (1/n) * Σ (ŷ - y)^2。

    (y_pred - y) 是形状 (n,) 的残差向量，与自身点积即平方和——向量化写法，
    等价于残差向量的范数平方 ‖e‖² = eᵀe 再除以 n。
    """
    n = y.shape[0]
    return float((y_pred - y) @ (y_pred - y) / n)


def add_bias(X: np.ndarray) -> np.ndarray:
    """给 X 右侧拼一列 1，把 b 吸收进 w（x0 恒为 1）。X: (n,d) → (n,d+1)"""
    n = X.shape[0]
    return np.hstack([X, np.ones((n, 1))])


def main() -> None:
    # 1. 造「真实规律 + 噪声」的数据：y = 2x + 1 + ε，ε ~ N(0, 0.8²)
    rng = np.random.default_rng(0)
    n = 120
    X = rng.uniform(0, 5, size=(n, 1))              # (n,1)：保持二维，习惯 batch 维
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.8, size=n)

    # 2. 用「随手猜」的参数算损失，亲眼看到损失随参数变化（b 固定 1.0 只看 w）
    #    —— 「损失是参数的函数」的一维切面演示
    for w_try in [0.0, 1.0, 2.0, 3.0]:
        y_hat = w_try * X[:, 0] + 1.0
        print(f"w = {w_try:.1f} → MSE = {mse_loss(y_hat, y):.4f}")

    # 3. 闭式解（正规方程）：w* = (XᵀX)^{-1} Xᵀ y
    Xb = add_bias(X)                                 # (n,2)，b 并入 w
    w_closed = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y   # 求得 [w, b]
    print(f"闭式解：w = {w_closed[0]:.4f}, b = {w_closed[1]:.4f}")
    print(f"闭式解的 MSE = {mse_loss(Xb @ w_closed, y):.4f}")

    # 4. 可视化：散点 + 拟合线
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], y, s=15, alpha=0.6, label="data")
    xs = np.linspace(0, 5, 50)
    plt.plot(xs, w_closed[0] * xs + w_closed[1], "r-", lw=2, label="closed-form fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression fit")
    plt.legend()
    plt.savefig("L02_fit.png", dpi=150)
    plt.show()
    print("拟合线图已保存为 L02_fit.png")


if __name__ == "__main__":
    main()
