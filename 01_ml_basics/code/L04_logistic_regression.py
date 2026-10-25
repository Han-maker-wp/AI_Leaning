"""L04：纯 numpy 逻辑回归二分类——训练循环 + loss 曲线 + 决策边界。

这也是阶段 Checkpoint（从零实现逻辑回归）的雏形：独立完成时别抄这份代码。
运行前：uv pip install numpy matplotlib
运行：python3 code/L04_logistic_regression.py
输出：L04_loss_curve.png 与 L04_decision_boundary.png
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^{-z})，把任意实数压进 (0,1) 当概率用。"""
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    """二元交叉熵：-(1/n) Σ [y·log p + (1-y)·log(1-p)]。

    np.clip 把 p 限制在 [eps, 1-eps]，避免 log(0) 产生 inf。
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def make_blobs(rng: np.random.Generator, n_per_class: int = 150):
    """生成两个可分的二维高斯团簇：类 0 中心 (-2,-2)，类 1 中心 (2,2)。"""
    X0 = rng.normal(loc=-2.0, scale=1.2, size=(n_per_class, 2))
    X1 = rng.normal(loc=2.0, scale=1.2, size=(n_per_class, 2))
    X = np.vstack([X0, X1])                                       # (2n, 2)
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])   # (2n,)
    return X, y


def main() -> None:
    rng = np.random.default_rng(42)
    X, y = make_blobs(rng)

    # 参数初始化与超参数
    w = np.zeros(X.shape[1])                   # (2,)
    b = 0.0
    lr = 0.1
    epochs = 200
    history = []

    # 训练循环（结构与 L03 完全一致：前向 → 损失 → 梯度 → 更新）
    for epoch in range(1, epochs + 1):
        p = sigmoid(X @ w + b)                 # (2n,) 预测为正类的概率
        loss = bce_loss(p, y)

        error = p - y                          # (2n,) 概率减真值
        grad_w = X.T @ error / len(y)          # (2,)  梯度与线性回归同构
        grad_b = float(np.mean(error))

        w -= lr * grad_w
        b -= lr * grad_b
        history.append(loss)
        if epoch % 40 == 0 or epoch == 1:
            acc = float(np.mean((p >= 0.5) == y))
            print(f"epoch {epoch:3d} | loss = {loss:.4f} | acc = {acc:.3f}")

    # 混淆矩阵（演示计算方法；正经做法应在「测试集」上算）
    pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n混淆矩阵 TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={precision:.3f} recall={recall:.3f} "
          f"f1={2 * precision * recall / (precision + recall):.3f}")

    # 图 1：loss 下降曲线
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("Logistic regression training")
    plt.grid(True)
    plt.savefig("L04_loss_curve.png", dpi=150)

    # 图 2：决策边界——网格逐点预测概率并着色
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])   # (200*200, 2) 批量预测
    zz = sigmoid(grid @ w + b).reshape(xx.shape)       # 每个网格点的正类概率

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)   # 概率热力图
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c="blue", label="class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c="red", label="class 1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision boundary (p >= 0.5)")
    plt.legend()
    plt.savefig("L04_decision_boundary.png", dpi=150)
    plt.show()
    print("图已保存：L04_loss_curve.png / L04_decision_boundary.png")


if __name__ == "__main__":
    main()
