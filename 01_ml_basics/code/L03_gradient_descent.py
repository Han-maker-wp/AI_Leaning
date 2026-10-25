"""L03：纯 numpy 手写梯度下降训练线性回归——未来所有训练循环的骨架。

运行前：uv pip install numpy matplotlib
运行：python3 code/L03_gradient_descent.py
输出：L03_loss_curve.png
"""
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # ========== 第 0 步：准备数据 y = 2x + 1 + 噪声 ==========
    rng = np.random.default_rng(0)
    n = 200
    X = rng.uniform(0, 5, size=(n, 1))                      # (n,1)
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.6, size=n)    # (n,)

    # ========== 第 1 步：初始化参数（一切从这里开始） ==========
    w = np.zeros(1)          # 也可以随机小值：rng.normal(0, 0.01, size=1)
    b = np.zeros(1)
    lr = 0.05                # 学习率 η：改大改小观察现象（练习 1）
    epochs = 100             # 轮数

    history = []             # 记录每轮 loss，用于画下降曲线

    # ========== 第 2 步：训练循环（背下这个结构！） ==========
    #   前向 → 损失 → 梯度 → 更新 → 记录
    #   对应 PyTorch：model(X) → loss_fn → loss.backward() → optimizer.step() → 日志
    for epoch in range(1, epochs + 1):
        # --- 前向传播：用当前参数算预测 ---
        y_pred = X[:, 0] * w[0] + b[0]                      # (n,)

        # --- 计算损失（监控训练进度用） ---
        loss = float(np.mean((y_pred - y) ** 2))

        # --- 反向传播：手写解析梯度 ---
        error = y_pred - y                                  # (n,) 残差 ŷ - y
        grad_w = 2.0 * np.mean(error * X[:, 0])             # ∂L/∂w
        grad_b = 2.0 * np.mean(error)                       # ∂L/∂b

        # --- 参数更新：沿负梯度走一小步 ---
        w[0] -= lr * grad_w
        b[0] -= lr * grad_b

        history.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss = {loss:.4f} | "
                  f"w = {w[0]:.4f}, b = {b[0]:.4f}")

    print(f"\n训练结果：w = {w[0]:.4f}, b = {b[0]:.4f}（真值 2.0 与 1.0）")

    # ========== 第 3 步：画 loss 下降曲线 ==========
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), history)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Gradient descent on linear regression")
    plt.grid(True)
    plt.savefig("L03_loss_curve.png", dpi=150)
    plt.show()
    print("loss 曲线已保存为 L03_loss_curve.png")


if __name__ == "__main__":
    main()
