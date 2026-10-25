"""L07：matplotlib 快览——折线图（loss 曲线雏形）与散点图（看数据）。

运行前：uv pip install numpy matplotlib
运行：python3 code/L07_matplotlib_demo.py
输出：L07_loss_curve.png 与 L07_scatter.png 两张图片
"""
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    rng = np.random.default_rng(0)

    # ---------- 图 1：折线图（之后每课画 loss 曲线都是这个套路） ----------
    epochs = np.arange(1, 11)
    train_loss = np.exp(-0.3 * epochs)                 # 人为造一条指数下降曲线
    val_loss = np.exp(-0.3 * epochs) + 0.05 + 0.01 * epochs   # 验证集稍高

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o", label="train")
    plt.plot(epochs, val_loss, marker="s", label="val")
    plt.xlabel("epoch")            # 注：默认字体不支持中文，图内文字用英文
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("L07_loss_curve.png", dpi=150)   # 保存成文件（无界面环境也 OK）
    plt.show()                                   # 有图形界面时弹窗显示

    # ---------- 图 2：散点图（拿到数据后的第一个动作） ----------
    x = rng.uniform(0, 10, size=80)
    y = 0.8 * x + 1.0 + rng.normal(0, 1.0, size=80)    # 带噪声的线性关系
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=15, alpha=0.7, label="samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter with noise")
    plt.legend()
    plt.savefig("L07_scatter.png", dpi=150)
    plt.show()
    print("已保存：L07_loss_curve.png、L07_scatter.png")


if __name__ == "__main__":
    main()
