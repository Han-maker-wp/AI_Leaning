"""L05：多项式拟合阶数对比——直观看见过拟合（train MSE 降、test MSE 升）。

运行前：uv pip install numpy matplotlib
运行：python3 code/L05_overfitting_demo.py
输出：L05_overfitting.png
"""
import numpy as np
import matplotlib.pyplot as plt


def true_fn(x: np.ndarray) -> np.ndarray:
    """真实规律：一条 sin 曲线（模型事先不知道，只能从 15 个带噪点里学）。"""
    return np.sin(2.0 * x)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def main() -> None:
    # 1. 少样本（15 个）+ 噪声：过拟合的温床
    rng = np.random.default_rng(7)
    n_train = 15
    x_train = rng.uniform(0, 3, size=n_train)
    y_train = true_fn(x_train) + rng.normal(0, 0.25, size=n_train)
    x_test = np.sort(rng.uniform(0, 3, size=100))
    y_test = true_fn(x_test)                  # 测试目标用无噪声真值，检验「学到规律」

    degrees = [1, 3, 14]                      # 欠拟合 / 恰当 / 过拟合 三连
    print(f"{'degree':>6} | {'train MSE':>10} | {'test MSE':>10}")
    print("-" * 36)

    plt.figure(figsize=(14, 4))
    xs = np.linspace(0, 3, 300)

    for i, deg in enumerate(degrees, start=1):
        coeffs = np.polyfit(x_train, y_train, deg)   # 最小二乘拟合
        poly = np.poly1d(coeffs)                     # 系数变成可调用的函数

        train_mse = mean_squared_error(y_train, poly(x_train))
        test_mse = mean_squared_error(y_test, poly(x_test))
        print(f"{deg:>6} | {train_mse:>10.4f} | {test_mse:>10.4f}")

        plt.subplot(1, len(degrees), i)
        plt.scatter(x_train, y_train, s=20, label="train data")
        plt.plot(xs, true_fn(xs), "g--", lw=1.5, label="true f(x)")
        plt.plot(xs, poly(xs), "r-", lw=2, label=f"degree {deg} fit")
        plt.title(f"deg={deg}  test MSE={test_mse:.3f}")
        plt.ylim(-1.6, 1.6)
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("L05_overfitting.png", dpi=150)
    plt.show()
    print("对比图已保存为 L05_overfitting.png")
    print("结论：阶数 1 欠拟合（两条误差都高）；阶数 3 贴近真规律；")
    print("      阶数 14 穿过每个训练点（train MSE≈0）却在测试集彻底失控。")


if __name__ == "__main__":
    main()
