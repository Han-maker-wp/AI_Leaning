"""L03：autograd 自动微分演示。

内容：
  1) 计算图示例 y=(x*2)^3 的反向传播（与手算 216 对账）
  2) .grad 的累积陷阱与正确清零
  3) no_grad / detach
  4) 用 autograd 复现前置课程手写的线性回归

运行：python L03_autograd_demo.py
对应讲义：02_pytorch_dl/L03_autograd自动微分.md
"""
import torch


def demo_graph():
    """y = (x*2)^3 在 x=3 处求导：手算 dy/dx = 24x^2 = 216。"""
    x = torch.tensor(3.0, requires_grad=True)
    a = x * 2          # 中间节点：a = 6
    b = a ** 3         # b = 216
    y = b              # 输出

    y.backward()       # 反向传播

    print(f"y = (x*2)^3, x=3 -> y = {y.item()}, x.grad = {x.grad.item()}（期望 216）")
    print("y.grad_fn:", y.grad_fn)                    # PowBackward0 链
    print("next:", y.grad_fn.next_functions)          # 指向上一步 MulBackward0


def demo_accumulation():
    """.grad 是累积的：不清零会跨 backward 相加。"""
    x = torch.tensor(3.0, requires_grad=True)
    for _ in range(3):
        y = x * 2
        y.backward()
    print(f"连续 3 次 backward 且不清零: x.grad = {x.grad.item()}（2+2+2=6，陷阱！）")

    # 正确姿势：每次 backward 前清零
    x = torch.tensor(3.0, requires_grad=True)
    for _ in range(3):
        if x.grad is not None:
            x.grad.zero_()
        y = x * 2
        y.backward()
    print(f"每次清零后: x.grad = {x.grad.item()}（正确：单次梯度 2）")


def demo_no_grad_detach():
    x = torch.tensor(3.0, requires_grad=True)

    with torch.no_grad():          # 整段不建图：评估/推理场景
        y1 = x * 2
    print("no_grad 里算出的 requires_grad:", y1.requires_grad)   # False

    y2 = (x * 2).detach()          # 摘下单个 tensor：当纯数据用
    print("detach 后 requires_grad:", y2.requires_grad)          # False


def demo_linear_regression():
    """用 autograd 复现手写线性回归：唯一区别是梯度不再手推。"""
    torch.manual_seed(0)

    # 造数据：y = 2x + 1 + 噪声
    x = torch.rand(100, 1)
    y = 2 * x + 1 + 0.01 * torch.randn(100, 1)

    w = torch.zeros(1, 1, requires_grad=True)   # 参数 = 需要求导的叶子
    b = torch.zeros(1, requires_grad=True)
    lr = 0.5

    for step in range(100):
        y_hat = x @ w + b                  # 前向（自动建图）
        loss = ((y_hat - y) ** 2).mean()   # MSE
        loss.backward()                    # 反向：w.grad / b.grad 自动填好

        with torch.no_grad():              # 更新本身不能进计算图
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()                     # 清零累积梯度（关键！）
        b.grad.zero_()

        if step % 20 == 0:
            print(f"  step {step:3d}  loss={loss.item():.4f}")

    print(f"  结果: w = {w.item():.4f}（真值 2）, b = {b.item():.4f}（真值 1）")


if __name__ == "__main__":
    print("== 1) 计算图与 backward ==")
    demo_graph()
    print("== 2) .grad 累积陷阱 ==")
    demo_accumulation()
    print("== 3) no_grad / detach ==")
    demo_no_grad_detach()
    print("== 4) autograd 线性回归 ==")
    demo_linear_regression()
