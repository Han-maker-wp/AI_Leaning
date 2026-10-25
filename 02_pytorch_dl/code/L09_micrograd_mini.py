"""L09：micrograd 迷你自动微分引擎。

Karpathy "Neural Networks: Zero to Hero" 第 1 集（micrograd）思路的精简实现：
纯 Python 标量运算实现 Value 类（data/grad/_backward）+ 拓扑排序反向传播，
并用它搭一个 3-4-4-1 的 MLP 完成训练——揭开 PyTorch backward() 的黑盒。

运行：python L09_micrograd_mini.py
对应讲义：02_pytorch_dl/L09_micrograd从零理解反向传播.md
"""
import math
import random


class Value:
    """计算图上的一个节点：标量 + 梯度 + 局部反向函数 + 父节点集合。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---------- 基本运算：每个 op 造输出节点 + 注册输出的反向函数 ----------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad            # d(a+b)/da = 1
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad     # d(ab)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "只支持常数次幂"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad   # d(a^k)/da = k a^(k-1)
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad  # d tanh/da = 1 - tanh^2
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---------- 由基本运算组合出的便捷运算 ----------

    def __neg__(self):              # -a
        return self * -1

    def __sub__(self, other):       # a - b
        return self + (-other)

    def __radd__(self, other):      # 1 + a
        return Value(other) + self

    def __rsub__(self, other):      # 1 - a
        return Value(other) + (-self)

    def __rmul__(self, other):      # 2 * a
        return Value(other) * self

    def __truediv__(self, other):   # a / b = a * b^-1
        if isinstance(other, (int, float)):
            other = Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):  # 2 / a
        return Value(other) / self

    # ---------- 反向传播：拓扑排序 + 逆序调用 ----------

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)      # 后序：所有父节点排在我前面
        build(self)

        self.grad = 1.0             # 起步种子 dL/dL = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ---------- 用 Value 搭神经网络：神经元 -> 层 -> MLP ----------

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), Value(0.0)) + self.b
        return z.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + list(n_outs)
        self.layers = [Layer(a, b) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


def main():
    random.seed(42)

    # ---------- 1. 验证 L03 的计算图例子：y=(x*2)^3 ----------
    x = Value(3.0)
    y = (x * 2) ** 3
    y.backward()
    print(f"(x*2)^3 在 x=3: y={y.data}, dy/dx={x.grad}（手算/PyTorch 均为 216）")

    # 多路径求和验证：y = x + x，dy/dx 应为 2（+= 累积的功劳）
    x2 = Value(1.0)
    z = x2 + x2
    z.backward()
    print(f"x + x 在 x=1: dz/dx={x2.grad}（期望 2：两条路径梯度相加）")

    # ---------- 2. 训练一个 3-4-4-1 的 MLP ----------
    model = MLP(3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0],
          [3.0, -1.0, 0.5],
          [0.5, 1.0, 1.0],
          [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    print("参数个数:", len(model.parameters()))

    for step in range(100):
        # 前向 + MSE 损失
        y_pred = [model(x) for x in xs]
        loss = sum(((yp - yt) ** 2 for yp, yt in zip(y_pred, ys)), Value(0.0))

        # 清零（对应 optimizer.zero_grad）
        for p in model.parameters():
            p.grad = 0.0
        # 反向（对应 loss.backward）
        loss.backward()
        # 更新（对应 optimizer.step）
        for p in model.parameters():
            p.data -= 0.05 * p.grad

        if step % 10 == 0:
            print(f"step {step:3d}  loss={loss.data:.6f}")

    print("最终预测:", [round(model(x).data, 3) for x in xs])
    print("目标    :", ys)


if __name__ == "__main__":
    main()
