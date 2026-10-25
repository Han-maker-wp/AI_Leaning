# 第 9 课：micrograd——从零理解反向传播

> 🎬 开场钩子：一个悬念——给 PyTorch 施魔法的自动微分引擎，核心思想用纯 Python 写下来只需约一百行。今天你不调用 `torch.autograd`，而是亲手把 `backward()` 造出来。写完你会发现：九课里最「黑盒」的东西，其实是最诚实的那一个——每一步都是你已经会的数学。

⏱ 建议学时：3 小时（含跟随 Karpathy 视频敲代码）

## 本课目标

1. 实现 Value 类：data（前向值）/ grad（梯度）/_backward（局部反向函数）；
2. 实现 add / mul / pow / tanh 四个运算及其局部导数；
3. 用拓扑排序组织反向传播顺序，说清「为什么必须拓扑排序」；
4. 用 Value 搭出 2 隐层 MLP 并用梯度下降训练它；
5. 从此 `loss.backward()` 对你不再是黑盒。

## 9.1 这一课在做什么

配套视频：Karpathy "Neural Networks: Zero to Hero" 第 1 集（YouTube，有搬运）。本课是其思路的精简中文实现，完整可运行版在 `code/L09_micrograd_mini.py`。

PyTorch 的 autograd 引擎工业级复杂（性能、算子覆盖、分布式），但**核心思想只有约 100 行**。你将用纯 Python 标量运算写出一个能训练神经网络的迷你自动微分引擎 micrograd。写完后，L03 的计算图、.grad 累积、链式法则回传——每一件事你都有了代码级理解。

## 9.2 Value 类：图上的节点

每个 Value 节点保存四样东西：

```python
class Value:
    """计算图上的一个节点：一个标量 + 它的梯度 + 怎么反向传播。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data               # 前向计算的数值
        self.grad = 0.0                # dL/dself，反向传播时填充
        self._backward = lambda: None  # 「把 out.grad 按局部导数分给父节点」的函数
        self._prev = set(_children)    # 产生本节点的父节点（图的边）
        self._op = _op                 # 运算名（调试用）
```

设计的精妙在 `_backward` 闭包：**每个运算在创建输出节点时，顺便把「这个运算的反向逻辑」注册到输出节点上**。反向传播 = 逆着图依次调用每个节点的 `_backward`。

> 🍿 彩蛋｜回调注册的老套路：`_backward` 闭包的玩法，就是 Swing 的 `addActionListener`——创建按钮（输出节点）时顺手注册监听器（反向逻辑），注册时不执行，事件来了（backward 被调用）才跑。观察者模式，二十年没变过。

## 9.3 四个运算与局部导数

```python
def __add__(self, other):                       # z = a + b
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
        self.grad += 1.0 * out.grad             # d(a+b)/da = 1
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

def __mul__(self, other):                       # z = a * b
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad      # d(ab)/da = b
        other.grad += self.data * out.grad
    out._backward = _backward
    return out

def __pow__(self, k):                           # z = a**k（k 是常数）
    out = Value(self.data ** k, (self,), f"**{k}")

    def _backward():
        self.grad += k * self.data ** (k - 1) * out.grad   # d(a^k)/da = k·a^(k-1)
    out._backward = _backward
    return out

def tanh(self):                                 # z = tanh(a)
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1.0 - t * t) * out.grad   # d tanh/da = 1 - tanh²
    out._backward = _backward
    return out
```

对照 L03 的反向表格：每个 `_backward` 就是表里那列「局部导数 × out.grad」。

**为什么用 `+=` 而不是 `=`**：一个节点可能被使用多次（`y = x * x` 里 x 同时是乘法两侧输入；或 `z = x + x`），它会从多条路径收到梯度，链式法则要求**求和**。这就是 PyTorch 里 `.grad`「累积」语义的出处——不是设计怪癖，是数学要求。它同时解释了为什么每轮训练要 zero_grad：上一轮的求和结果不能带进这一轮。

（工程补全：`__neg__` / `__sub__` / `__truediv__` / `__radd__` / `__rmul__` / `__repr__` 都由这四个基本运算组合出来，例如 `a - b = a + (-1)*b`、`a / b = a * b**-1`。完整版见代码文件。）

## 9.4 backward()：拓扑排序 + 逆序调用

```python
def backward(self):
    # 1) 深度优先遍历，得到拓扑序（父节点总排在子节点前面）
    topo, visited = [], set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)          # 后序插入：我的所有父节点都排在我前面
    build(self)

    # 2) 从输出（self）开始，逆拓扑序逐个调用 _backward
    self.grad = 1.0                 # 起步种子：dL/dL = 1
    for v in reversed(topo):
        v._backward()
```

两个「为什么」：

- **为什么要拓扑排序**：调用某节点的 `_backward` 时，它的 `out.grad` 必须已经**收齐**了所有下游路径传来的梯度。拓扑序保证先填好下游（父）节点、再算上游（子）节点。（实现上是「后序遍历的逆序」，效果即反拓扑序。）
- **为什么 self.grad = 1**：dL/dL = 1 是链式法则的起点（L03 讲过）。

跑通 L03 的例子三方对账（手算 = PyTorch = micrograd）：

```python
if __name__ == "__main__":
    x = Value(3.0)
    y = ((x * 2) ** 3)              # (3*2)^3 = 216
    y.backward()
    print(x.grad)                   # 216.0
```

> 🍿 彩蛋｜拓扑排序你早就在用：Maven/Gradle 解析依赖、决定哪个模块先编译，走的就是拓扑排序——被依赖的先构建。micrograd 的反向传播同理：先收齐下游节点的梯度，才轮到上游。工具链里的老朋友，换了个部门上班。

## 9.5 用 Value 搭神经网络

神经元 = 权重乘输入求和 + 偏置，过 tanh；层 = 一排神经元；MLP = 层的堆叠：

```python
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
    def __init__(self, n_in, n_outs):            # n_outs：各层宽度
        sizes = [n_in] + list(n_outs)
        self.layers = [Layer(a, b) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
```

这就是 `nn.Module.parameters()` 递归收集的手写版——现在你知道它在收集什么：所有作为参数的 Value 节点。

## 9.6 训练它（依然是八股文）

```python
if __name__ == "__main__":
    random.seed(42)
    model = MLP(3, [4, 4, 1])                     # 3 输入 -> 4 -> 4 -> 1 输出

    xs = [[2.0, 3.0, -1.0],
          [3.0, -1.0, 0.5],
          [0.5, 1.0, 1.0],
          [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]                  # 4 个样本的目标

    for step in range(100):
        # 前向：逐样本预测，MSE 汇总成标量 loss
        y_pred = [model(x) for x in xs]
        loss = sum(((yp - yt) ** 2 for yp, yt in zip(y_pred, ys)), Value(0.0))

        # 反向前清零（对应 zero_grad！）
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()                          # 一次填好所有参数的 grad

        # 更新（对应 optimizer.step()）
        for p in model.parameters():
            p.data -= 0.05 * p.grad

        if step % 10 == 0:
            print(step, loss.data)               # 一路下降到 ~0.001
```

注意：八股文一个不少——清梯度、前向、损失、反向、更新，只是每个环节都由你亲手写。loss 降到接近 0，说明这个约 100 行的引擎真的「学会了」这 4 个样本。

## 9.7 与 PyTorch 的对照

| micrograd | PyTorch | 差别 |
|---|---|---|
| Value 标量节点 | tensor（整个矩阵一个节点） | PyTorch 算子是张量级，一次传递成千上万个标量 |
| `_backward` 闭包 | 每个运算的 C++/CUDA 反向核 | 机制相同 |
| 手写拓扑遍历 | autograd 引擎自动做 | 机制相同 |
| `p.data -= lr * p.grad` | `optimizer.step()` | 机制相同 |
| `p.grad = 0.0` | `optimizer.zero_grad()` | 机制相同 |

可选验证（强烈建议做）：用 `torch.tensor(..., requires_grad=True)` 复刻同一个 3-4-4-1 网络，同样输入下比较两边每个参数的梯度——应精确一致到小数点后多位。做过一次，「PyTorch 只是把这套东西规模化」的体感就建立了。

## 练习

1. 给 Value 加 `exp()` 与 `sigmoid()`（各约 5 行）。
   提示：局部导数分别是 exp(a) 与 σ(a)(1-σ(a))。
2. 把 9.6 的 tanh 换成你实现的 sigmoid，重训并比较前期收敛速度。
   提示：留意前 20 步 loss 下降快慢即可。
3. 用 torch 复刻 9.6 的网络（3-4-4-1），核对某个权重的梯度。
   提示：前向用 `@` 与 `torch.sigmoid`；MSE 用 `((pred - y)**2).mean()`。
4. 故意把某个 `_backward` 里的 `+=` 改成 `=`，用 `y = x + x` 复现梯度错误。
   提示：正确值 2.0，改后变 1.0。

## 小结与自测

1. Value 节点存哪四样东西？
   答：data、grad、_backward、_prev（父节点集合）。
2. 为什么梯度用 += 累积？
   答：节点被多条路径使用时，链式法则要求各路径梯度求和。
3. 反向传播为什么需要拓扑排序？
   答：保证调用某节点 _backward 前，其 out.grad 已收齐全部下游梯度。
4. PyTorch 的 zero_grad / step 对应 micrograd 哪两行？
   答：`p.grad = 0.0` / `p.data -= lr * p.grad`。
5. PyTorch 相对 micrograd 的增强是什么？
   答：张量化（矩阵级节点）、GPU 并行、工程优化；核心机制完全相同。

## 下一课预告

阶段 2 收官。你已具备：张量运算、autograd、nn.Module、数据管道、完整训练循环、调参与显存直觉、反向传播的内部原理。下一阶段进入 NLP 与 Transformer：分词与词向量 → RNN 概览 → Attention 精讲 → 论文精读 → HuggingFace → 第一次微调。
