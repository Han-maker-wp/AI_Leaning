# 第 4 课：nn.Module 与 MLP

> 🎬 开场钩子：一个悬念——全世界的 PyTorch 模型几乎都继承自同一个类 `nn.Module`，却几乎没有人重写过 `backward()`。梯度怎么算、参数怎么被自动收集、模型怎么一键搬上 GPU？答案不是魔法，而是一套你熟到骨子里的 Java 设计：抽象基类 + 模板方法 + 自动注册。这节课把它拆开给你看。

⏱ 建议学时：2 小时

## 本课目标

1. 理解 `nn.Linear` 就是「会学习的 wx + b」，掌握其参数形状约定；
2. 会用 `nn.ReLU` / `nn.Sequential` 快速搭 MLP；
3. 能自定义 Module：`__init__` 里定义层，`forward` 里定义前向；
4. 从 Java OOP 视角说清 Module 继承设计的动机；
5. 用 `nn.Module` + optimizer 重写线性回归，删掉所有手动参数管理。

## 4.1 痛点：手动管理参数撑不住了

上一课的线性回归只有 2 个参数，手写 `w`、`b`、`grad.zero_()` 还能忍受。但 L01 算过：784→256→128→10 的 MLP 有 26 万个参数，手写会疯掉。需要的抽象是：

- **层**：封装「一组参数 + 一次前向运算」的可复用单元；
- **容器**：把层组装成模型，自动收集所有参数；
- **更新器**：拿着收集好的参数统一做梯度下降。

对应 PyTorch 的 `nn.Module` / `nn.Sequential` / `torch.optim.*`。

## 4.2 nn.Linear：会学习的 wx + b

```python
import torch
import torch.nn as nn

if __name__ == "__main__":
    layer = nn.Linear(in_features=4, out_features=2)   # 4 维输入 -> 2 维输出
    print(layer.weight.shape)    # [2, 4] —— 注意是 (out, in)，转置的形状！
    print(layer.bias.shape)      # [2]

    x = torch.randn(3, 4)        # 3 个样本、每个 4 维：形状约定 [batch, features]
    y = layer(x)                 # 等价于 x @ W.T + b
    print(y.shape)               # [3, 2]
```

与 L01 手推的对应：`nn.Linear` 的前向就是 `x @ W^T + b`（PyTorch 把权重存成 `[out, in]`，乘时转置，实现更高效）。`layer.weight`、`layer.bias` 都是 `requires_grad=True` 的 tensor——上一课的三件套它自带。

初始化：创建时参数自动随机初始化（默认 Kaiming 系方案，对 ReLU 网络友好）。能直接用；知道「初始化会影响训练难度」这件事即可，细节先不展开。

## 4.3 nn.ReLU 与 nn.Sequential

```python
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    print(model)
    out = model(torch.randn(5, 4))   # [5, 2]
```

`nn.Sequential` 把各层按顺序串联：上一层的输出接下一层的输入。`nn.ReLU()` 是「层」形式的激活（无参数的运算包装成 Module），好处是能统一塞进容器。第一个真正的 MLP（L01 结构的代码版）：

```python
    mlp = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),          # 输出层不加激活，原因见 L06（CrossEntropyLoss 的约定）
    )
```

## 4.4 自定义 Module

`Sequential` 只能表达「一条直路」。稍微复杂（复用子模块、多输入、分支）就要自定义：

```python
import torch
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()                        # 必须：把本类注册进基类机制
        self.fc1 = nn.Linear(in_dim, hidden_dim)  # __init__ 里「定义」带参数的层
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))                 # forward 里只写前向运算
        return self.fc2(h)

if __name__ == "__main__":
    model = TwoLayerMLP(784, 128, 10)
    logits = model(torch.randn(32, 784))          # 直接调用 model(x)
    print(logits.shape)                           # [32, 10]
```

两条铁律：

1. `__init__` 里**定义**带参数的层（赋值给 `self.xxx`）；不要在 `forward` 里新建——每次调用都会创建新参数，永远学不到东西；
2. 调用时用 `model(x)` 而不是 `model.forward(x)`——基类的 `__call__` 在调用 forward 前后挂着钩子机制，直接调 `.forward()` 会绕过它们。

无参数操作（`x.reshape`、`torch.tanh` 等）直接写在 forward 里即可，不必包成层。

## 4.5 为什么设计成类继承——Java 直觉

你写过 Java，这个设计一眼亲切。把 `nn.Module` 看成**抽象基类 + 生命周期托管**，类似继承 `AbstractList` 或实现某个框架的 SPI：

| 你来实现的（像 @Override） | 框架替你做的（像模板方法 / 容器） |
|---|---|
| `__init__`：声明结构（组装成员） | `__setattr__` 钩子：每当你赋值一个 `nn.Linear`，自动登记它的参数 |
| `forward`：声明前向计算 | `__call__`：统一入口，调 forward 并处理钩子 |
| | `parameters()`：递归收集所有已注册参数（含子 Module） |
| | `.to(device)`：递归把所有参数搬到 GPU |
| | `.train()` / `.eval()`：递归切换模式（dropout / batchnorm 行为） |
| | `state_dict()`：导出所有参数，供保存/加载 |

关键设计是**自动注册 + 递归组合**：Module 里可以嵌套 Module（`self.block = nn.Sequential(...)`），`parameters()` 会递归找齐——像 Spring 容器自动装配 Bean 依赖：你只声明结构，容器负责收集与生命周期。这套约定的回报：L07 一行 `torch.save(model.state_dict(), path)` 存整个模型；L08 一行 `model.to("cuda")` 把任意复杂的模型整体搬上 GPU。前提只有一个：**层必须在 `__init__` 里赋值给 self**。

> 🍿 彩蛋｜组合模式的老朋友：Module 套 Module 长出的参数树，正是 GoF 组合模式——和 Swing 的 Component/Container 一脉相承：JPanel 里放 JButton，Module 里放 Module，容器递归地「认识」每个孩子，`parameters()` 才能一网打尽。

## 4.6 parameters() 与 state_dict()

```python
    model = TwoLayerMLP(784, 128, 10)
    for name, p in model.named_parameters():
        print(name, tuple(p.shape), p.numel())
    # fc1.weight (128, 784) 100352
    # fc1.bias   (128,)      128
    # fc2.weight (10, 128)   1280
    # fc2.bias   (10,)       10
    total = sum(p.numel() for p in model.parameters())
    print(total)   # 101770
```

`parameters()` 返回生成器，`state_dict()` 返回「名字 → tensor」的有序字典（L07 保存 checkpoint 用的就是它）。数参数量这个动作后面天天做：显存估算（L08）、模型选型、过拟合判断（参数量 vs 数据量）都靠它。

## 4.7 用 nn.Module 重写线性回归

对照 L03 的版本：`w`、`b` 的手动管理消失了，更新交给 optimizer。这是「框架级」训练脚本的雏形，L06 把它扩成完整训练循环：

```python
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(100, 1)
    y = 2 * x + 1 + 0.01 * torch.randn(100, 1)

    model = LinearRegression()
    loss_fn = nn.MSELoss()                                   # 损失也用现成的
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)   # 优化器接管更新

    for step in range(100):
        optimizer.zero_grad()          # 1. 清零所有参数的梯度（对比 L03 手写两个 zero_）
        loss = loss_fn(model(x), y)    # 2. 前向 + 算损失
        loss.backward()                # 3. 反向
        optimizer.step()               # 4. 更新（对比 L03 的 no_grad 手动减）

        if step % 20 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    w = model.linear.weight.item()
    b = model.linear.bias.item()
    print(f"w = {w:.4f}（真值 2）, b = {b:.4f}（真值 1）")
```

四步 `zero_grad → forward → backward → step` 就是训练循环的心脏。L06 把它套进 DataLoader 的双层循环里，成为你要背下来的「八股文」。

## 练习

1. 用 `nn.Sequential` 搭 784→512→256→10 的 MLP，打印总参数量。
   提示：`sum(p.numel() for p in model.parameters())`。
2. 故意把 `nn.Linear` 写在 `forward` 里，训练若干步后检查 `model.parameters()` 是否为空。
   提示：未注册的参数不会被收集，optimizer 无参数可更新。
3. 给 `TwoLayerMLP` 加参数 `n_hidden_layers`，支持任意层数。
   提示：`__init__` 里用循环往 `nn.ModuleList` 添加层（查文档 `nn.ModuleList`）。
4. 打印线性回归模型的 `state_dict()`，观察键名与 L03 手写 w/b 的对应。
   提示：键名是 `linear.weight`、`linear.bias`。

## 小结与自测

1. nn.Linear 的 weight 形状与前向计算式？
   答：`[out, in]`，前向 `x @ W^T + b`。
2. `model(x)` 与 `model.forward(x)` 的区别？
   答：前者走基类 `__call__`（带钩子）；直接调 forward 绕过机制，应避免。
3. Module 靠什么自动收集参数？
   答：`__setattr__` 钩子登记 `__init__` 中赋值的层；`parameters()` 递归收集。
4. 为什么层不能在 forward 里创建？
   答：每次调用生成新参数，既不注册也无法学习。
5. `optimizer.step()` 替代了 L03 的哪段代码？
   答：no_grad 里的手动 `w -= lr * w.grad`（对所有参数的循环）。

## 下一课预告

模型有了，下一课解决「数据怎么喂」：Dataset 与 DataLoader。为什么不能一次全喂、batch 大小怎么影响训练、以及贯穿后续所有代码的形状约定 `[batch, features]`。
