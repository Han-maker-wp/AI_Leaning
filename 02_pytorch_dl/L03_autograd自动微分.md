# 第 3 课：autograd 自动微分

> 🎬 开场钩子：2024 年，诺贝尔物理学奖颁给了一位从不自称物理学家的研究者——Hinton（与提出联想记忆网络的 Hopfield 同获）。评奖委员会表彰的这条研究线，起点之一正是 1986 年他与 Rumelhart、Williams 发表在 Nature 的反向传播论文。今天的主角 `backward()`，流淌着那篇论文的血液。

⏱ 建议学时：2 小时

## 本课目标

1. 理解计算图：前向传播时动态构建、反向传播按链式法则逐步回传；
2. 会用 `requires_grad` / `backward()` / `.grad` 三件套；
3. 能读懂 `grad_fn`，知道 `.grad` 是累积的（这就是训练循环必须 zero_grad 的原因）；
4. 知道 `no_grad()` / `detach()` 各自的适用场景；
5. 用 autograd 复现前置课程手写的线性回归。

## 3.1 回顾：你曾经手写梯度

机器学习课里你写过（numpy）：

```
y_hat = X @ w + b
loss  = mean((y_hat - y)^2)
# 手推：dL/dw = 2/N * X^T(y_hat - y)；dL/db = 2/N * sum(y_hat - y)
w -= lr * grad_w
b -= lr * grad_b
```

两个参数要手推 2 个公式；换成 2 层 MLP 就是几十万个参数——手推不现实。而所有这些导数本质上只是**链式法则的机械应用**，正是计算机最擅长的事。autograd 的承诺：**你只写前向计算（怎么做预测），反向的梯度它自动算**。从此精力从「求导」解放到「设计模型」。

## 3.2 计算图：autograd 的数据结构

前向传播时，PyTorch 把每步运算记录成一张有向无环图（DAG）：节点是 tensor，边是运算。叶子节点是你创建的输入/参数，中间节点是每步结果。`backward()` 从输出出发**沿图反向**，对每个运算应用链式法则，把梯度累积到各节点的 `.grad`。

文字版计算图（本课贯穿例子）：`y = (x * 2)^3`，取 `x = 3`：

```
前向（从左到右）：    x ──[×2]──► a ──[pow 3]──► b ──(恒等)──► y
                     3           6              216          216

反向（从右到左）：    x ◀────────a ◀─────────────b ◀─────────y
局部导数：            da/dx=2     db/da=3a²=108   dy/db=1    起步 dy/dy=1
```

反向回传全过程（「反向传播」四个字的字面意思）：

| 步骤 | 节点 | 收到的上游梯度 | 局部导数 | 写入本节点 / 回传量 |
|---|---|---|---|---|
| 0 | y | 约定 dL/dL = 1.0 | — | y.grad = 1.0 |
| 1 | b = a³ | y.grad = 1.0 | db/da = 3a² = 108 | b.grad = 1.0；回传给 a：1.0 × 108 = 108 |
| 2 | a = 2x | b 回传 108 | da/dx = 2 | a.grad = 108；回传给 x：108 × 2 = 216 |
| 3 | x | a 回传 216 | — | **x.grad = 216** |

手算验证：`y = (2x)³ = 8x³`，`dy/dx = 24x²`，在 x=3 处 = 216。一致。

核心思想一句话：**每个运算只需知道自己的局部导数**（乘法、幂、tanh 各有简单公式），全局梯度由链式法则沿图自动组装。L09 你会亲手实现这套机制，届时这张表会逐行变成代码。

> 🍿 彩蛋｜这就是 AOP：autograd 的玩法 Java 老兵秒懂——你只写业务逻辑（forward），框架像动态代理一样在每个运算前后织入记账与求导（backward）。Spring AOP 织入的是日志与事务，PyTorch 织入的是链式法则。

## 3.3 三件套：requires_grad / backward() / .grad

```python
import torch

if __name__ == "__main__":
    x = torch.tensor(3.0, requires_grad=True)   # 叶子节点：声明「要对它求导」
    a = x * 2        # 中间节点：autograd 自动记录这个运算
    b = a ** 3
    y = b            # 输出

    y.backward()     # 反向传播：从 y 出发沿图回传

    print(x.grad)    # tensor(216.) —— 与手算一致
```

规则要点：

- `requires_grad=True` 只需加在**需要梯度的叶子**（参数）上；由它参与的运算生成的中间 tensor 自动可导；
- `backward()` 只能在**标量输出**上调用（损失恰好是标量）；向量输出要先汇总，如 `y.sum().backward()`；
- `.grad` 存的是「当前已累积的梯度」，**不清零会一直加**：

```python
    x = torch.tensor(3.0, requires_grad=True)
    for _ in range(3):
        y = x * 2
        y.backward()
    print(x.grad)    # tensor(6.) —— 三次 backward 的 2+2+2 累加！
```

这就是训练循环里每步都要 `optimizer.zero_grad()` 的根本原因：新一轮前向的梯度应从 0 开始，否则梯度跨 batch 累积。（有一种「梯度累积」技巧故意利用它模拟大 batch，L08 提到。）

## 3.4 grad_fn：图上的面包屑

每个由运算产生的 tensor 都带 `grad_fn`，指向「产生它的运算」的反向实现。打印这条链就是反向传播要走的路：

```python
    x = torch.tensor(3.0, requires_grad=True)
    y = (x * 2) ** 3
    print(y.grad_fn)                  # <PowBackward0 ...>
    print(y.grad_fn.next_functions)   # ((<MulBackward0 ...>, 0),) —— 指向上一步
```

不需要读懂这些对象内部，建立映射感即可：看到 `MulBackward0` / `AddBackward0` / `TanhBackward0` 就知道图里有哪些运算；`grad_fn is None` 说明该 tensor 不在图里（requires_grad=False，或在 no_grad 里算的）。

> 🍿 彩蛋｜面包屑比童话靠谱：小节标题里的「面包屑」典出童话——进森林的孩子沿路撒面包屑，好循着回家。顺带一提，童话里那对孩子的面包屑被鸟吃了，计划失败；PyTorch 的 grad_fn 面包屑可靠得多，一根都不会丢。

## 3.5 no_grad() 与 detach()：什么时候「不要图」

计算图有成本：前向时要记录元数据，反向前中间结果都得留着。**不需要梯度的场景就该关掉**，省内存且更快：

- `with torch.no_grad():` —— 块内所有运算不建图。典型场景：**评估/推理**（验证集算准确率、线上预测）。
- `.detach()` —— 返回一个「从图上摘下来」的新 tensor（共享数据、无 grad_fn、不会回传）。典型场景：把中间结果当**纯数据**用（打印、画图、喂给另一个不走梯度的流程）。

```python
    x = torch.tensor(3.0, requires_grad=True)
    with torch.no_grad():
        y1 = x * 2
    print(y1.requires_grad)    # False

    y2 = (x * 2).detach()
    print(y2.requires_grad)    # False
```

区别一句话：`no_grad` 是「这一段都不建图」的上下文管理器；`detach` 是「把这一个 tensor 摘下来」的方法。

## 3.6 用 autograd 复现前置课程的线性回归

对照你手写过的 numpy 版：数据、模型、损失完全一样，唯一变化是**梯度不再手推**。完整可运行（对应 `code/L03_autograd_demo.py` 的最后一节）：

```python
import torch

if __name__ == "__main__":
    torch.manual_seed(0)

    # 造数据：y = 2x + 1 + 噪声
    x = torch.rand(100, 1)
    y = 2 * x + 1 + 0.01 * torch.randn(100, 1)

    # 参数是需要求导的叶子，初始化为 0
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.5

    for step in range(100):
        y_hat = x @ w + b                  # 前向（自动建图）
        loss = ((y_hat - y) ** 2).mean()   # MSE
        loss.backward()                    # 反向：w.grad / b.grad 自动填好

        with torch.no_grad():              # 手动更新参数：更新本身不能进图
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()                     # 清零累积的梯度（关键！）
        b.grad.zero_()

        if step % 20 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    print(f"w = {w.item():.4f}（真值 2）, b = {b.item():.4f}（真值 1）")
```

三个必问的「为什么」：

1. **为什么更新要在 no_grad 里？** `w -= ...` 是对参数的运算；若不在 no_grad 里，会被记进计算图，图越滚越大且语义错误（梯度流进了更新操作本身）。下一课这步交给 `optimizer.step()`，就再也不用手写了。
2. **为什么 `w.grad.zero_()` 可以不在 no_grad 里？** 它修改的是 `.grad`（一个普通 tensor），不是被追踪的前向运算。
3. **为什么打印 loss 要 `.item()`？** 直接持有 tensor 会连着计算图引用；`.item()` 取纯数值，让图可以被释放。

## 练习

1. 手推 `y = (3x + 1)²` 在 x=2 处的梯度，并用 autograd 验证。
   提示：展开 `9x² + 6x + 1` 求导后代入；答案是 42。
2. 构造一个被使用两次的节点（如 `y = x * x`），backward 后观察 x.grad。
   提示：两条路径的梯度会相加，d(x²)/dx = 2x。
3. 把 3.3 的累积示例改成「三次独立正确的求导」，确认每次 x.grad 都是 2。
   提示：每次 backward 前 `x.grad.zero_()`（或置 `x.grad = None`）。
4. 把 3.6 的学习率改成 0.1 与 5.0 各跑一次，观察 loss 行为差异。
   提示：太小收敛慢、太大震荡甚至发散——为 L07 调参做铺垫。

## 小结与自测

1. autograd 解决了什么问题？
   答：只写前向，反向梯度由链式法则沿计算图自动求出，替代手推。
2. 为什么每个 batch 都要 zero_grad？
   答：`.grad` 是累积的，不清零会把上一轮梯度加进这一轮。
3. no_grad 与 detach 的区别与场景？
   答：no_grad 关闭整段建图（评估循环）；detach 摘下单个 tensor 当纯数据用。
4. grad_fn 是什么？
   答：指向产生该 tensor 的运算之反向实现的引用，串成反向传播路径。
5. 反向传播的起点梯度是多少？
   答：1（dL/dL = 1），从输出往输入逐节点乘局部导数并求和。

## 下一课预告

上一节你还在手动管理 `w`、`b` 并手写更新。第 4 课引入 `nn.Module`：参数的声明、收集、更新全部交给框架——用你熟悉的 Java OOP 视角看，它是一套设计精良的接口与模板。
