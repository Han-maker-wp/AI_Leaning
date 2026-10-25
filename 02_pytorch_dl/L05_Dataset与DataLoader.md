# 第 5 课：Dataset 与 DataLoader

> 🎬 开场钩子：一个悬念——只要你实现两个方法，PyTorch 就能替你打乱数据、切批次、多进程并行加载，背后是内存数组、CSV 还是数据库，它一概不问。这份契约小得惊人，小得像 Java 世界里一个只有两个方法的接口。今天看看这份「小契约」如何撑起所有深度学习的数据管道。

⏱ 建议学时：1.5 小时

## 本课目标

1. 说清为什么分批训练（显存约束 + 噪声有益 + 硬件效率）；
2. 会用 TensorDataset + DataLoader 跑通最小批训练；
3. 会写自定义 Dataset（`__len__` + `__getitem__` 两个方法）；
4. 掌握 DataLoader 常用参数：batch_size / shuffle / drop_last / num_workers；
5. 养成「形状思维」：拿到任何 batch 先看 `.shape`。

## 5.1 为什么要分批

前置课程里你对小数据做过全量梯度下降（每步用全部样本算一次梯度）。深度学习默认改用 mini-batch（小批量），三个原因：

1. **内存/显存放不下**：100 万样本 × 784 维 float32 ≈ 3GB，全量前向还要存中间激活——放不进显存。分批后每次只放 batch_size 份。
2. **随机噪声反而有益**：小批量梯度是全量梯度的无偏但含噪估计。噪声帮优化器跳出尖锐局部极小与鞍点，实测泛化往往更好（全量梯度太「确定」，容易停在较差的解）。
3. **硬件效率**：矩阵乘在 batch 维度上几乎白送（GPU 并行），batch=1 与 batch=64 的耗时常常接近——单位样本成本大幅下降。

代价与取舍：batch 越小噪声越大、训练曲线越抖；越大越稳但显存越多、每 epoch 更新次数越少（学习率通常要配合调大，L07 讲）。常用起点 32/64；MNIST 这类任务 CPU 也轻松。

> 🍿 彩蛋｜行业黑话 epoch：把全部训练数据完整过一遍叫一个 epoch（纪元）——好比把整个代码库 code review 了一遍；batch 则是每次送审的一个文件。以后在论文里看到「trained for N epochs」，脑子里放这个画面就够了。

## 5.2 TensorDataset：最简 Dataset

`Dataset` 是个抽象协议：能回答「一共多少条」（`__len__`）与「取第 i 条」（`__getitem__`）。`TensorDataset` 把已有的 tensor 打包成这个协议，零成本：

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    X = torch.randn(100, 4)      # 100 个样本、4 个特征
    y = torch.randn(100, 1)

    ds = TensorDataset(X, y)     # 要求各 tensor 第 0 维一致（样本数相同）
    print(len(ds))               # 100
    xi, yi = ds[0]               # 取单条：一个 (x, y) 元组
    print(xi.shape, yi.shape)    # torch.Size([4]) torch.Size([1])
```

## 5.3 自定义 Dataset：像实现一个 Java 接口

Java 直觉：`Dataset` 就是个接口，规定两个方法——类似实现 `List<E>` 要实现 `size()` 与 `get(int)`。框架（DataLoader）只通过这两个方法访问数据，完全不管你背后是内存数组、CSV 还是数据库：

```python
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
```

真实数据（图片路径、文本行、数据库行）几乎都走这条路。第 3 阶段微调 BERT 时你会再写一次——`__getitem__` 返回原始文本，tokenize 放到批级别（collate）进行。

> 🍿 彩蛋｜隐式的 implements：Python 这套「协议」是隐式的——不用写 implements，只要 `__len__` / `__getitem__` 的签名对了，DataLoader 就认你是 Dataset（鸭子类型）。Java 要显式声明接口，Python 只看你「像不像」；这也是 `len(ds)`、`ds[0]` 能直接工作的原因。

## 5.4 DataLoader：分批、打乱、多进程

```python
    loader = DataLoader(
        ds,
        batch_size=16,     # 每批 16 条
        shuffle=True,      # 每个 epoch 重新打乱（训练集必须 True！）
        drop_last=False,   # 最后不足一批的是否丢弃（评估建议 False）
        num_workers=0,     # 0 = 主进程加载；Linux 下数据预处理重时可设 2~4
    )

    for xb, yb in loader:  # 每次迭代吐出一个 batch 的 (x, y)
        print(xb.shape, yb.shape)   # torch.Size([16, 4]) torch.Size([16, 1])
        break
    print(len(loader))     # 7 = ceil(100/16)：batch 数
```

参数细节：

- `shuffle=True` 打乱的是**样本顺序**，每个 epoch 重新洗牌。为什么重要：若数据按类别排序（前 5000 条全是标签 0……），不打乱则每批梯度极端偏斜，训练不收敛或崩溃。这是新手大坑。
- `drop_last=True` 通常只在最后一个不满批会影响统计（如 BatchNorm）时开启。
- `num_workers`：子进程并行做数据加载（读图、解码等重活）。内存 tensor 场景收益小，用 0 即可；大图数据集在 Linux 脚本下设 2~4 明显提速。jupyter / Windows 下多进程易踩坑，教学代码一律 0。

## 5.5 形状思维：[batch, features]

从本课起，所有模型第一层都是 `nn.Linear(features, ...)`，输入统一是 `[batch, features]`：

```
单个样本：  x        [784]
一个批次：  xb       [64, 784]      batch=64
模型输出：  logits   [64, 10]       10 类
损失：      标量
```

三个习惯：

1. 拿到任何新数据，先 `print(x.shape, y.shape)`；模型内部第一维永远是 batch，不要硬编码 batch 大小（用 `x.size(0)` 或 reshape 时用 `-1`）。
2. 报形状错误时，读错误信息里的两个形状（如 `mat1 and mat2 shapes cannot be multiplied (64x28 and 784x256)`）——第一个数 64 就是 batch，说明你忘了把 [64,1,28,28] 展平成 [64,784]，L06 的 `nn.Flatten()` 干这个。
3. 训练循环里统计 loss / 准确率都要除以样本数——通过 `.size(0)` 累加，与 batch 大小解耦。

## 5.6 完整小实验：批训练回归

把 L04 的线性回归改成 DataLoader 喂批（完整可运行，对应 `code/L05_dataset_demo.py`）：

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.rand(256, 1)
    y = 3 * X + 2 + 0.05 * torch.randn(256, 1)

    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(20):
        for xb, yb in loader:          # 内层：遍历所有 batch
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    print(model.weight.item(), model.bias.item())   # 约 3 与 2
```

双层循环 `for epoch → for batch` 就是训练循环的骨架。L06 在里面加上训练/评估两套逻辑与统计，就是完整的工程项目代码。

## 练习

1. 把 batch_size 从 32 改成 256 与 4，观察同样 epoch 数下的收敛差异。
   提示：更新次数 = batch 数；小 batch 更新次数多、噪声大。
2. 构造一个「按标签排序」的假数据集，比较 shuffle=True/False 的训练行为。
   提示：把 y=0 的样本全放前面、y=1 全放后面。
3. 自定义一个从 CSV 行（用列表模拟）读取的 Dataset。
   提示：`__init__` 里读入存为 list，`__getitem__` 转 tensor 返回。
4. 打印每个 batch 的实际形状，验证最后一批大小（drop_last=False 时）。
   提示：100 条、batch=16 时最后一批只有 4 条。

## 小结与自测

1. 为什么分批？
   答：显存/内存放不下全量 + 批噪声有益泛化 + 并行效率高。
2. Dataset 协议的两个方法？
   答：`__len__` 与 `__getitem__`。
3. 训练集为什么必须 shuffle？
   答：避免按序分批导致每批分布偏斜、梯度方向极端。
4. num_workers 什么时候调大？
   答：数据加载/预处理重（读图解码）且在 Linux 脚本下；内存 tensor 用 0。
5. 模型第一维为什么不能硬编码 batch 大小？
   答：最后一批与评估集大小都可能不同于 batch_size，模型须对任意 batch 通用。

## 下一课预告

万事俱备。下一课是本阶段最重要的一课：用前 5 课的全部知识，从下载数据到画曲线，完整写出一个 MNIST 分类器，并背下「训练循环八股文」。
