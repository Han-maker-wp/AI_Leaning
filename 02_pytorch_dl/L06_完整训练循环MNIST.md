# 第 6 课：完整训练循环——MNIST 分类器

> 🎬 开场钩子：深度学习第一次大规模商业落地，不是搜索引擎也不是推荐系统，而是——认支票。上世纪 90 年代，Yann LeCun 的 LeNet 在银行系统里读支票上的手写数字，而 MNIST 正源自他 1998 年前后的手写数字研究。今天，这份「祖传数据」是深度学习的 Hello World，也是你独立训练的第一个完整模型——本阶段最重要的一课，从它开始。

⏱ 建议学时：3 小时（**本阶段最重要一课**）

## 本课目标

1. 会用 torchvision 下载 MNIST 并正确做归一化；
2. 能独立写出并逐行解释「模型 → DataLoader → 训练循环 → 评估循环 → 曲线」全流程；
3. 在 CPU 上把 MLP 训到约 97~98% 测试准确率；
4. 能默写「训练循环八股文」模板，并说出每一行的作用。

## 6.1 任务与数据：MNIST

MNIST：70000 张 28×28 灰度手写数字图片（60000 训练 + 10000 测试），标签 0~9，深度学习的 Hello World。

安装：`uv pip install torchvision --index-url https://download.pytorch.org/whl/cpu`（与 torch 配套的 CPU 版；matplotlib 用 `uv pip install matplotlib`）。

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                        # PIL 图片 -> tensor：0~255 缩放到 [0,1]，形状 [1,28,28]
    transforms.Normalize((0.1307,), (0.3081,)),   # 用 MNIST 训练集的均值/标准差做标准化
])

train_set = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
```

为什么两步变换：

- ToTensor：像素是 uint8（0~255），直接喂网络数值太大；先统一缩到 [0,1]；
- Normalize：0.1307 / 0.3081 是前人在 MNIST 训练集上算好的像素均值/标准差（单通道所以括号里各一个数；彩色图会是 3 个）。把数据变成均值 0、方差 1 附近，损失面更圆、各特征梯度尺度一致——lr 好调、收敛快。

工程直觉：**归一化是深度学习里性价比最高的操作之一**。机器学习课讲过的标准化，在这里由 transform 机制自动化完成。

## 6.2 模型：MLP

```python
import torch.nn as nn

def build_mlp():
    return nn.Sequential(
        nn.Flatten(),               # [batch,1,28,28] -> [batch,784]：展开成长向量
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),         # 输出 10 个 logits（每类一个分数）
    )
```

三个为什么：

1. **为什么 Flatten**：MLP 只吃 `[batch, features]`，要把 28×28 图像展开成 784 维（代价是丢失空间结构——CNN 保留它，L08 通识）。
2. **为什么输出 10 维**：10 类各一个分数，谁大判谁（`argmax`）。
3. **为什么输出层不加 softmax**：`nn.CrossEntropyLoss` 内部先做 log_softmax 再算 NLL。手动先 softmax 等于做两次，梯度会算错。约定：**模型出 logits，损失函数负责 softmax**（推理要概率时单独 `torch.softmax(logits, dim=-1)`）。

## 6.3 损失与优化器

```python
import torch

loss_fn = nn.CrossEntropyLoss()                            # 多分类标准损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam 是省心起点（L07 对比 SGD）
```

CrossEntropyLoss 对照你手写过的逻辑回归：相当于「softmax + 取正确类的负对数似然」，推导与二分类 BCE 同源，只是框架合并了实现。标签直接给整数类别（0~9），**不是** one-hot——这也是 PyTorch 的约定。

## 6.4 完整脚本（分段逐行讲解）

完整可运行版在 `code/L06_mlp_mnist.py`，下面分段拆解。

### 数据管道

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)  # 评估无需打乱，大 batch 更快
```

> 🍿 彩蛋｜batch_size 的直觉：它很像数据库连接池的池大小——太小，来回开销占大头；太大，内存扛不住也未必更快。调优思路同款：从默认值起步，看吞吐（每 epoch 耗时）与资源（显存占用）的数据再说话。

### 训练循环（核心八股文所在）

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()                                    # 切训练模式（dropout/BN 等需要；MLP 无影响但习惯必须养）
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)            # 数据搬设备（CPU 时是空操作）
        optimizer.zero_grad()                        # ① 清梯度（.grad 是累积的！L03）
        logits = model(x)                            # ② 前向
        loss = loss_fn(logits, y)                    # ③ 损失（标量）
        loss.backward()                              # ④ 反向：填 .grad
        optimizer.step()                             # ⑤ 用 .grad 更新参数

        total_loss += loss.item() * y.size(0)        # 统计：批均损失还原成批总损失再累加
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n
```

六件事（统计不算）：`清梯度 → 前向 → 损失 → 反向 → 更新 → 记录`。五行主体就是「八股文」，背下来。

统计三行的技巧：`loss.item() * y.size(0)` 把「批均损失」还原成「批总损失」累加，最后除以总样本数，得到真实的 epoch 平均（最后一批可能不满）。`(logits.argmax(dim=1) == y)` 是布尔向量，`.sum().item()` 数对了几个。`.item()` 处处出现：把 tensor 变 python 数，避免持有计算图引用。

### 评估循环

```python
@torch.no_grad()                                     # 装饰器等价于 with torch.no_grad() 包住整个函数
def evaluate(model, loader, loss_fn, device):
    model.eval()                                     # 切评估模式
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n
```

与训练循环只差三处：`no_grad`（不建图：省内存、更快、防止误反传）、没有 backward/step、`model.eval()`。`model.train()` / `model.eval()` 必须成对出现——本模型没有受模式影响的层，但从第一天养成习惯（第 3 阶段 BERT 里的 Dropout 就吃这个开关）。

### 主流程与曲线

```python
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mlp().to(device)
    # ... DataLoader、loss_fn、optimizer 如前 ...

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc)):
            history[k].append(v)
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")
    plot_history(history)
```

### 曲线绘制

```python
import matplotlib
matplotlib.use("Agg")            # 无显示器的服务器用 Agg 后端；本地桌面可删掉改 plt.show()
import matplotlib.pyplot as plt

def plot_history(h, path="mnist_curves.png"):
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["test_loss"], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(epochs, h["train_acc"], label="train")
    axes[1].plot(epochs, h["test_acc"], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
```

怎么读曲线（L07 深讲）：train loss 应稳定下降；test acc 应升到 97% 上下；若 train 与 test 曲线分叉（train 继续降、test 回升）是过拟合信号——本模型 5 个 epoch 内基本看不到，L07 会人为造出来给你诊断。

## 6.5 运行与预期结果

CPU 上每 epoch 约 10~30 秒（机器而异），5 个 epoch 全程几分钟；首次运行下载数据集（约 10MB）到 `data/`。典型输出：

```
epoch 1: train_loss=0.2845 train_acc=0.9160 test_loss=0.1412 test_acc=0.9564
epoch 2: train_loss=0.1217 train_acc=0.9630 test_loss=0.1061 test_acc=0.9679
...
epoch 5: train_loss=0.0234 train_acc=0.9923 test_loss=0.0758 test_acc=0.9779
```

数字会随随机种子略有差异，test_acc 落在 0.975~0.98 都正常。

## 6.6 训练循环八股文（背下来）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel(**kwargs).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    # ---- 训练 ----
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()          # ① 清梯度
        logits = model(x)              # ② 前向
        loss = loss_fn(logits, y)      # ③ 损失
        loss.backward()                # ④ 反向
        optimizer.step()               # ⑤ 更新
    # ---- 评估 ----
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            ...
```

默写标准：不看资料写出 ①~⑤ 的顺序，并回答两问——① 为什么不能省（.grad 累积）；④ 与 ⑤ 各自动了什么（④ 填 .grad，⑤ 用 .grad 改参数）。第 3 阶段微调 BERT、未来微调 LLM，主体都是这份模板——变的只是模型与数据。

> 🍿 彩蛋｜「八股文」的自嘲：工程师把必背模板戏称八股文——从 Java 面试八股到训练循环五步，用法一脉相承：先能闭眼默写，再谈理解与变通。这不是贬义，出手不错靠的就是肌肉记忆。

## 6.7 常见报错速查

| 报错 / 现象 | 原因 | 修法 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (64x28 and 784x256)` | 忘了 Flatten 或 reshape 维度错 | 打印 x.shape，加 `nn.Flatten()` |
| loss 不降，acc 停在 10% | lr 太大/太小、忘 step、标签错位 | 先查 optimizer.step() 是否在循环内 |
| train acc 高但 test acc 异常低 | 忘了 model.eval()（若用 BN/Dropout）或数据没 shuffle | 评估前切 eval；检查 loader |
| `Expected all tensors to be on the same device` | 模型与数据不同设备 | 两者都 .to(device) |
| loss 变 nan | lr 太大导致数值上溢 | lr 降 10 倍 |

## 练习

1. 把隐藏层 256/128 改成 512/256 与 64/32，比较最终测试准确率与训练时间。
   提示：参数量与耗时都随之变化；大网络在 MNIST 上收益递减。
2. 去掉两个 ReLU（层直连），重新训练并解释结果。
   提示：L01 的「无激活等价于一层」在实验上会怎么显形？
3. 把 batch_size 改成 16 与 512，观察训练曲线抖动与耗时变化。
   提示：抖动来自批噪声；每 epoch 更新次数 = 60000/batch_size。
4. 从测试集取第 0 张图，打印模型预测与真实标签（单张推理）。
   提示：`test_set[0]` 返回 (tensor[1,28,28], int)；进模型前 `unsqueeze(0)` 加 batch 维。

## 小结与自测

1. 训练循环五步的顺序与各自的作用？
   答：zero_grad 清旧梯度 → forward 算预测 → loss 量化误差 → backward 求梯度 → step 更新参数。
2. 为什么输出层不加 softmax？
   答：CrossEntropyLoss 内置 log_softmax，外部再做会重复且梯度错误。
3. model.train() / model.eval() 改变什么？
   答：切换 Dropout / BatchNorm 等模式相关层的行为；MLP 无影响但必须成对习惯。
4. 评估为什么用 no_grad？
   答：不建计算图，省显存提速，并防止误反传。
5. Normalize 的两个常数是什么？
   答：MNIST 训练集的像素均值 0.1307 与标准差 0.3081（前人统计的定值）。

## 下一课预告

模型能训了，下一步是「训得更好」：学习率调度、早停、checkpoint 保存/加载，以及拿到一对分叉曲线时如何诊断过拟合。第 7 课全是工程实战技巧，也是 Checkpoint 三组对比实验的方法论。
