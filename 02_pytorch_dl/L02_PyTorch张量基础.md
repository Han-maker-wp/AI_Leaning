# 第 2 课：PyTorch 张量基础

> 🎬 开场钩子：PyTorch 的前身叫 Torch——「火炬」，2002 年点燃于纽约大学；2017 年 Facebook AI 接过火种推出 PyTorch，如今已是学界与开源 LLM 生态的主流框架。4 个月后你微调大模型的每一步，都举着这支火炬。而火炬的最小燃料单元，就是今天的主角：张量。

⏱ 建议学时：2 小时

## 本课目标

1. 正确安装 CPU 版 PyTorch 并验证版本；
2. 掌握 tensor 的创建、属性（shape / dtype / device）、运算与索引切片；
3. 分清 reshape 与 view，理解它们与内存共享的关系；
4. 掌握 tensor 与 ndarray 互操作（共享内存的坑）；
5. 会用 `@` 做矩阵乘，理解 float32 默认 dtype 的显存含义，初识 `.to(device)`。

## 2.1 安装

建议为每门课/每个项目建独立虚拟环境。推荐 uv（更快）：

```bash
# 在项目目录下
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

传统 pip 同理：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- CPU 版约 200MB，无 NVIDIA 依赖，本课程前 8 课完全够用。
- GPU 版概念：PyTorch 官网（pytorch.org）下载页按 CUDA 版本选择命令（如 cu121），安装的是带 CUDA 后端的包；同一套代码通过 `.to("cuda")` 自动切换到 GPU。L08 详谈。
- 验证（`python` 还是 `python3` 取决于发行版）：

```bash
python -c "import torch; print(torch.__version__)"
```

能打印 `2.x.x` 即成功。后续课需要时再装：`uv pip install torchvision matplotlib --index-url https://download.pytorch.org/whl/cpu`（matplotlib 不在那个源里时直接 `uv pip install matplotlib`）。

## 2.2 tensor 是什么：numpy ndarray + 三个超能力

对你最省事的定义：**tensor ≈ numpy ndarray + GPU 计算 + 自动微分 + 深度学习生态**。九成 API 名字与行为和 numpy 一致（`zeros`、`arange`、广播、布尔索引……），本课重点讲差异。

创建与属性的完整演示（与 `code/L02_tensor_basics.py` 对应）：

```python
import torch

if __name__ == "__main__":
    print(torch.__version__)                # 2.x

    # ---- 创建 ----
    a = torch.tensor([1.0, 2.0, 3.0])       # 从 python 列表（复制数据）
    b = torch.zeros(2, 3)                   # 全 0
    d = torch.randn(2, 3)                   # 标准正态分布——权重初始化的常客
    e = torch.arange(0, 10, 2)              # tensor([0, 2, 4, 6, 8])
    f = torch.randperm(10)                  # 0..9 的随机排列——打乱数据用

    # ---- 属性 ----
    x = torch.randn(3, 4)
    print(x.shape)                          # torch.Size([3, 4])
    print(x.dtype)                          # torch.float32（默认！numpy 是 float64）
    print(x.device)                         # cpu
    print(x.requires_grad)                  # False（下一课的主角）
```

## 2.3 运算与索引切片：和 numpy 几乎逐条一致

```python
    # ---- 逐元素运算与广播 ----
    u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = torch.tensor([10.0, 100.0])
    print(u + v)        # 广播：v 被复制到每一行
    print(u * 2)

    # ---- 索引切片：与 numpy 语法一致 ----
    m = torch.arange(12).reshape(3, 4).float()
    print(m[0])         # 第 0 行
    print(m[:, 1])      # 第 1 列
    print(m[1:, :2])    # 切块
    print(m[m > 5])     # 布尔掩码，返回一维
```

一个值得记住的差异：很多 PyTorch 操作有两套等价写法——函数版 `torch.sum(x)` 与方法版 `x.sum()`。文档里两者都会出现。

## 2.4 reshape 与 view

- `view(shape)`：只改「形状描述」，不复制数据，返回的 tensor 与原 tensor **共享内存**；前提是内存连续（contiguous），否则报错。
- `reshape(shape)`：尽量返回 view；内存不连续时自动复制一份，永远不报错。

```python
    t = torch.arange(12)
    v1 = t.view(3, 4)          # OK：arange 结果内存连续
    v2 = t.reshape(2, 6)       # OK
    t[0] = 99                  # 改原 tensor
    print(v1[0, 0].item())     # 99 —— v1 与 t 共享内存，跟着变了！
```

经验法则：写通用代码用 `reshape`（安全）；确知连续、在意性能时用 `view`（模型 forward 里常见，如把图像展平）。内存不连续时可先 `.contiguous()` 复制成连续再 view。

## 2.5 与 numpy 互操作：共享内存是坑也是特性

```python
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    t_shared = torch.from_numpy(arr)    # 共享内存（零复制）
    arr[0] = 100.0
    print(t_shared)                     # tensor([100., 2., 3.])  跟着变了！

    t_copy = torch.tensor(arr)          # 复制数据（安全）
    back = t_copy.numpy()               # tensor -> ndarray，同样共享内存
```

记住两条：
- `torch.from_numpy(a)` 与 `t.numpy()` 都是**共享内存**的视图级转换，零开销但会互相影响；
- `torch.tensor(a)` 是复制，安全。调试「数据莫名其妙变了」时，先想想是不是共享内存在作怪。

另外，取单个数值用 `.item()`（0 维 tensor 变 python 数）——累积统计、打印、拼日志时常用。

> 🍿 彩蛋｜似曾相识的「视图」：`torch.from_numpy` 的共享内存，就是 Java 里 `Arrays.asList(arr)` / `list.subList(...)` 的老把戏——返回的不是拷贝，是盯着同一块内存的视图，改一边、两边动。把 numpy 的坑直接映射到 Java 的坑，这题你早就会。

## 2.6 点积与矩阵乘：`*` 不是 `@`

从 numpy 平移过来的重点（也是新手最大 bug 来源）：

- `*`：逐元素乘（带广播），`[3,4] * [3,4] -> [3,4]`；
- `@` / `torch.matmul`：矩阵乘，`[m,n] @ [n,p] -> [m,p]`，内维必须相等；
- 向量点积是一维特例：`u @ v -> 标量`。

```python
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = A @ B                  # [3,5]；torch.matmul(A, B) 等价
    u = torch.randn(4)
    dp = u @ u                 # 点积
```

形状检查习惯：模型报 `mat1 and mat2 shapes cannot be multiplied` 时，第一反应是打印两个操作数的 `.shape`、检查内维。这个习惯在 L06 立刻会用到（784 = 28×28 的展平）。

## 2.7 dtype：float32 是深度学习的通货

```python
    x = torch.randn(3)          # 默认 float32（注意 numpy 默认 float64！）
    y = x.double()              # 转 float64
    z = x.half()                # 转 float16（半精度）
```

为什么默认 float32 而不是 numpy 的 float64？

- 神经网络对数值精度不敏感，float32 的 7 位有效数字足够；
- GPU 的 float32 吞吐远高于 float64（消费级卡高一个数量级）；
- 省一半内存：float64 每数 8 字节，float32 只要 4 字节，float16 / bfloat16 只要 2 字节。

伏笔（量化篇）：模型显存占用 ≈ 参数量 × 每参数字节数。以 BERT-base 约 1.1 亿参数为例：float32 约 440MB，float16 约 220MB，int8 量化约 110MB——「把每个参数存的字节变少」就是量化。现在记住换算关系，后面阶段直接用。

另一个高频坑：不同 dtype 混算会直接报错（如 float32 与 float64 相乘）。从 numpy 拿数据时留意 `arr.astype(np.float32)`。

## 2.8 `.to(device)` 初识

```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(3, 4).to(device)   # 把数据搬到 GPU（无 GPU 则留在 cpu）
    print(x.device)
```

同一运算的所有参与方必须在同一设备上，跨设备会报错。现在只需记住这个写法；模型/数据/损失各自要怎么搬、显存怎么算，L08 专门讲。

## 练习

1. 不运行，先推算下列输出的 shape 再验证：`torch.randn(32, 1, 28, 28)` 经 `.view(32, -1)` 与 `.reshape(-1, 28)` 后各是什么？
   提示：`-1` 由总元素数自动推断；总元素 32×784。
2. 构造一个共享内存的 numpy↔tensor 例子，让对方「意外」变化。
   提示：`from_numpy` 之后改 numpy 数组。
3. 用 `@` 实现「3 个样本、4 个特征过 1 个神经元（无 bias）」的前向计算。
   提示：`[3,4] @ [4,1]` 或 `[3,4] @ [4]`。
4. 计算 1 亿参数在 float32 / float16 / int8 下各占多少 MB。
   提示：字节数 = 参数量 × 4 / 2 / 1，再除以 1024²。

## 小结与自测

1. tensor 与 ndarray 最大的两点差异？
   答：自动微分与 GPU 支持（加上深度学习生态）。
2. view 与 reshape 的区别？
   答：view 零复制但要求内存连续；reshape 必要时自动复制，永远可用。
3. `t.numpy()` 与 `torch.from_numpy` 的坑？
   答：都与原数据共享内存，改一个另一个跟着变。
4. `*` 与 `@` 的区别？
   答：逐元素乘（带广播）vs 矩阵乘（内维必须匹配）。
5. 为什么深度学习默认 float32？
   答：精度够用、GPU 吞吐高、相对 float64 省一半内存。

## 下一课预告

第 3 课讲 autograd：只写前向计算，PyTorch 自动求导。你将第一次用 `backward()` 取代手推梯度公式，并用它复现前置课程的线性回归。
