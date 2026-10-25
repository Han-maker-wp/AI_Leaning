# 第 8 课：GPU 训练与显存

> 🎬 开场钩子：2012 年，赢下 ImageNet、唤醒深度学习时代的 AlexNet，是在两张 GTX 580 上训练出来的——彻头彻尾为游戏玩家设计的显卡。游戏显卡从此意外成为 AI 硬通货。这节课搞懂 GPU 训练怎么运转、显存花在哪，也为几周后的「本地 GPU vs 云 GPU」决策点打好底子。

⏱ 建议学时：1.5 小时

## 本课目标

1. 掌握 device 的惯例写法，写出 CPU/GPU 通用的训练脚本；
2. 说清 `.to(device)` 必须覆盖哪些对象（模型、输入、标签）；
3. 能用粗略公式估算训练显存（参数 / 梯度 / 优化器状态 / 激活值）；
4. 拿到 OOM 报错有一套排查顺序；
5. CNN 通识：能解释卷积核与池化在做什么、为什么 LLM 路线不强求。

## 8.1 device 惯例写法

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"   # Apple 芯片是 "mps"
print(torch.cuda.is_available())      # 本课程 CPU 环境，输出 False
model = model.to(device)              # 模型搬过去（原地修改 model 本身）
x = x.to(device)                      # 每个输入张量都要搬
```

这套写法让同一份脚本在有无 GPU 的机器上都能跑——L06 的脚本已经这样写了。要点：`model.to(device)` 是原地操作（返回值也是 model）；而**张量的 `.to(device)` 不改变原变量**，必须 `x = x.to(device)` 赋值接收——新手最常见 bug。

## 8.2 `.to(device)` 覆盖清单

| 对象 | 写法 | 说明 |
|---|---|---|
| 模型 | `model.to(device)` | 递归搬所有参数 |
| 输入 batch | `x = x.to(device)` | 每个循环里 |
| 标签 batch | `y = y.to(device)` | 与 x 同批 |
| 带参数的损失（少见） | 如 `nn.CrossEntropyLoss(weight=w)` 里的 w | 无参数损失（MSE、默认 CE）无需 |

模型在 cuda、输入在 cpu 时报错 `Expected all tensors to be on the same device, but found ... cuda:0 and cpu!`——看到就检查三件套。无参数的损失函数跟随输入自动计算，不用单独搬。

## 8.3 显存都被什么占了（粗算公式）

设参数量为 P（`sum(p.numel() for p in model.parameters())`），float32 训练时：

| 项 | 大小 | 说明 |
|---|---|---|
| 参数 | 4P | 权重本体 |
| 梯度 | 4P | backward 填的 .grad |
| 优化器状态 | 0（纯 SGD）/ 4P（动量）/ 8P（Adam） | Adam 存一阶、二阶矩各一份 |
| 激活值 | 与 batch × 序列 × 隐层成正比 | 反向传播要对中间结果求导，必须留着 |

粗算（Adam）：**约 16P 字节 + 激活**。代入两个数感受量级：

- L06 的 MLP：P ≈ 26 万 → 16P ≈ 4MB，激活才是大头（随 batch 增长）——CPU 都随便跑；
- BERT-base：P = 1.1 亿 → 参数 4P = 440MB，Adam 训练 16P ≈ 1.76GB，再加激活（batch × seq 越大越多）；
- 7B 模型全参微调：仅参数+梯度+Adam ≈ 16 × 7e9 = 112GB——单卡放不下。这就是为什么大模型微调要用 LoRA / 量化（第 4、5 阶段的主题），今天先记住数字的来源。

推理（不训练）没有梯度与优化器状态，只要 4P（float32）甚至 2P（fp16）——所以「推理能跑、一训练就 OOM」很正常。

激活值的直觉：每层的输出 `[batch, features]` 都要存到 backward 用完为止；batch 翻倍、序列翻倍，激活大致线性翻倍。`no_grad()` 评估不建图、不存激活，所以评估显存远小于训练。

## 8.4 batch_size 与显存

显存 ≈ 固定项（16P）+ 激活项（∝ batch_size）。于是：

- OOM 时降 batch_size 立竿见影（激活项线性下降）；
- batch 大小不影响参数占用——不是「模型变大」，是「中间结果变多」；
- 小 batch 可用**梯度累积**模拟大 batch：连续 N 个 batch 只累积梯度不 step，第 N 个再 step（L03 说过的 `.grad` 累积特性的「故意用法」）。

## 8.5 OOM 排查 checklist（按顺序试）

报错形如 `CUDA out of memory`：

1. 评估/推理代码是否包在 `no_grad()` 里？（最常见低级错误）
2. 降 batch_size（64 → 32 → 16）；
3. 缩短序列长度 / 裁剪输入（NLP 的 max_length）；
4. 开混合精度：`with torch.autocast(device_type="cuda", dtype=torch.bfloat16):` 包住前向+损失（参数仍 fp32、计算用半精度，显存近乎减半；需 GPU 支持，CPU 不适用）；
5. 换更小模型 / 冻结部分层（`p.requires_grad_(False)`，被冻结的层不存梯度与优化器状态）；
6. 梯度累积模拟大 batch；
7. 实在不行：`torch.cuda.empty_cache()` 清缓存（治标），或换更大显存的卡 / 多卡（本课程不展开）。

诊断工具：`torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()` 打印当前与峰值，看哪一步涨上去。

> 🍿 彩蛋｜OOM 是本圈的 NPE：`CUDA out of memory` 之于炼丹师，约等于 NullPointerException 之于 Java 程序员——人人见过、出现时机永远不合适、报错位置几乎从不是病因所在（病根常在三行之前）。好在它比 NPE 讲道理：有 checklist 可查。

## 8.6 混合精度代码骨架（有 GPU 后照抄）

混合精度（fp16 / bfloat16）是 OOM 与提速的常用手段，先把骨架放在这里备用（CPU 上跑不了，认脸即可）：

```python
# 前向与损失包在 autocast 里；backward 与 step 照旧
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

原理：参数仍是 float32（主副本），矩阵乘等大计算用半精度——显存近半、Tensor Core 加速。bfloat16 相对 float16 的优势是动态范围大（指数位多、尾数位少），不易溢出，现代 GPU 首选。

顺带一提云 GPU 实践：真项目租按小时计费的云 GPU（或 Colab 免费档），device 惯例写法保证脚本零改动迁移；数据集与权重先上传，代码后上传。第 5 阶段微调 LLM 时这套流程会天天用。

> 🍿 彩蛋｜bfloat16 的 b：不是 better 也不是 big，是 Brain——它源自 Google Brain 团队为 TPU 设计的浮点格式，故得名 Brain Float。尾数少、指数多，牺牲精度换动态范围，正对深度学习的胃口。

## 8.7 附：CNN 一页通识（LLM 路线不强求）

LLM 路线上 CNN 不是必修，但 MNIST 这类图像任务的 SOTA 是 CNN，通识与面试也需要懂概念。

**卷积（convolution）**：一个小窗口权重（卷积核，如 3×3）在图像上滑动，每到一处做「窗口内元素加权求和」输出一个数，滑完全图得到特征图：

```
输入 5×5 图像            3×3 卷积核             输出 3×3 特征图
1 1 1 0 0               1 0 1
0 1 1 1 0               0 1 0       核滑过每个位置，
0 0 1 1 1               1 0 1       每个输出 = 核与对应窗口的逐元素乘加
1 0 0 1 1
0 0 1 1 0               （局部感受野：每个输出只看输入的一小片）
```

与 MLP 展平相比的两个关键性质：

- **局部感受野**：每个输出只看输入一小片——符合图像「邻近像素相关」的先验；
- **权重共享**：同一个核滑过全图——参数量从 784×256 级降到 3×3=9 个，且天然获得平移不变性。

**池化（pooling）**：如 2×2 max-pooling，每 2×2 小块只留最大值——下采样：缩小特征图、扩大感受野、增加鲁棒性。

CNN 典型结构：[卷积 ×N + 激活 (+ 池化)] 重复几轮 → Flatten → 全连接输出。MNIST 上两层小 CNN 可达 99%+（MLP 约 98%）。想动手：把 L06 模型换成两个 `nn.Conv2d` 试试（pytorch.org 官方 60 分钟教程有现成示例）；想懂直觉：3B1B 神经网络系列讲 CNN 的集数。第 3 阶段开始与图像告别，专注语言。

## 练习

1. 估算：7B 参数模型 float32 推理需多少 GB？Adam 全参微调呢？
   提示：4P / 16P，P=7e9，除以 1024³。
2. 写一个 `param_count(model)` 函数，打印 L06 模型的参数量与 16P 的 MB 估算。
   提示：`sum(p.numel() for p in model.parameters())`。
3. 在有 GPU 的机器上（或云 GPU）故意漏搬一个 batch 张量，复现 device 报错。
   提示：没有 GPU 就记住报错信息的样子。
4. 手算 `nn.Conv2d(1, 16, kernel_size=3)` 的参数量，并与 `named_parameters` 对照。
   提示：`out_channels × in_channels × k × k + out_channels`。

## 小结与自测

1. `model.to(device)` 与 `x.to(device)` 的行为差异？
   答：模型是原地修改；张量不是，必须赋值接收。
2. Adam 训练时显存粗算公式？
   答：约 16 × 参数量 字节（参数 4 + 梯度 4 + Adam 状态 8），再加激活。
3. 为什么推理比训练省显存？
   答：无梯度、无优化器状态，no_grad 下也不存激活。
4. OOM 第一步查什么？
   答：评估是否 no_grad；其次降 batch_size。
5. CNN 相对 MLP 的两条核心性质？
   答：局部感受野与权重共享（参数骤减 + 平移不变性）。

## 下一课预告

最后一课放大招：不调 `torch.autograd`，从零手写一个约 100 行的迷你自动微分引擎（Karpathy 的 micrograd 思路），用链式法则复刻 `backward()` 的全部机制——揭开黑盒，阶段收官。
