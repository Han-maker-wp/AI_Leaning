# 阶段 2：深度学习 + PyTorch

> 前置：阶段 1（Python / numpy 向量化 + 机器学习基础：手写过 numpy 线性回归、逻辑回归与梯度下降训练循环）
> 本阶段环境：Linux + Python 3.10+，前期 CPU 训练即可（无 GPU 假设）
> 预计总学时：约 20 小时（9 课 + Checkpoint 实战）

## 阶段目标

学完本阶段，你应该能：

1. 熟练使用 PyTorch 训练循环：不加参考资料，独立写出「模型定义 → DataLoader → 训练 → 评估 → 曲线绘制」的完整脚本；
2. 真正理解反向传播：既会用 `loss.backward()`，也能用约 100 行代码手写出它的原理（L09 micrograd）；
3. 建立工程直觉：调参优先级、过拟合诊断、显存估算与 OOM 排查。

一句话定位：把「手写梯度下降训练线性回归」升级成「用框架训练神经网络」。这是 4 个月后微调大模型的地基——未来微调脚本 90% 的行数，就是本阶段的训练循环。

## 课程大纲

| 课次 | 主题 | 核心内容 | 学时 |
|---|---|---|---|
| L01 | 深度学习全景 | 线性回归→神经元、非线性激活的必要性、MLP、「深度」的含义 | 1.5h |
| L02 | PyTorch 张量基础 | 安装、tensor 运算、reshape/view、numpy 互操作、dtype/device | 2h |
| L03 | autograd 自动微分 | 计算图、backward、no_grad/detach、复现线性回归 | 2h |
| L04 | nn.Module 与 MLP | nn.Linear、Sequential、自定义 Module、Java OOP 视角 | 2h |
| L05 | Dataset 与 DataLoader | 分批的原因、自定义 Dataset、形状思维 | 1.5h |
| L06 | 完整训练循环 MNIST | **本阶段最重要一课**：训练循环八股文 | 3h |
| L07 | 训练技巧与调参 | 学习率调度、早停、checkpoint、过拟合诊断、Adam vs SGD | 2h |
| L08 | GPU 训练与显存 | device 惯例、显存构成与估算、OOM checklist、CNN 通识 | 1.5h |
| L09 | micrograd 从零理解反向传播 | 手写 Value 类与 2 层 MLP，揭开 backward() 黑盒 | 3h |

配套可运行代码在 `code/` 目录：`L02_tensor_basics.py`、`L03_autograd_demo.py`、`L05_dataset_demo.py`、`L06_mlp_mnist.py`、`L09_micrograd_mini.py`。

## Checkpoint（阶段通关检验）

完成以下三件事才算通过本阶段：

1. **独立训练分类器**：不看讲义，从空文件开始写一个 MLP，在 MNIST 或 FashionMNIST 上达到 ≥ 97%（MNIST）/ ≥ 88%（FashionMNIST）的测试准确率。允许回看 L06 的「训练循环八股文」模板，但每一行都要能说出为什么这样写。
2. **至少 3 组超参对比实验并分析**：固定数据与评估方式，至少变动 3 组超参（例如学习率 {0.1, 0.01, 0.001}，或优化器 SGD vs Adam，或隐藏层宽度），画出对比曲线，写约 200 字的结论：哪组最好、为什么、曲线形态差异说明了什么。
3. **口头复述**：向空气讲解「一次 `loss.backward()` 到 `optimizer.step()` 之间发生了什么」。讲不清楚就回看 L03 与 L09。

## 资源清单

- 3Blue1Brown 神经网络系列（B 站搜官方中字）：最好的直觉材料。L01 前必看第 1 集；L03/L09 前建议看反向传播相关集数。
- 《动手学深度学习》（zh.d2l.ai，PyTorch 版）：本阶段主参考书，用于概念查漏补缺与更系统的公式推导。
- pytorch.org 官方教程（"Deep Learning with PyTorch: A 60 Minute Blitz"）：快速过一遍官方叙事。
- Andrej Karpathy "Neural Networks: Zero to Hero"（YouTube，有搬运）：L09 的配套视频，强烈建议看完第 1 集。

## 学习建议

- 每课先读讲义、再跑/改配套代码、最后做练习；练习是检验「以为懂了」和「真懂了」的唯一标准。
- 遇到 API 疑问优先查 pytorch.org 官方文档；讲义不覆盖的参数细节，文档永远比博客准确。
- 从 L06 起，所有训练脚本都建议放进 git 管理——阶段 3 的微调脚本将直接复用这些代码骨架。
