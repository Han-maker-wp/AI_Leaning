"""L02：PyTorch 张量基础速览。

运行：python L02_tensor_basics.py
对应讲义：02_pytorch_dl/L02_PyTorch张量基础.md
"""
import numpy as np
import torch


def main():
    print("torch 版本:", torch.__version__)

    # ---------- 1. 创建 ----------
    a = torch.tensor([1.0, 2.0, 3.0])   # 从 python 列表创建（复制数据）
    b = torch.zeros(2, 3)               # 全 0
    d = torch.randn(2, 3)               # 标准正态分布：权重初始化的常客
    e = torch.arange(0, 10, 2)          # tensor([0, 2, 4, 6, 8])
    f = torch.randperm(10)              # 0..9 的随机排列：打乱数据用
    print("创建:", a.shape, b.shape, e, f, sep="\n")

    # ---------- 2. 属性 ----------
    x = torch.randn(3, 4)
    print("shape:", x.shape, "| dtype:", x.dtype, "| device:", x.device)

    # ---------- 3. 运算与索引 ----------
    u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = torch.tensor([10.0, 100.0])
    print("广播加法:\n", u + v)
    print("逐元素乘 2:\n", u * 2)

    m = torch.arange(12).reshape(3, 4).float()
    print("m:\n", m)
    print("第 0 行:", m[0])
    print("第 1 列:", m[:, 1])
    print("布尔掩码 m>5:", m[m > 5])

    # ---------- 4. view / reshape 与共享内存 ----------
    t = torch.arange(12)
    v1 = t.view(3, 4)        # view：零复制，要求内存连续
    _ = t.reshape(2, 6)      # reshape：必要时自动复制，永远可用
    t[0] = 99                # 修改原 tensor
    print("view 与原 tensor 共享内存:", v1[0, 0].item() == 99.0)   # True

    # ---------- 5. numpy 互操作 ----------
    arr = np.array([1.0, 2.0, 3.0])
    t_shared = torch.from_numpy(arr)    # 共享内存（零复制）！
    arr[0] = 100.0
    print("from_numpy 共享内存:", t_shared[0].item() == 100.0)     # True

    t_copy = torch.tensor(arr)          # torch.tensor 是复制（安全）
    back = t_copy.numpy()               # tensor -> ndarray，同样共享内存
    print("tensor.numpy() 可用:", back.shape == (3,))

    # ---------- 6. 点积与矩阵乘 ----------
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    print("A @ B 形状:", (A @ B).shape)              # [3, 5]
    vec = torch.randn(4)
    print("点积 vec @ vec =", (vec @ vec).item())    # 标量

    # ---------- 7. dtype ----------
    print("默认 dtype:", torch.randn(1).dtype)       # torch.float32
    print("half 后 dtype:", torch.randn(1).half().dtype)  # torch.float16
    n_params = 110_000_000                           # BERT-base 约 1.1 亿参数
    print(f"显存估算: fp32={n_params * 4 / 1024**2:.0f}MB, "
          f"fp16={n_params * 2 / 1024**2:.0f}MB, int8={n_params * 1 / 1024**2:.0f}MB")

    # ---------- 8. device ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(3, 4).to(device)
    print("当前 device:", x.device)


if __name__ == "__main__":
    main()
