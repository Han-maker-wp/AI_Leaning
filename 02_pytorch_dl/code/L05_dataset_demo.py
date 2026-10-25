"""L05：Dataset 与 DataLoader 演示。

内容：
  1) TensorDataset + DataLoader 的形状观察（形状思维：[batch, features]）
  2) 自定义 Dataset（__len__ / __getitem__，像实现 Java 的 List 接口）
  3) 用 DataLoader 喂批训练一个线性回归（双层循环骨架）

运行：python L05_dataset_demo.py
对应讲义：02_pytorch_dl/L05_Dataset与DataLoader.md
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ReviewDataset(Dataset):
    """自定义 Dataset 示例：__getitem__ 返回 (文本, 标签)。

    框架只通过 __len__ 和 __getitem__ 访问数据，背后可以是内存、CSV、数据库。
    """

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def demo_shapes():
    X = torch.randn(100, 4)      # 100 个样本、4 个特征
    y = torch.randn(100, 1)
    ds = TensorDataset(X, y)

    print("数据集大小 len(ds):", len(ds))
    xi, yi = ds[0]               # 取单条是一个 (x, y) 元组
    print("单条形状:", xi.shape, yi.shape)

    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=False, num_workers=0)
    print("batch 数 len(loader):", len(loader))     # ceil(100/16) = 7

    last_size = None
    for xb, yb in loader:
        last_size = xb.shape
    print("每批形状 [batch, features]:", last_size)  # 最后一批只有 100-96=4 条！

    ds2 = ReviewDataset(["好评", "差评", "很好用"], [1, 0, 1])
    print("自定义 Dataset 取第 0 条:", ds2[0], "| len:", len(ds2))


def demo_batch_training():
    """双层循环 for epoch -> for batch：训练循环的骨架。"""
    torch.manual_seed(0)
    X = torch.rand(256, 1)
    y = 3 * X + 2 + 0.05 * torch.randn(256, 1)

    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(20):
        for xb, yb in loader:
            opt.zero_grad()                  # ① 清梯度
            loss = loss_fn(model(xb), yb)    # ② 前向 + ③ 损失
            loss.backward()                  # ④ 反向
            opt.step()                       # ⑤ 更新

    print(f"训练后: weight={model.weight.item():.4f}（真值 3）, "
          f"bias={model.bias.item():.4f}（真值 2）")


if __name__ == "__main__":
    print("== 1) TensorDataset / DataLoader / 自定义 Dataset ==")
    demo_shapes()
    print("== 2) DataLoader 喂批训练线性回归 ==")
    demo_batch_training()
