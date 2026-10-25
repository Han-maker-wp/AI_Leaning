"""L05：类与对象演示——继承/魔术方法/鸭子类型/dataclass/property。

运行：python3 code/L05_class_demo.py
"""
from dataclasses import dataclass, field


class Dataset:
    """最简类：数据 + 行为放一起。类属性 kind 被所有实例共享（类似 Java static）。"""

    kind = "tabular"                        # 类属性（static）

    def __init__(self, name: str, rows: list[list[float]]) -> None:
        # self 显式出现在每个方法的第一个参数——它就是 Java 的 this
        self.name = name                    # 实例属性，每个对象各自一份
        self.rows = rows

    def __len__(self) -> int:               # 让 len(ds) 可用
        return len(self.rows)

    def __getitem__(self, i: int):          # 让 ds[0]、for 循环可用
        return self.rows[i]

    def __repr__(self) -> str:              # 面向开发者的表示，打印/调试时看到
        return f"Dataset(name={self.name!r}, n={len(self.rows)})"


class LabeledDataset(Dataset):              # 继承：括号里写父类
    def __init__(self, name: str, rows, labels: list[int]) -> None:
        super().__init__(name, rows)        # 复用父类初始化
        self.labels = labels

    def summary(self) -> str:
        counts: dict[int, int] = {}
        for y in self.labels:
            counts[y] = counts.get(y, 0) + 1
        return f"{self.name}: {len(self)} 条，类别分布 {counts}"


@dataclass
class TrainConfig:
    """dataclass：自动生成 __init__/__repr__/__eq__，相当于 Java record。

    可变默认值必须用 field(default_factory=...)，原理同 L04 的可变默认值陷阱。
    """
    lr: float = 1e-3
    epochs: int = 10
    tags: list[str] = field(default_factory=list)

    @property
    def total_steps_hint(self) -> int:
        """property：像访问属性一样调用计算逻辑（无括号）。"""
        return self.epochs * 100


class FakeModule:
    """简化版 nn.Module 骨架：体会「模型是类、层是属性、调用走 forward」。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        # 假装这两个 list 是可训练参数（真正的权重是 numpy 数组/tensor）
        self.weight = [[0.01] * in_dim for _ in range(out_dim)]
        self.bias = [0.0] * out_dim
        self.training = True

    def forward(self, x: list[float]) -> list[float]:
        out = []
        for row, b in zip(self.weight, self.bias):
            out.append(sum(wi * xi for wi, xi in zip(row, x)) + b)
        return out

    def __call__(self, x: list[float]) -> list[float]:
        # model(x) 之所以能调用，是因为实例实现了 __call__
        return self.forward(x)


def describe(obj) -> str:
    """鸭子类型：不检查类型，只管对象「有没有这个能力」。"""
    return obj.summary() if hasattr(obj, "summary") else repr(obj)


def main() -> None:
    ds = LabeledDataset("iris-mini", [[5.1, 3.5], [4.9, 3.0]], [0, 1])
    print(ds)                       # 触发 __repr__
    print(len(ds), ds[0])           # 触发 __len__ 与 __getitem__
    print(ds.summary())

    print(Dataset.kind, ds.kind)    # 类属性通过类或实例都能读

    cfg = TrainConfig(lr=0.01)
    cfg.tags.append("baseline")
    print(cfg)                      # dataclass 自动生成的 repr
    print(f"total_steps_hint = {cfg.total_steps_hint}")   # property 不加括号

    model = FakeModule(in_dim=2, out_dim=3)
    logits = model([5.1, 3.5])      # 等价于 model.forward(...)——__call__ 的功劳
    print(f"logits = {[round(v, 4) for v in logits]}")

    print(describe(ds))             # 有 summary → 调用（鸭子类型）
    print(describe(cfg))            # 没有 summary → 回退到 repr


if __name__ == "__main__":
    main()
