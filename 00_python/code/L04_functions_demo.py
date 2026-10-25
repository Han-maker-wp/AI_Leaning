"""L04：函数与函数式特性演示——参数机制/lambda/函数是一等公民。

运行：python3 code/L04_functions_demo.py
"""
from typing import Callable


def evaluate(xs: list[float], metric: Callable[[list[float]], float]) -> float:
    """把「怎么评估」作为参数传进来——函数是一等公民。

    Callable[[list[float]], float] 表示：接收 list[float]、返回 float 的函数类型。
    """
    return metric(xs)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)   # 生成器表达式：不建中间列表


def flexible_report(title: str, *columns: str, **options: object) -> None:
    """*columns 收集多余位置参数为 tuple；**options 收集多余关键字参数为 dict。"""
    print(f"== {title} ==")
    print(f"列：{columns}")            # tuple
    print(f"选项：{options}")          # dict


def bad_append(item: str, bucket: list[str] = []) -> list[str]:
    # 陷阱演示：默认 list 只在函数定义时创建一次，所有调用共享同一个对象
    bucket.append(item)
    return bucket


def good_append(item: str, bucket: list[str] | None = None) -> list[str]:
    # 正确姿势：默认 None，函数体内新建
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


def make_multiplier(factor: float) -> Callable[[float], float]:
    """返回函数：闭包捕获了 factor——「工厂」生成行为。"""
    def multiply(x: float) -> float:
        return x * factor
    return multiply


def main() -> None:
    losses = [0.9, 0.6, 0.4, 0.35, 0.31]

    # 1. 把评估函数当参数传（关键字参数调用，自解释）
    print(f"mean = {evaluate(losses, metric=mean):.4f}")
    print(f"variance = {evaluate(losses, metric=variance):.6f}")

    # 2. lambda：一句话的小函数，最常作 sorted 的 key
    models = [("cnn", 0.91), ("rnn", 0.87), ("transformer", 0.95)]
    ranked = sorted(models, key=lambda m: m[1], reverse=True)   # 按准确率降序
    print(f"模型排行：{ranked}")

    # 3. map/filter：返回迭代器，list() 展开后等价于推导式
    doubled = list(map(lambda v: v * 2, [1, 2, 3]))
    big = list(filter(lambda v: v > 0.4, losses))
    print(f"map 结果：{doubled}；filter 结果：{big}")

    # 4. *args / **kwargs：参数的收集
    flexible_report("训练报告", "loss", "acc", lr=0.001, epochs=10)

    # 5. 可变默认值陷阱：连续调用 bad_append 会「记住」上一次的内容
    print(f"bad 第一次：{bad_append('a')}")
    print(f"bad 第二次：{bad_append('b')}")     # ['a', 'b'] —— 惊喜！
    print(f"good 第二次：{good_append('b')}")   # ['b'] —— 干净

    # 6. 闭包：函数工厂
    times3 = make_multiplier(3.0)
    print(f"times3(10) = {times3(10)}")


if __name__ == "__main__":
    main()
