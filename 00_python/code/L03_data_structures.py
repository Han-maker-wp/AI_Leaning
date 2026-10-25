"""L03：核心数据结构演示——list/tuple/dict/set/切片/推导式。

运行：python3 code/L03_data_structures.py
"""


def main() -> None:
    # ---------- list：最常用的有序序列（≈ ArrayList） ----------
    losses = [0.9, 0.7, 0.5, 0.4]
    losses.append(0.35)             # 尾部追加，O(1)
    losses.extend([0.3, 0.28])      # 批量追加
    print(f"losses = {losses}")
    print(f"最后一个 loss：{losses[-1]}")       # 负索引从尾部数
    print(f"最小 loss：{min(losses)}，最大：{max(losses)}")

    sorted_losses = sorted(losses)  # 返回新列表，不动原列表
    losses.sort()                   # 原地排序
    print(f"排序后：{losses}；副本：{sorted_losses}")

    # ---------- tuple：不可变序列，主打「解包」 ----------
    point = (3, 4)
    x, y = point                    # 解包：Java 里只能 point[0]/point[1]
    print(f"x={x}, y={y}")

    a, b = 1, 2
    a, b = b, a                     # 交换变量不需要中间变量
    print(f"交换后 a={a}, b={b}")

    def min_max(xs):
        return min(xs), max(xs)     # 「多返回值」本质是返回一个 tuple

    lo, hi = min_max(losses)
    print(f"min={lo}, max={hi}")

    # ---------- dict：键值对（≈ HashMap），ML 的配置/指标全靠它 ----------
    config = {"lr": 0.001, "epochs": 10, "batch_size": 32}
    config["optimizer"] = "adam"    # 新增/覆盖
    print(f"lr = {config.get('lr')}")          # get 不存在返回 None
    print(f"wd = {config.get('wd', 0.0)}")     # get 带默认值，不抛 KeyError

    for key, value in config.items():          # 遍历键值对
        print(f"  {key} = {value}")

    # ---------- set：去重与集合运算（≈ HashSet） ----------
    tokens = ["train", "train", "eval", "test", "eval"]
    vocab = set(tokens)             # 去重
    print(f"去重后 vocab：{vocab}")
    print(f"'train' 在集合中吗：{'train' in vocab}")   # in 对 set 是 O(1)

    batch1 = {"a", "b", "c"}
    batch2 = {"b", "c", "d"}
    print(f"交集：{batch1 & batch2}，并集：{batch1 | batch2}，差集：{batch1 - batch2}")

    # ---------- 切片：所有序列通用的子序列语法 ----------
    epochs = list(range(10))        # [0..9]
    print(f"前 3 个：{epochs[:3]}")
    print(f"后 3 个：{epochs[-3:]}")
    print(f"隔一个取一个：{epochs[::2]}")
    print(f"反转：{epochs[::-1]}")  # 步长 -1 即反转，也适用于字符串

    # ---------- 推导式：一行完成「筛选 + 变换」 ----------
    values = [3, -1, 4, -1, 5, -9, 2, 6]
    squares = [v * v for v in values]                 # map：逐个变换
    positives = [v for v in values if v > 0]          # filter：筛选
    pos_squares = [v * v for v in values if v > 0]    # filter + map
    print(f"平方：{squares}")
    print(f"正数：{positives}")
    print(f"正数的平方：{pos_squares}")

    # dict 推导式：给每个指标名生成一份假的历史记录
    history = {name: [float(i) for i in range(3)] for name in ["loss", "acc"]}
    print(f"history = {history}")

    # 推导式 vs 普通 for 循环（等价，但推导式更短、意图更清晰）
    result = []
    for v in values:
        if v > 0:
            result.append(v * v)
    assert result == pos_squares    # assert：条件为假时抛 AssertionError，用于自检
    print("推导式与 for 循环结果一致")


if __name__ == "__main__":
    main()
