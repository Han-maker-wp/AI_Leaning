"""L02：Python 语法核心演示——对照 Java 的差异点逐一验证。

运行：python3 code/L02_syntax_demo.py
"""


def main() -> None:
    # 1. 动态类型：名字可以随时绑定到任何类型的对象
    x = 10            # 此刻 x 指向 int
    x = "hello"       # 同一个名字，改指向 str —— Java 里这是编译错误
    print(type(x))    # <class 'str'>，类型在运行时才知道

    # 2. 缩进即代码块：冒号 + 4 空格缩进
    score = 85
    if score >= 90:
        grade = "A"
    elif score >= 80:          # Java 的 else if 写作 elif
        grade = "B"
    else:
        grade = "C"
    print(f"grade = {grade}")

    # 3. for 是 for-each：直接遍历元素，没有 i < n 的三段式
    for item in ["train", "eval", "test"]:
        print(f"phase: {item}")

    for idx, item in enumerate(["a", "b", "c"], start=1):
        print(f"{idx}: {item}")

    total = 0
    for i in range(10):        # 等价 for (int i = 0; i < 10; i++)
        total += i
    print(f"sum(0..9) = {total}")

    # 4. None：判空一律用 is
    value = None
    if value is None:
        print("value 是 None")

    def find_config(key):      # 模拟「查不到返回 None」的函数
        configs = {"lr": 0.01}
        return configs.get(key)

    cfg = find_config("batch_size")
    default = cfg if cfg is not None else 32   # 三元表达式：x if cond else y
    print(f"batch_size = {default}")

    # 5. f-string：大括号内可放任意表达式与格式说明符
    lr, epoch = 0.001, 7
    print(f"epoch={epoch}, lr={lr}, lr*100={lr * 100:.1f}%")

    # 6. 真值测试：空容器、0、None、空串都为假
    empty_list = []
    if not empty_list:         # Java 要写 empty_list != null && !isEmpty()
        print("空列表为假")

    # 7. 可变 vs 不可变
    a = [1, 2, 3]
    b = a              # b 与 a 指向同一个 list 对象（传引用，不是拷贝）
    b.append(4)
    print(f"a 也变成了 {a}")

    c = a.copy()       # 浅拷贝后才是两个独立对象
    c.append(99)
    print(f"copy 后互不影响：a={a}, c={c}")

    s = "abc"
    # s[0] = "x"       # 取消注释会报 TypeError：str 不可变
    s2 = s.upper()     # 只能生成新串，原串不变
    print(f"字符串不可变，upper 产生新对象：{s2}")


if __name__ == "__main__":
    main()
