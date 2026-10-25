# 第 2 课：Python 语法核心对比 Java

> 🎬 开场钩子：先来道速答题：`x = 10`，下一行接 `x = "hello"`——这段代码能跑吗？在 Java 里你连编译都过不去；在 Python 里它不仅合法，还合法得理直气壮。分歧的根源在于两门语言对「变量」的根本设定不同，本课结束时，「名字标签」会替换掉你直觉里那个「类型盒子」。

⏱ 建议学时：1.5 小时

## 本课目标

1. 能解释「动态类型」下变量的真实含义（名字绑定，不是类型盒子）
2. 用缩进正确组织代码块，不再下意识找大括号
3. 熟练使用 if/for/while，尤其理解 **for 就是 for-each**
4. 会用 None 判空（用 `is`）、f-string 格式化、真值测试简化条件
5. 能判断一个对象是可变还是不可变，并预判赋值/传参时的行为

完整可运行示例见 `code/L02_syntax_demo.py`，本课讲义与其逐节对应。

## 2.1 动态类型：变量是「名字标签」，不是「类型盒子」

```python
x = 10          # 名字 x 绑定到 int 对象 10
x = "hello"     # 同一个名字，改绑到 str 对象——Java 里这是编译错误
print(type(x))  # <class 'str'>，类型跟着对象走，运行时才知道
```

> ☕ Java 视角：Java 的 `int x = 1` 声明了一块带类型的「盒子」，装错了类型编译器报错。Python 的 `x = 1` 是把名字 `x` 贴到对象上，变量本身没有类型，对象才有。你可以想象成「所有变量都是 Object 引用 + 运行时类型检查」。代价：类型错误藏到运行时；对策：类型提示（第 4 课）+ 测试。

一个直接推论：函数参数、返回值都不声明类型，IDE 补全依赖类型提示——所以现代 Python 代码都提倡写提示。

## 2.2 缩进即语法：大括号的退役

```python
score = 85
if score >= 90:
    grade = "A"          # 冒号 + 4 空格缩进构成代码块
elif score >= 80:        # Java 的 else if 在这里写作 elif
    grade = "B"
else:
    grade = "C"
```

缩进不是风格，是语法。缩进错了是 `IndentationError`，程序根本跑不起来。约定：统一 4 个空格，编辑器把 Tab 设为 4 空格、开启「空格缩进」。

空块不允许（Java 的 `{}` 空体合法），需要占位时写 `pass`：

```python
def todo_later():
    pass        # 先占个位，之后再填
```

## 2.3 控制流：if/while 面熟，for 彻底不同

逻辑运算符换符号：`and` / `or` / `not`（不是 `&&` `||` `!`）。取反用 `not x`，很少用 `!= None`。

链式比较是合法的：`if 0 < x < 10:`（Java 只能 `0 < x && x < 10`）。

三元表达式语序不同：`b if cond else a` 对应 Java 的 `cond ? b : a`。

**for 是 for-each**——Python 没有 C 式三段 for，`for` 直接遍历任何可迭代对象：

```python
for item in ["train", "eval", "test"]:   # 相当于 Java 的 for (var item : list)
    print(item)

for i, item in enumerate(["a", "b", "c"], start=1):   # 要下标时：enumerate
    print(i, item)

for i in range(10):        # range(10) 产生 0..9，等价 for (int i = 0; i < 10; i++)
    total += i
```

> ☕ Java 视角：Python 的 for 就是增强 for 循环。需要索引就 `enumerate`，需要纯计数就 `range(n)`。「i 从几加到几」的心智模型换成「对序列里每个元素做什么」——这是后面向量化思维（第 7 课）的铺垫：能用「对每个元素」表达的，最终多半不该用循环写。

while 与 Java 语义一致，配 `break` / `continue` 使用，不再展开。

## 2.4 None：Python 的 null，但判空用 is

```python
value = None
if value is None:          # 惯例：判 None 一律用 is，不要用 ==
    print("是 None")
```

`None` 是全局唯一的单例对象（类似 Java 的 `null` 指向，但 None 本身是个真对象）。访问 None 的不存在属性报 `AttributeError`，就是 Python 世界的 NullPointerException。

> ☕ Java 视角：`is` 比较的是「同一个对象」（引用相等 ==），`==` 默认也比较对象身份、但可被类重载成「值相等」（如同重写 equals）。因为 `==` 可能被重载，判 None 用 `is` 才绝对可靠。Python 没有 Optional，惯用法是「返回 None + 调用方 is None 检查」，或者 `cfg if cfg is not None else 默认值`。

## 2.5 f-string：字符串格式化的现代标准

```python
lr, epoch = 0.001, 7
print(f"epoch={epoch}, lr={lr}, lr*100={lr * 100:.1f}%")
# epoch=7, lr=0.001, lr*100=0.1%
```

引号前加 `f`，大括号里可以放任意表达式，冒号后是格式说明符（`:.4f` 保留 4 位小数、`:>8` 右对齐宽 8）。ML 圈打印 loss、进度条全靠它。

> ☕ Java 视角：相当于把 `String.format` 的参数内联到字符串里，可读性高一档。旧代码里的 `%` 格式化和 `.format()` 是历史产物，新代码统一 f-string。

> 🍿 彩蛋｜Python 之禅：在 REPL 敲 `import this`，会打印出《Python 之禅》（PEP 20）——这是官方亲手埋的彩蛋。其中一句 "There should be one-- and preferably only one --obvious way to do it"（做事应有一种、且最好只有一种显而易见的写法）与 Perl「条条大路通罗马」的理念针锋相对——上一句「新代码统一 f-string」正是它的日常落地。

## 2.6 真值测试：空容器为假

```python
empty = []
if not empty:              # 空列表为假
    print("空列表为假")

if items:                  # 惯用写法：非空才执行
    process(items)
```

以下值在布尔上下文中都为**假**：`False`、`None`、`0`、`0.0`、`""`、`[]`、`{}`、`set()`、`()`。其余全为真。

> ☕ Java 视角：Java 里 `if (xs != null && !xs.isEmpty())` 两连击，Python 一个 `if xs:` 搞定（Python 没有「空指针 + 空容器」双重语义的坑）。注意 `if x:` 对 `x = 0` 也为假——判断「是不是 0」要显式写 `if x == 0`。

## 2.7 可变与不可变：一半坑的根源

不可变：`int` `float` `str` `tuple`。可变：`list` `dict` `set`。

关键行为：

```python
a = [1, 2, 3]
b = a              # b 和 a 指向同一个 list 对象
b.append(4)
print(a)           # [1, 2, 3, 4] —— a 也"变了"

c = a.copy()       # 浅拷贝，两个独立对象
```

> ☕ Java 视角：Python 赋值和传参永远是「传对象引用」（相当于 Java 里只传对象、从不传值拷贝），且没有基本类型/包装类型之分。`b = a` 不是拷贝内容，是复制引用。要独立副本就显式 `.copy()`（浅）或 `copy.deepcopy()`（深）。字符串不可变 → 所有 str 方法都返回新串，`s.upper()` 不改变 s。

比较操作：`==` 比值（list 逐元素比），`is` 比身份（同一对象）。日常用 `==`，判 None/单例用 `is`。

## 2.8 异常处理速览（todo 项目要用）

```python
def read_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:            # 捕获特定异常，相当于 Java 的 catch
            print("不是整数，重输")

# raise ValueError("说明")            # 主动抛出，相当于 throw
```

> ☕ Java 视角：`try/except/finally` ≈ `try/catch/finally`；异常也是类、也有继承层级（都继承 BaseException）；没有 checked exception，编译器不强迫你捕获。

## 完整演示代码

以下与 `code/L02_syntax_demo.py` 相同，可直接运行：

```python
"""L02：Python 语法核心演示——对照 Java 的差异点逐一验证。运行：python3 code/L02_syntax_demo.py"""


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

    # 3. for 是 for-each：直接遍历元素
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

    def find_config(key):      # 模拟「查不到返回 None」
        configs = {"lr": 0.01}
        return configs.get(key)

    cfg = find_config("batch_size")
    default = cfg if cfg is not None else 32   # 三元：x if cond else y
    print(f"batch_size = {default}")

    # 5. f-string：大括号内可放任意表达式
    lr, epoch = 0.001, 7
    print(f"epoch={epoch}, lr={lr}, lr*100={lr * 100:.1f}%")

    # 6. 真值测试：空容器、0、None、空串都为假
    empty_list = []
    if not empty_list:
        print("空列表为假")

    # 7. 可变 vs 不可变
    a = [1, 2, 3]
    b = a              # b 与 a 指向同一个 list 对象
    b.append(4)
    print(f"a 也变成了 {a}")

    c = a.copy()       # 浅拷贝后才是两个独立对象
    c.append(99)
    print(f"copy 后互不影响：a={a}, c={c}")

    s = "abc"
    # s[0] = "x"       # 取消注释会报 TypeError：str 不可变
    s2 = s.upper()     # 只能生成新串
    print(f"字符串不可变，upper 产生新对象：{s2}")


if __name__ == "__main__":
    main()
```

## 练习

1. 写函数 `clamp(v, lo, hi)` 把 v 限制在 [lo, hi] 内，用三元表达式一行完成。提示：`lo if v < lo else ...` 可以嵌套。
2. 用 for + enumerate 打印九九乘法表的第 5 行（5x1 到 5x9）。提示：f-string 里写 `{i}*5={i*5}`。
3. 验证 `==` 与 `is` 的差异：构造两个「值相等」的 list，分别用两种方式比较并打印结果。提示：两个独立构造的相等 list，`==` 为 True，`is` 为 False。
4. 思考题（下一课揭晓）：`def f(x, cache=[])` 为什么危险？提示：默认值只在 def 执行时求值一次。

## 小结与自测

1. Python 变量与 Java 变量的本质区别？—— 名字到对象的绑定，类型属于对象且运行时才检查。
2. Java 的 `for (int i=0; i<10; i++)` 在 Python 怎么写？—— `for i in range(10):`。
3. 判 None 为什么用 `is`？—— None 是单例，is 比身份不会被重载过的 `==` 欺骗。
4. 哪些值为假？—— False、None、0、0.0、空串、空 list/tuple/dict/set。
5. `b = a` 后 `b.append(4)`，a 会变吗？—— 会，两者是同一对象；`.copy()` 之后才独立。
6. try/except 对应 Java 的什么？—— try/catch，异常同样是类层级，但没有 checked exception。

## 下一课预告

第 3 课进入核心数据结构 list/tuple/dict/set：它们对应你熟悉的 ArrayList/HashMap/HashSet，外加两个 Java 没有的利器——切片语法与推导式。
