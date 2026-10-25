# 第 6 课：sklearn 实战与树模型

> 🎬 开场钩子：本课登场的模型，输出就是你写了二十年的 if-else——「花瓣长度 ≤ 2.45 就判 setosa」。区别只有一个：这些 if-else 不是你写的，是模型从数据里自己挑的。更妙的是，把几百棵各自带偏见的树凑在一起投票，居然比任何一棵单独调教的树都准——这就是随机森林的群体智慧。

⏱ 建议学时：2 小时

## 本课目标

1. 熟练使用 estimator 范式：fit / predict / score / get_params
2. 会用 Pipeline 串联预处理与模型，并解释它为什么防数据泄漏
3. 说出决策树节点在学什么（if-else 的自动生成），以及剪枝超参数的作用
4. 解释随机森林 = bagging + 特征扰动，为什么能降方差
5. 跑通完整小项目：划分 → Pipeline → 交叉验证 → 混淆矩阵 → 特征重要性

完整代码见 `code/L06_sklearn_demo.py`。

## 6.1 fit/predict：估算器统一范式

sklearn 里一切模型都实现同一套接口（第 1 课已见面，现在正式立规）：

```python
model = SomeModel(超参数=值)   # 构造：只配超参数，不碰数据
model.fit(X_train, y_train)   # 训练：从数据学参数
model.predict(X_test)         # 预测
model.score(X_test, y_test)   # 快捷评估（分类默认准确率，回归默认 R²）
model.get_params()            # 查看超参数（网格搜索靠它反射）
```

配套生态全部围绕这个约定工作：换模型只改一行构造，`cross_val_score`、`GridSearchCV`、Pipeline 对任何模型一视同仁。

> ☕ Java 视角：这是教科书级的「面向接口编程 + 策略模式」：estimator 是隐式接口（鸭子类型，阶段 0 L05），LogisticRegression/DecisionTree/RandomForest 是可插拔实现。你写 Java 时用 Spring 注入换实现，这里用换类名换实现——但契约没有 interface 强制，靠的是社区纪律。这个设计让 sklearn 活了十几年：新算法只要实现 fit/predict 就能进生态。

## 6.2 Pipeline：把预处理与模型串成流水线

典型流程：标准化 → 模型。朴素写法会埋雷：

```python
# 危险写法：先对「全部数据」标准化再切分
X_scaled = StandardScaler().fit_transform(X)   # ← 测试集的均值方差泄入了 scaler！
X_train, X_test = train_test_split(X_scaled, ...)

# 正确写法：scaler 只在训练集上 fit
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)            # 用训练集的参数转换测试集
```

`fit` = 从数据学参数（均值/方差），`transform` = 应用参数。**测试集永远只能被 transform，不能参与 fit**——否则「考试前看过考卷统计量」，评估虚高（第 1 课的数据泄漏落地成具体场景）。

Pipeline 把正确姿势自动化：

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),          # 第一步：预处理
    ("clf", LogisticRegression(max_iter=500)),   # 第二步：模型
])
pipe.fit(X_train, y_train)     # 串联执行：scaler.fit_transform → clf.fit
pipe.predict(X_test)           # scaler.transform → clf.predict
```

Pipeline 本身也是 estimator——可以整体丢进 `cross_val_score`，每折各自 fit scaler，泄漏彻底绝缘。

> ☕ Java 视角：Pipeline ≈ 责任链模式，每步一个处理器；与责任链不同的是它把「训练态参数」（scaler 的均值）跟流水线一起持久化，`joblib` 序列化后就是可部署的推理服务雏形。

## 6.3 决策树：机器学出来的一堆 if-else

对你最亲切的模型——因为它就是你写了几十年的东西，只是这次 if-else 的条件是学出来的：

```
if 花瓣长度 <= 2.45:
    类别 = setosa
elif 花瓣宽度 <= 1.75:
    if 花瓣长度 <= 4.95: 类别 = versicolor
    else: 类别 = virginica
else:
    类别 = virginica
```

训练（ID3/CART 等算法）在每一步做同一件事：**遍历「哪个特征 + 哪个阈值」，选出让划分后子节点最纯的那一刀**。纯度用 gini 不纯度或信息熵度量——一次二分把信息熵降低最多的就是好问题（信息增益，你概率课的老朋友）。

性质：可解释性极强（feature_importances_ 直接读）、无需标准化（只比大小不比尺度）、天然支持非线性。但**单棵树不加限制必过拟合**（每个叶子只装一个样本就 100% 纯）——所以剪枝超参数是必修课：`max_depth`（最大深度）、`min_samples_leaf`（叶节点最少样本数）。第 5 课的「模型复杂度旋钮」在树上就是这个旋钮。

## 6.4 随机森林：bagging 的群体智慧

一棵树是方差大户（换个训练集树形大变）。随机森林的两招去方差：

1. **Bootstrap 采样（bagging）**：从 n 个样本里有放回抽 n 个，训练一棵树；重复 B 次（如 200 棵）。每棵树看到了不同的数据子集、错得各不相同
2. **特征扰动**：每个节点分裂时只在随机选的一部分特征里找最优刀——强迫树与树之间「专长不同」

预测时全体投票（分类）或平均（回归）。直觉：**if 每棵树的误差互不相关，B 棵树平均后方差降到 1/B**（相关系数会打折，但方向如此）——大数定律在模型界的应用。

同类还有梯度提升家族（XGBoost、LightGBM）：逐棵训练、每棵修正前面的残差。表格/结构化数据任务至今仍是它们的天下（不是 LLM 路线重点，记住名字与定位即可）。

> 🍿 彩蛋｜XGBoost 名字拆开念：eXtreme Gradient Boosting——极限梯度提升。它在结构化数据比赛里曾长期霸榜，以至于圈内一度形成默契：遇到表格数据，先跑一版 XGBoost 当基线，再谈别的。

## 6.5 特征工程入门

「数据和特征决定上限，模型只是逼近上限」：

- **数值特征**：标准化/归一化（树模型不需要，线性/神经网络需要）；长尾分布取 log
- **类别特征**：one-hot 编码（`pd.get_dummies` 或 sklearn 的 OneHotEncoder——名字给你，用时查文档）
- **缺失值**：删除、按中位数/众数填充
- **组合特征**：交叉（如「年龄段 × 城市」）、比率（点击/曝光）
- **泄漏自查**：任何「用了未来信息/测试信息」构造的特征都是毒药

> ☕ Java 视角：特征工程 ≈ 数据清洗 + DAO 层建模，只是产物不是入库的实体而是 (n, d) 矩阵。你做业务系统时对「口径」的洁癖，在这里就是对特征口径的洁癖——同一个意思。

## 6.6 完整小项目：鸢尾花分类

与 `code/L06_sklearn_demo.py` 相同：

```python
"""L06：sklearn 小项目——Pipeline + 鸢尾花分类 + 决策树/随机森林对比。

运行前：uv pip install scikit-learn
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    # 1. 数据与划分（stratify=y 保证各类比例一致）
    iris = load_iris()
    X, y = iris.data, iris.target          # X:(150,4)  y:(150,)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    # 2. Pipeline：标准化 + 逻辑回归。
    #    scaler 只在 fit 时「学习」训练集的均值方差——包进 Pipeline 后，
    #    交叉验证每折都会重新 fit，杜绝数据泄漏。
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X_train, y_train)
    print(f"LogisticRegression 测试准确率：{pipe.score(X_test, y_test):.4f}")

    # 3. 交叉验证：5 折，比单次切分更稳（Pipeline 整体送进去）
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"5 折 CV 准确率：{scores.mean():.4f} ± {scores.std():.4f}")

    # 4. 决策树：max_depth 限制复杂度，观察 train 与 test 的差距
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    print(f"DecisionTree(max_depth=3) train={tree.score(X_train, y_train):.4f} "
          f"test={tree.score(X_test, y_test):.4f}")

    # 5. 随机森林：多树投票
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(X_train, y_train)
    print(f"RandomForest(200 trees) test={forest.score(X_test, y_test):.4f}")

    # 6. 详细评估：混淆矩阵 + 每类 precision/recall/F1（第 4 课的概念落地）
    y_pred = forest.predict(X_test)
    print("\n混淆矩阵（行=真实，列=预测）：")
    print(confusion_matrix(y_test, y_pred))
    print("\nclassification report：")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 7. 顺便看一眼「学出来的 if-else」里谁重要
    importances = dict(zip(iris.feature_names, forest.feature_importances_.round(3)))
    print(f"特征重要性：{importances}")


if __name__ == "__main__":
    main()
```

跑完后按这个顺序复盘（也是所有 sklearn 项目的标准动作）：切分 → Pipeline → CV 选超参 → 测试集一次评估 → 混淆矩阵细看 → 特征重要性复盘。

## 6.7 树模型在 LLM 路线中的定位

明确预期：**树模型不是通往 LLM 的主线**——你的目标路线是神经网络 → PyTorch → 预训练模型 → 微调。但 30 分钟的投入换三样东西：

1. fit/predict/Pipeline 范式在后续每个框架里都有影子（PyTorch 的 Trainer/HuggingFace 的 Trainer）
2. 偏差-方差、正则、交叉验证这些「非深度学习也成立」的通用判断力
3. 工作里遇到表格数据任务（销量预测、风控、日志分类）时，随机森林/GBDT 常常是性价比之王

## 练习

1. 把决策树 `max_depth` 设为 `None`（不限制），对比 train/test 差距，再看 `tree.get_depth()`。提示：train 分数冲到接近 1.0，第 5 课的现场重演。
2. 构造泄漏版对照实验：先 `StandardScaler().fit_transform(X)` 再切分，与 Pipeline 版比较 CV 分数。提示：鸢尾花太简单可能看不出差，思考「什么数据集上这个差别会致命」。
3. 用 `sklearn.datasets.make_classification` 造 1000 样本、20 特征（其中 5 个 informative）的数据重跑本课流程。提示：`from sklearn.datasets import make_classification`，看特征重要性能否找出那 5 个。
4. 对鸢尾花把某个特征列用 `rng.permutation` 打乱，重跑随机森林看 importance 变化。提示：被打乱的特征与标签失去关联，importance 应明显下降。

## 小结与自测

1. 为什么预处理必须包进 Pipeline？—— 预处理只能在训练折上 fit，防数据泄漏，且部署时一条龙。
2. fit 与 transform 的区别？—— fit 学参数（如均值方差），transform 应用参数；Pipeline 自动串联两者。
3. 决策树每个节点在学什么？—— 「哪个特征 + 哪个阈值」能让划分后子节点最纯（gini/信息增益最大）。
4. 随机森林靠什么降方差？—— bootstrap 采样 + 特征随机 + 多树投票（bagging）。
5. 树模型在 LLM 路线中的定位？—— 非主线，懂直觉即可；表格数据任务的常胜将军。
6. estimator 范式对应 Java 的什么设计思想？—— 面向接口/策略模式：统一契约、实现可插拔。

## 下一课预告

阶段 1 完结——去做 README 里的 Checkpoint：不看讲义，纯 NumPy 从零写逻辑回归（训练循环 + loss 曲线 + 决策边界）。下一阶段（PyTorch 上手）将把你手写的循环升级：`loss.backward()` 接管求导、`optimizer.step()` 接管更新、GPU 接管算力，`nn.Module` 接管建模型。教程与文档见 pytorch.org，进阶配合 zh.d2l.ai 与 huggingface.co/learn。
