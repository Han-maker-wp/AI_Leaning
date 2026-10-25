"""L06：sklearn 小项目——Pipeline + 鸢尾花分类 + 决策树/随机森林对比。

运行前：uv pip install scikit-learn
运行：python3 code/L06_sklearn_demo.py
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
    #    交叉验证的每一折都会重新 fit，杜绝数据泄漏。
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X_train, y_train)
    print(f"LogisticRegression 测试准确率：{pipe.score(X_test, y_test):.4f}")

    # 3. 交叉验证：5 折，比单次切分更稳（Pipeline 整体送进去）
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"5 折 CV 准确率：{scores.mean():.4f} ± {scores.std():.4f}")

    # 4. 决策树：max_depth 限制复杂度（不限制必然过拟合，见练习 1）
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    print(f"DecisionTree(max_depth=3) train={tree.score(X_train, y_train):.4f} "
          f"test={tree.score(X_test, y_test):.4f}")

    # 5. 随机森林：bootstrap 采样 + 特征随机 + 多树投票（bagging 降方差）
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(X_train, y_train)
    print(f"RandomForest(200 trees) test={forest.score(X_test, y_test):.4f}")

    # 6. 详细评估：混淆矩阵 + 每类 precision/recall/F1（L04 概念落地）
    y_pred = forest.predict(X_test)
    print("\n混淆矩阵（行=真实，列=预测）：")
    print(confusion_matrix(y_test, y_pred))
    print("\nclassification report：")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 7. 「学出来的 if-else」里谁重要：特征重要性
    importances = dict(zip(iris.feature_names, forest.feature_importances_.round(3)))
    print(f"特征重要性：{importances}")


if __name__ == "__main__":
    main()
