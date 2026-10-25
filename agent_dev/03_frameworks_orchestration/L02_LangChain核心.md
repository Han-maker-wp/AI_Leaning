# 第 2 课：LangChain 核心——三大抽象与链式组合（含批判视角）

> 🎬 开场钩子：同一个需求——"输入产品评论，输出情感标签和置信度"：裸写大约 40 行（拼 prompt、发请求、抠 JSON、容错重试）；用 LangChain 大约 15 行。但论坛上你也会刷到另一种帖子：有人把 LangChain 从生产栈里整个移除，理由是"排查一个 prompt 问题花了两小时，最后发现是框架偷偷改了我的消息结构"。同一件事既让人离不开又让人想删掉——本课的目标就是给你一把尺子：它到底替你做了什么，代价是什么。

⏱ 建议学时：2.5 小时

## 本课目标

- 写出 LangChain 三大抽象（ChatModel / PromptTemplate / OutputParser）各自解决的问题与最小用法
- 用 LCEL 的 `|` 管道把三段组合成链，并解释每一段的输入输出类型约束
- 用 `with_structured_output` + Pydantic 拿到类型安全的结构化输出
- 说清"胶水层帮倒忙"的三种典型情形，并知道对应的绕开手段

## 一、三大抽象：每一段都在消灭一类重复代码

### 1.1 ChatModel：统一"调模型"这一步

裸写时你面对的是 HTTP 细节；LangChain 把它收敛成一个对象：

```python
# 文件头说明：依赖 langchain-openai 包；需要环境变量 OPENAI_API_KEY
# 安装：uv pip install langchain langchain-openai（版本以官方文档为准）
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 参数名以官方文档为准
resp = llm.invoke("用一句话解释什么是 token")
print(resp.content)      # 文本内容
print(resp.response_metadata)  # 元数据（含 token 用量等，字段以文档为准）
```

它的真实价值不在封装 HTTP，而在**换实现零成本**：换成 Anthropic 或本地模型（装对应包），下游代码不动。这是典型的"面向接口编程"收益。

> ☕ Java 视角：这就是 JDBC 的价值——`DriverManager` 背后换成哪家驱动，业务代码不变。ChatModel 是 LLM 界的 `Connection` 接口。

### 1.2 PromptTemplate：消息构造的模板引擎

把"系统提示 + 用户输入 + 少量格式控制"从字符串拼接升级为声明式模板：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是电商评论分析器，只输出 JSON。"),
    ("human", "评论：{review}"),
])
messages = prompt.invoke({"review": "物流很快，包装一般"}).to_messages()
print(messages[-1].content)  # 渲染结果：评论：物流很快，包装一般
```

模板的价值是**可测试**：你可以单独渲染、单独断言，而不是在最终请求里才发现变量没替换。阶段 4 讲评估时你会回来感谢这一点。

### 1.3 OutputParser：从自由文本到结构化数据

模型输出是字符串，你的业务要对象。OutputParser 家族负责"提示模型按格式输出 + 把输出解析成 Python 对象 + 解析失败时修一次"。

> ☕ Java 视角：Pydantic 在这里的角色 = DTO + Bean Validation 合体。`with_structured_output(Sentiment)` 一步拿到"已校验的强类型对象"，相当于 Jackson 反序列化 + `@Valid` 一起做掉，而且 schema 是直接从类定义生成的。

## 二、LCEL：用 `|` 把三段串成链

LangChain 表达式（LCEL）的核心想法：每一段都是 `Runnable`（有标准输入输出类型的可执行单元），用管道符连接，前一段的输出类型必须能喂进后一段：

```python
# 文件头说明：依赖 langchain-openai 包；需要环境变量 OPENAI_API_KEY
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Sentiment(BaseModel):
    """情感分析结果——Pydantic 模型即输出契约"""
    label: str = Field(description="positive / negative / neutral")
    confidence: float = Field(ge=0, le=1, description="置信度")


def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是电商评论分析器。"),
        ("human", "评论：{review}"),
    ])
    # 链：渲染提示 → 调模型 → 结构化解析，| 前后类型必须衔接
    structured_llm = llm.with_structured_output(Sentiment)
    chain = prompt | structured_llm
    result = chain.invoke({"review": "物流很快，包装一般"})
    print(result)          # Sentiment 对象，字段可直接点用
    print(result.label)    # 比如 positive


if __name__ == "__main__":
    main()
```

一个高频误用（练习 1 会让你亲手踩一次）：`with_structured_output` 之后**不要**再接 `StrOutputParser`——前者输出对象，后者期待字符串，LCEL 的类型约束只是文档层面的约定（不像编译器会报错），接错只会在 `invoke` 运行时挂，这是它的第一个坑。

`|` 之后你还能自然使用 `.invoke`（单次）、`.batch`（并发批处理）、`.stream`（流式）——同一个链对象三种执行模式，这是 LCEL 真心好用的部分。

> 🍿 彩蛋｜"胶水代码"（glue code）的黑话由来：行业里把"不实现业务逻辑、只负责把各组件粘起来"的代码叫胶水。它没有错——但胶水一旦有了自己的行为（偷偷加空格、吞异常、改消息顺序），你就得去读它的源码。判断一个抽象层值不值，就看它的行为是否"透明到可以忽略"。

## 三、批判视角：胶水层何时帮倒忙

经过 L01 和本课，你可以攒出一张"帮倒忙清单"：

### 3.1 抽象泄漏：你终究要读它的源码

典型症状：模型返回的 tool call 帧格式各家不同，框架做了归一化——直到某天某个字段归一化错了，你的 Agent 静默丢参数。此时调试路径是：你的代码 → 框架适配层 → 原始响应三层展开。**规则：关键路径上，你必须有能力随时降级到"直接看原始请求/响应"**（`llm.invoke` 的 debug 日志开关，见官方文档 debugging 章节）。

### 3.2 版本快闪：0.x 阶段的破坏性变更

LangChain 处于 0.x 阶段，大版本间多次出现"导入路径迁移、类改名、包拆分（langchain-core 拆出）"。你半年前写的教程代码今天跑不通，大概率不是你的错。

> ☕ Java 视角：像极了 Spring 2→3、Jakarta EE 改名那轮迁移之痛——但节奏快十倍。对策也一样：锁版本号（`uv pip install "langchain==x.y.z"`）、升级当成独立任务排期，而不是顺手升。

### 3.3 隐式行为：默认值在替你做决定

历史消息如何裁剪、失败如何重试、结构化输出的 schema 怎么塞进提示——框架都有默认实现。默认值与你的场景不匹配时，"框架级 bug"其实是"默认值错配"。**对策：对每一处影响 prompt 或 token 用量的默认行为，显式配置一次，宁可啰嗦。**

### 绕开手段（不是弃用）

- 只用"薄"的部分：`ChatModel` 抽象 + `with_structured_output`，其余裸写
- 链超过三层、或需要循环/分支时：直接换 LangGraph（下一课），不要用链套链模拟状态机
- 生产代码锁精确版本，升级走独立 PR + 回归测试（阶段 4 L02 会接上这个话题）

### 3.4 何时整个绕开：三个信号

1. **整条链就是"一次调用"**：拼好 prompt → 调模型 → 解析 JSON，没有任何复用诉求——三个抽象带来的间接性只剩成本
2. **你在框架里花的时间超过在业务上**：判断标志是"我在读框架 changelog 的频率高于读自己代码的频率"
3. **需要极致控制 prompt 字节**：例如上下文预算极紧、每个 token 都要精打细算的场景，任何替你拼装消息的层都是干扰

绕开不等于全弃：常见的稳态是**混合**——模型抽象与结构化输出用框架（换模型省事），编排与工具完全自研（透明可控）。这正是 L06"分层混合"策略在单课尺度上的预演。

## 练习

1. 把阶段 1 裸写的"评论情感分析"改成上面的 LCEL 链，跑通后故意把 `StrOutputParser` 接在 `with_structured_output` 后面，观察并记录报错信息。
   提示：报错发生在 invoke 时而非组装时，想想为什么。
2. 给 `Sentiment` 增加一个 `aspects: list[str]`（评论涉及方面），重新运行，检查模型对复杂输入的输出是否稳定。
   提示：在 Field 里写清 description，模型对字段用途的理解直接影响稳定性。
3. 打开框架的 debug/详细日志开关（查官方文档 debugging 页），抓一次完整请求，对比"你以为发给模型的提示"与"实际发送的提示"。
   提示：重点看 system 消息前后是否多了框架注入的内容。
4. 写 5 行笔记回答：你的下一个项目里，LangChain 的哪部分你打算用、哪部分打算绕开？
   提示：用第三节的"帮倒忙清单"逐条对照。

## 小结与自测

- Q1：三大抽象各消灭哪类重复代码？
  一句话：ChatModel 消灭调用协议差异、PromptTemplate 消灭消息拼接、OutputParser 消灭字符串到对象的解析容错。
- Q2：`with_structured_output(Sentiment)` 输出什么类型？
  一句话：Pydantic 对象（已校验），后面不能再接期待字符串的解析器。
- Q3：胶水层"帮倒忙"的三种典型情形？
  一句话：抽象泄漏（要读源码才能调试）、0.x 破坏性变更、隐式默认值替你做决定。
- Q4：同一个 LCEL 链对象支持哪三种执行模式？
  一句话：invoke 单次、batch 并发批处理、stream 流式。

## 下一课预告

第 3 课进入本阶段主角 LangGraph：把 `while True` 循环升级成显式状态机，你会看到循环、分支、人工中断如何从"代码技巧"变成"图的形状"。
