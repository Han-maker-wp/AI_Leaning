# 第 5 课：多 Agent 协作——四种模式与那个最容易被忽略的成本

> 🎬 开场钩子：2023 年，Stanford 的研究者搭了一个"生成式智能体小镇"：25 个 agent 各有身份与日程，在模拟社区里互相串门、传播消息，论文（arXiv 2304.03442）发布后引发轰动——agent 们居然自发组织出了社交活动。轰动之余，工程视角该问的问题是：25 个 agent 互相说话，一次"对话"要烧多少 token？谁在记账？谁在仲裁分歧？这节课把浪漫的部分剥掉，只留工程师要面对的部分：模式与成本。

⏱ 建议学时：2 小时

## 本课目标

- 说出单 Agent 解决不了、确实需要多 Agent 的两类真实需求（上下文隔离 / 异质职责）
- 画出路由、流水线、辩论、监督者四种协作模式的结构，各举一个适用场景
- 用"分布式系统"视角解释多 Agent 的三大新增成本：通信 token、调试复杂度、级联失败
- 在模拟代码里实践两条通信成本控制手段：结构化消息、摘要传递

## 一、先泼冷水：多数"多 Agent"是伪需求

上节课的判断标准再用一次：**看是否真的需要多个独立上下文**。两类真实需求：

1. **上下文隔离**：单 Agent 的历史塞不下所有信息——如"代码 agent"需要完整代码库上下文，"测试 agent"需要测试框架上下文，两者互塞会挤爆窗口并互相干扰
2. **异质职责**：不同角色的提示词、工具集、权限差异大到放在一个 prompt 里互相打架——如"检索员只读数据库，写手只能产出文案，审核员有权否决"

此外都是伪需求：任务本身线性、信息量小、角色差异弱——单 Agent + 多工具更便宜、更好调试。

> ☕ Java 视角：这就是微服务拆分的老辩论——"单体优先（monolith first）"派的核心论据：拆分带来的网络调用、序列化、分布式调试成本，只有在团队与业务的边界真的需要对齐时才划算。把"服务间 HTTP 调用"换成"agent 间 token 传递"，论据一字不改照样成立。

## 二、四种协作模式

### 2.1 路由（Router）：一个前台，分流给专家

```
用户 ──> [router agent] ──> 技术专家 agent
                      └──> 账务专家 agent ──> 汇总回答
```

前台 agent 不解决问题，只做分类与转发；各专家独立带各自的工具与提示。适用：客服分流、工单分类。这是成本最低的多 Agent 模式——专家之间不通信。

### 2.2 流水线（Pipeline）：固定工序，接力传递

```
[检索 agent] ──产物──> [写作 agent] ──草稿──> [审校 agent] ──终稿
```

固定顺序、单向流动，每个环节收上一步产物（不是全量对话）。适用：内容生产、报告生成——阶段 2 的 RAG 加一个"审校环节"就是两段流水线。工程上最接近你熟悉的批处理管道。

### 2.3 辩论（Debate）：多个 agent 独立作答再互相批驳

```
问题 ──> [agent A 独立答] ─┐
      ──> [agent B 独立答] ─┼──> 互相批驳 N 轮 ──> [裁判 agent] 定稿
```

同一问题背靠背独立求解，再互相挑错，最后裁判合成。适用：高价值、低容错的判断题（方案评审、数学推理）。代价：token 消耗是单 Agent 的好几倍——只配得上高价值问题。

### 2.4 监督者（Supervisor）：一个大脑调度多个工人

```
           ┌──────────────<──────────────┐
用户 ──> [supervisor] ──派活──> [worker A] ──汇报──┘
            └──────────派活──> [worker B]
```

supervisor 维护全局计划，决定下一步派给谁、何时结束。适用：任务能拆成子任务、需要动态调度的场景。这是最像"项目经理"的模式，也是最容易失控循环的模式（supervisor 和 worker 互相"我觉得你先"）。

> 🍿 彩蛋｜多 Agent 的祖师爷竟是个周末项目：BabyAGI——2023 年 4 月 Yohei Nakajima 的周末项目，核心就是一个任务列表循环：生成任务 → 排优先级 → 执行 → 根据结果生成新任务。它简单到"配不上框架"这个词，却启发了随后一大批多 Agent 编排工具（包括监督者模式的流行）。工程直觉：**先把调度循环写明白，再谈模式名词**。

## 三、通信成本：token 是新的带宽

多 Agent 系统本质是个分布式系统，老三样成本全部复活，只是单位从毫秒/字节变成了 token：

| 分布式系统的成本 | 多 Agent 里的形态 |
|---|---|
| 网络传输 + 序列化 | agent 间消息全部按 token 计费（历史重复传递最烧钱） |
| 调试链路变长 | 一次失败要回放多个上下文才能定位 |
| 级联失败 | 前一个 agent 的幻觉被下游当事实引用并放大 |

控制手段（按性价比排序）：

1. **传产物不传历史**：下游只拿上游的结论/结构化产物，不附完整对话。一条"检索到 3 条结果 + 摘要"的消息，比转发 8000 token 原始对话便宜一个数量级
2. **结构化消息**：用 Pydantic 定义 agent 间消息（字段：结论、依据、置信度、遗留问题），既省 token 又防"下游误解上游"
3. **摘要传递**：必须传长内容时，先压成摘要（有损，但可控）
4. **预算硬上限**：给整条链路设 token 预算，超限强制收敛——和给微服务设超时同一性质

## 四、动手：监督者模式的可运行模拟

下面这个脚本用规则函数代替真实模型，专注看清**消息结构与调度循环**（不依赖任何 API key）：

```python
# 文件头说明：纯标准库 mock 演示，无需安装依赖、无需 API key
from dataclasses import dataclass, field


@dataclass
class Task:
    """supervisor 与 worker 之间的结构化消息：传产物，不传历史"""
    description: str
    result: str = ""
    done: bool = False


@dataclass
class Worker:
    name: str
    skills: list[str]                       # 该工人能接的任务类型
    history: list[Task] = field(default_factory=list)

    def work(self, task: Task) -> None:
        # 真实项目：这里把任务描述发给对应角色的 LLM
        task.result = f"[{self.name}] 已完成：{task.description}（mock 结果）"
        task.done = True
        self.history.append(task)


def supervisor_loop(plan: list[str], workers: list[Worker],
                    max_turns: int = 10) -> list[Task]:
    """监督者调度循环：按计划派活，汇报结果，直到计划完成或触顶收敛。"""
    tasks = [Task(d) for d in plan]
    for turn in range(max_turns):           # 预算硬上限：防失控循环
        pending = [t for t in tasks if not t.done]
        if not pending:
            break
        task = pending[0]
        # 调度规则（mock）：找第一个技能匹配的工人；找不到则标记失败避免死循环
        worker = next((w for w in workers if any(s in task.description for s in w.skills)), None)
        if worker is None:
            task.result = "无匹配工人，任务失败"
            task.done = True                # 失败也置 done：不让坏任务卡死整条链
        else:
            worker.work(task)
        print(f"第 {turn + 1} 轮：{task.description} -> {task.result[:30]}…")
    return tasks


def main() -> None:
    workers = [Worker("检索员", ["查", "搜"]), Worker("写手", ["写", "总结"])]
    plan = ["查一下竞品价格", "写一份竞品小结", "翻译成英文"]
    finished = supervisor_loop(plan, workers)
    print(f"\n完成 {sum(t.done for t in finished)}/{len(finished)} 项：")
    for t in finished:
        print(f"  {'✓' if t.done else '✗'} {t.description}：{t.result[:40]}")


if __name__ == "__main__":
    main()
```

跑一遍注意三个设计点：`Task` 是结构化产物（不是对话历史转发）；`max_turns` 是硬预算；"无匹配工人"标记失败而不是原地重试——**坏任务卡死整条链**是多 Agent 系统最常见的死法。

> ☕ Java 视角：把 `supervisor_loop` 里的元素换名：Task=工作项，Worker=线程池里的工人，max_turns=超时熔断，"无匹配工人标记失败"=死信队列（DLQ）。你会发现多 Agent 编排问题几乎可以逐条翻译成你做过的分布式任务调度问题——所以别被新名词唬住。

## 练习

1. 跑通上面的模拟脚本，然后给它加一个"审核员" worker（技能：审），让"写一份竞品小结"完成后自动追加一个审核任务。
   提示：审核任务的追加发生在 worker.work 汇报之后——想想改 supervisor_loop 还是用一个"任务生成"环节。
2. 把 `Task` 扩展两个字段：`evidence: list[str]`（依据）和 `confidence: float`（置信度），并在最终汇总里打印低置信度任务。
   提示：这正是"结构化消息防级联幻觉"的最小实现。
3. 挑一个你熟悉的业务（如工单处理），分别画出路由版和监督者版的结构图，标注每条边上传递的是"产物"还是"全量历史"。
   提示：数一数每条边的预估 token 量级，对比两种模式的总成本。
4. 思考题：辩论模式为什么只适合"高价值低容错"问题？用 token 成本公式说明。
   提示：N 个辩手 × R 轮批驳，token 是乘法关系。

## 小结与自测

- Q1：需要多 Agent 的两类真实需求？
  一句话：上下文隔离（单窗口装不下且互相干扰）与异质职责（提示词/工具/权限差异大到必须分角色）。
- Q2：四种模式里哪两种 agent 之间不互相通信？
  一句话：路由（专家互不通信）与流水线（只与相邻环节传产物）。
- Q3：性价比最高的通信成本控制手段？
  一句话：传产物不传历史——下游只拿结构化结论，不转发完整对话。
- Q4：监督者模式最常见的死法是什么？怎么防？
  一句话：调度循环不收敛（坏任务反复派发）；用硬预算上限 + 失败标记（不重试无解任务）。
- Q5：为什么说多 Agent 是分布式系统问题？
  一句话：通信成本、调试链路、级联失败三大成本全部复活，只是计量单位变成 token。

## 下一课预告

第 6 课收官本阶段：自研还是框架？一套可打分的决策框架，以及我给你的明确建议——不带"看情况"式的和稀泥。
