# 第 3 课：多工具编排与 ReAct——五十行循环撑起的"自主性"

> 🎬 开场钩子：2023 年 3 月，AutoGPT 开源后一度是 GitHub 增长最快的项目之一，人人都在谈"自主 Agent"；同年 4 月，Yohei Nakajima 把自己的周末项目 BabyAGI 发上网，又引燃一轮狂欢。营销词是"自主智能体"，掀开看：核心都是今天这节课要写的那几十行 while 循环。所谓自主，不过是"循环 + 工具 + 终止条件"的工程组合——你今天就能写出来。

⏱ 建议学时：2.5 小时（`code/L03_mini_agent.py` 是本阶段 checkpoint 的起点，务必吃透）

## 本课目标

1. 用注册表模式管理多个工具（schema 与实现单一事实源）
2. 处理一次响应里的多个 tool_calls（并行执行的条件与做法）
3. 说出三种终止条件并理解各自的必要性，写出一个不会失控的循环
4. 讲清 ReAct 模式（arXiv 2210.03629）与原生 function calling 的关系

## 3.1 工具注册表：让工具可插拔

单工具时 if/else 还能凑合，三个以上就该上注册表——工具名到实现的映射，加上 schema 列表自动生成：

```python
TOOL_SCHEMAS: list[dict] = [...]        # 1.2 课那样手写，或从注册项自动生成
TOOL_IMPLS: dict[str, callable] = {     # name → 实现，与 schema 一一对应
    "get_weather": get_weather,
    "calculate": calculate,
    "read_file": read_file,
}

def dispatch(name: str, args: dict) -> str:
    func = TOOL_IMPLS.get(name)
    if func is None:
        return tool_error("UNKNOWN_TOOL", f"未注册的工具: {name}")
    return safe_call(name, func, args)   # L02 的超时+错误包装+日志
```

好处不只是少写 if：**加工具 = 注册一项，循环零改动**——这个"开放扩展"的性质正是 Agent 生态的发动机（阶段 3 L04 的 MCP 协议，本质上就是把这张注册表标准化到跨进程）。

但注册表不是越大越好：**工具越多，选择越难**。一堆 description 语义相近的工具（search_docs / search_wiki / search_web），模型选错率随工具数上升。工程直觉：一个 Agent 常驻 5-10 个职责分明的工具；更多就该拆职责（分多个 Agent 或加"工具路由"），这是阶段 3 多 Agent 的动机之一。

## 3.2 并行调用：一次响应里的多个 tool_calls

模型可以在**一次响应里请求多个工具调用**（`msg.tool_calls` 是列表）。互不依赖的调用（查两个城市天气、读三个文件）就该并行执行，把 L05 的 asyncio 直接用进来：

```python
coros = [run_tool_async(tc) for tc in msg.tool_calls]   # 每个调用一个协程
results = await asyncio.gather(*coros)                  # 并行执行，结果顺序与请求一致
for tc, result in zip(msg.tool_calls, results):
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

两个纪律：① 并行的前提是**互不依赖**——有依赖时模型自然会拆成多轮（先读文件拿到数字，再算），你不用替它操心；② 并行也要过 Semaphore——L05 的教训在工具层同样成立，你的下游（数据库/外部 API）扛不住几十路齐发。

## 3.3 终止条件：自主性的缰绳

循环骨架定型：

```python
for round_no in range(1, MAX_ROUNDS + 1):      # 条件一：最大轮数
    resp = call_model(messages, TOOL_SCHEMAS)
    msg = resp.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:                     # 条件二：模型自然收尾
        return msg.content                     # 不再要工具 = 任务完成
    execute_and_backfill(msg.tool_calls)       # 执行并回填（含并行/错误文本化）
    if user_cancelled():                       # 条件三：外部中断
        return "已按用户要求停止"
return fail_safe_message()                     # 轮数耗尽：强制收敛而非裸崩
```

三个终止条件各有不可替代性：

- **自然收尾**（无 tool_calls）：正常出口，模型认为任务完成
- **最大轮数**：缰绳。模型可能陷入"再搜一次-还不满意-再搜"的循环，没有上限，token 烧穿预算；上限触发时不要裸退出，给用户一句诚实的"我在 N 轮内没能完成，已完成的部分是……"
- **外部中断**：用户喊停、超时、上游取消——自主系统必须有"急停按钮"，这是 Agent 与批处理任务的重要差别

> 🍿 彩蛋｜ReAct：Princeton 的 Yao 等人 2022 年发表的 ReAct（arXiv 2210.03629）把"推理"与"行动"交替写进同一个循环：Thought（我该查一下天气）→ Action（调工具）→ Observation（读到结果）→ Thought（现在可以回答了）→ Answer。当时没有原生 function calling，Action 就是让模型按约定格式输出一行文本、自己正则解析。原生协议出现后，"行动"那半已经被 tool_calls 标准化，但"先写显式推理再决定动作"的思想仍是提准确率的常用手段——你完全可以在 system prompt 里要求模型调工具前先说明理由，这就是 ReAct 的现代用法。

## 3.4 跑通 code/L03_mini_agent.py

配套脚本是一个**离线可跑的完整迷你 Agent**：计算器 + 文件读/写三个工具、注册表分发、错误文本化、轮数上限一应俱全；模型仍用条件分支模拟，剧本是"读 data.txt 里的两个数 → 算乘积 → 把结果写进 result.txt"。读代码时盯住三件事：

1. mock 模型如何**按消息历史决定下一步**——它"思考"的依据只有 messages，这正是真模型的工作方式
2. 每轮循环 messages 如何生长（assistant 含 tool_calls → tool 结果 → assistant……）
3. `MAX_ROUNDS` 触发时的降级输出

这个文件就是你 checkpoint 的起点：把 mock 换成真客户端、按 README 加上记忆与安全边界，迷你 Agent 就长成了你的毕业级玩具。

> ☕ Java 视角：整个循环像一个**事件驱动的调度器**：模型是决策源（相当于规则引擎），注册表是 handler 映射（相当于 Spring 的 handler mapping），messages 是共享状态（相当于会话上下文）。你写过的每个"接收请求-分发-处理-回写"的系统，结构上都是它的亲戚。

## 3.5 原生 function calling 与 ReAct 文本协议对照

同一段任务两种实现方式，差异值得列表记牢（阶段 3 换框架时你会再次需要它）：

| 维度 | ReAct 文本协议（2022 原版） | 原生 function calling |
|---|---|---|
| 行动表达 | 模型输出 `Action: search[北京天气]` 文本，你正则解析 | 模型输出结构化 tool_calls，SDK 直接给对象 |
| 可靠性 | 看模型心情：格式跑偏、漏 Action、混进闲话 | 协议保证：解析零成本 |
| 推理表达 | Thought 显式写在循环里 | 默认隐藏；可要求"先说明理由再调用"叠加回来 |
| 循环控制 | 你解析文本决定下一步 | 判 `tool_calls` 是否为空 |
| 适用 | 不支持 FC 的模型/极简场景 | 默认选择 |

结论一句话：**行动层用原生协议，推理层按需借 ReAct 的思想**——在 system prompt 里要求"调用工具前先一句话说明你的判断依据"，你就同时拿到了协议可靠性与推理透明度，这也是生产系统的常见配置。

## 3.6 循环之外：从 demo 到产品还缺什么

手写循环解决的是核心 30%，跑通 mini_agent 后你应该已经感觉到剩下 70% 的轮廓：

1. **流式与工具交错**：模型边说"我来查一下"边发工具请求，文本与 tool_calls 在流式里交错到达——解析比非流式复杂一档
2. **状态与恢复**：进程重启、用户隔天回来接着聊（L04 的持久化只是第一步）
3. **可观测**：一次回答 = N 次模型调用 + M 次工具执行，没有调用链追踪，线上排障靠猜
4. **成本护栏**：单会话 token 上限、单用户日配额——循环是能烧钱的
5. **并发会话**：多个用户同时用，状态隔离与资源竞争全要管

这五件事正是阶段 3（框架封装的对象）与阶段 4（工程化主题）的课表来源——你现在有资格说"我知道框架在帮我做什么"，因为你都亲手写过裸的版本。

## 3.7 从需求到工具清单：设计一个客服 Agent

用一张表走完"需求 → 工具 → 风险标签"的设计流（这就是 checkpoint 与毕业项目的第一步）：

| 用户需求 | 工具 | 副作用标签 | 设计要点 |
|---|---|---|---|
| "查一下我的订单到哪了" | query_order(order_id) | read | 参数校验格式；查不到返回带 hint 的错误 |
| "帮我取消订单" | cancel_order(order_id) | **dangerous** | 不可逆 → 人工确认 + 幂等键 |
| "发票怎么开" | search_kb(关键词) | read | 其实是 RAG 检索工具（阶段 2 主题） |
| "把这单的明细发我邮箱" | send_email(to, subject, body) | **dangerous** | 外发动作 → 确认；收件人白名单 |
| "算一下退款金额差" | calculate(...) | read | AST 白名单版（L05） |

三个设计观察：① read 类占大头——**能设计成只读就设计成只读**，dangerous 越少越安全；② 每个 dangerous 都天然需要 L02 的幂等与确认机制，标签驱动了工程要求；③ `search_kb` 悄悄预告了 RAG——检索器在 Agent 视角里"只是一个工具"，阶段 2 结束后你就能把它装进来。

多轮链路的预期消息形状（mini_agent 剧本的序列，对着 1.5 课的对照表读）：

```
user: 帮我算 data.txt 里两个数的乘积写到 result.txt
assistant: [tool_calls] read_file(data.txt)
tool: (文件内容)
assistant: [tool_calls] calculate(12, 34, mul)
tool: (408)
assistant: [tool_calls] write_file(result.txt, "408")
tool: (写入成功)
assistant: 已完成，结果是 408，已写入 result.txt。
```

## 练习

1. 给 mini_agent 加第 4 个工具 `list_dir()`，并改写 mock 剧本：先列目录确认文件名，再读取，再计算。
   提示：mock 按"是否已有 list 结果/文件内容/计算结果"三段分支。
2. 把 MAX_ROUNDS 从 5 降到 2，构造一个必然超限的剧本（mock 永远要求调工具），观察降级输出；再思考：降级消息里应该包含什么才算"诚实"？
   提示：已完成的部分、卡在哪一步、用户可以怎么办。
3. 纯 prompt 复刻 ReAct：不用 tools 参数，让 mock 模型输出 `Thought: .../Action: search[北京天气]/Observation: ...` 文本，你写正则解析并推进循环。
   提示：对比两版代码量与健壮性，体会"原生协议把什么标准化了"。

## 小结与自测

- 问：注册表的本质好处？答：加工具零改循环（开闭原则），schema 与实现单一事实源，天然支持跨进程标准化（MCP 的雏形）。
- 问：并行执行的前提？答：一次响应里的多个 tool_calls 互不依赖，且经过并发闸门（Semaphore）。
- 问：三种终止条件？答：模型自然收尾（无 tool_calls）、最大轮数上限（防失控烧钱）、外部中断（用户/超时）。
- 问：ReAct 与原生 function calling 的关系？答：ReAct 定义了推理+行动交替的思想（2022，文本协议实现）；原生协议把"行动"标准化，"显式推理"仍是叠加使用的技巧。
- 问：工具太多会怎样？答：语义相近的工具互相干扰，选择准确率下降；常驻 5-10 个职责分明的工具，更多就拆分。

## 下一课预告

第 4 课解决 Agent 的"金鱼记忆"：对话为什么越聊越贵、窗口怎么管理、摘要怎么压缩、会话怎么用 SQLite 持久化——重启不失忆的 Agent 才算软件，不然只是彩排。
