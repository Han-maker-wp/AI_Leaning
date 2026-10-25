# 第 1 课：手写 Function Calling——掀开你用过无数次的底牌

> 🎬 开场钩子：2023 年 6 月之前，想让模型调工具，开发者只能在 prompt 里贴一份 JSON Schema，恳求模型"请按这个格式输出你要调的函数"——然后自己解析那段大概率不合法的文本。2023 年 6 月，OpenAI 正式推出 function calling，"恳求"变成"协议"。你作为用户按过几百次的那个魔法按钮，今天要亲手写出它的全部内部机制。

⏱ 建议学时：2.5 小时（本课是全轨道地基中的地基，`code/L01_fc_loop.py` 必须逐行读懂）

## 本课目标

1. 纠正最大的误解：模型从不"执行"工具——说清它真正做的是什么
2. 会写 tool schema（JSON Schema 子集），理解 description 为什么是给模型的接口文档
3. 徒手写出完整调用循环：模型请求 → 代码执行 → 结果回填 → 终止判断
4. 能默画消息流转图，并解释 tool_call_id 的配对作用

## 1.1 先拆穿魔法：模型到底做了什么

你在产品里点过"联网搜索"，感觉是模型去搜了。真相分三步：

1. 你的代码在请求里**声明**可用的工具（名字、描述、参数 schema）
2. 模型**不执行任何东西**，它输出的是一段结构化意图："我想调 `search_web`，参数 `{"query": "..."}`"——这个能力是训练出来的，不是它长了手
3. **你的代码**解析这个意图、真正执行函数、把结果作为新消息回填，再次请求模型——它读到结果后继续生成

一句话：**模型负责决策，你的代码负责执行，消息列表负责传话**。这个分工不是技术细节，是安全模型的根基——"模型无法越过你的代码做任何事"，阶段 1 L05 的全部安全设计都站在这一条上。

> ☕ Java 视角：把模型想成一个只返回"调用指令 DTO"的远程服务：它吐 `ToolCallRequest{name, argsJson}`，你本地 `dispatch` 执行后把返回值塞回请求上下文再调它。整个 function calling 就是**你手写的一个 RPC 编排循环**，而"接口文档"从给前端看的 Swagger 变成了给模型看的 tool schema。

## 1.2 tool schema：给模型看的接口文档

声明工具用的是 JSON Schema 子集（以下为 chat 接口通用形态，字段细节以 platform.openai.com/docs 为准）：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前天气。仅支持中国城市；温度单位为摄氏度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如「北京」"},
                    "unit": {"type": "string", "enum": ["C"], "description": "温度单位，固定摄氏度"},
                },
                "required": ["city"],
            },
        },
    }
]
```

三个设计要点，每条都直接影响模型调用准确率：

- **description 是给模型读的**，不是给人看的注释。写清"何时该用"（查询当前天气）、范围（仅中国城市）、单位（摄氏度）——模型选错工具，八成是描述含糊
- **枚举优于自由字符串**：`enum: ["C"]` 让"unit"不可能填出华氏度。能用类型系统挡住的错误，就别指望模型自觉
- **必填参数最小化**：每个必填参数都是模型必须推理正确的点，少一个就少一分出错概率

请求时把 `tools` 传进去（`tool_choice` 默认 `"auto"` 即模型自主决定是否调用）。这些内容在 L02 会展开成完整的"工具设计学"。

## 1.3 完整循环：消息流转图

先看全景（这张图是本阶段最值得抄进笔记的东西）：

```
你                         API（模型）
│  messages + tools ──────────────▶│
│                                  │ 决定调工具：返回 assistant 消息，
│  ◀────── tool_calls: [id=call_1, │ 其中 tool_calls = [{id, name, arguments}]
│          get_weather, {...}]     │ （此时还没有最终回答）
│ 执行 get_weather（你的代码！）    │
│  messages += [assistant(含tool_calls),
│               tool(call_1, 结果)]│
│  messages 再发 ─────────────────▶│
│  ◀────── 最终自然语言回答 ────────│ （不再包含 tool_calls，循环结束）
```

对应的代码骨架（`code/L01_fc_loop.py` 为完整可跑版）：

```python
messages = [{"role": "user", "content": "北京今天多少度？"}]
for _ in range(MAX_ROUNDS):                     # 循环 + 轮数上限
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto")
    msg = resp.choices[0].message
    if not msg.tool_calls:                      # 模型不再要工具 → 终止
        print(msg.content)
        break
    messages.append(msg)                        # ① 回填 assistant 消息（含 tool_calls）
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)  # ② arguments 是 JSON 字符串！
        result = dispatch(tc.function.name, args)  # ③ 你的代码真正执行
        messages.append({                       # ④ 结果以 role=tool 回填，靠 id 配对
            "role": "tool", "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False)})
```

四个新手必踩的坑，提前打预防针：

1. **忘记第①步回填 assistant 消息** → API 直接报错：tool 消息必须跟在带对应 tool_calls 的 assistant 消息后面。配对靠 `tool_call_id`，就像 HTTP 请求与响应用序列号对应
2. **`arguments` 是字符串不是 dict**——模型输出的参数是序列化好的 JSON 文本，必须 `json.loads`（而且要按 L04 的容错思路处理它可能的格式伤）
3. **工具异常直接抛** → 整个循环崩掉。工具的错误要转成文本结果回填（L02 主题）
4. **没有轮数上限** → 模型陷入"调工具-不满意-再调"的循环，token 烧成无底洞（L03 详攻终止条件）

> 🍿 彩蛋｜还有另一条路：Meta 2023 年的 Toolformer（arXiv 2302.04761）展示了让模型**自己学会**什么时候调工具——把"何时调用"内化进模型参数。我们走的是工程路线：行为由外部协议约定，随时可改；两条路线并不对立，今天的 Agent 产品基本都是前者，后者在研究侧持续演进。

## 1.4 跑通 code/L01_fc_loop.py

配套脚本把上述循环做成**离线可跑版**：模型用条件分支模拟（看消息历史，先返回"我要调 get_weather"的 tool_calls，看到 tool 结果后返回最终回答），协议数据结构则与真实 API 完全同构——`tools` 列表、`tool_calls`、`tool_call_id`、`role="tool"` 一应俱全。跑通后做两个实验：

1. 在 `dispatch` 里故意 `raise`，观察循环如何崩——体会"工具异常必须转文本"的动机
2. 把 mock 的 `tool_calls` 返回改成两次工具请求，观察循环自然转到第二轮

然后把 mock 函数换成真客户端（脚本里留了切换开关的注释位）：协议零改动，这正是 mock 与真实同构的意义——**考试考的是循环怎么写，不是模型答什么**。

补充第三个实验：把用户问题改成"上海呢？"再跑——mock 直接失效（它的剧本只认北京）。这不是 bug，是边界认知：真模型的"通用决策能力"才是它区别于 if/else 的地方；但**协议层的一切（循环、回填、id 配对、终止）与模型能力无关**——所以 mock 才能教会你全部机制，换真模型只是把"决策函数"升级。

## 1.5 消息类型对照表：一次完整工具调用的 messages

把整轮对话的消息序列列全，调试时逐条对照（这是排查"循环怎么不走了"的地图）：

| 序 | role | 关键字段 | 谁产生 | 说明 |
|---|---|---|---|---|
| 1 | system | content | 你的代码 | 人设与工具使用约束 |
| 2 | user | content | 用户 | "北京今天多少度？" |
| 3 | assistant | tool_calls=[{id, name, arguments}] | 模型 | 调用请求；content 可能为空 |
| 4 | tool | tool_call_id, content=结果 JSON | 你的代码 | 与序 3 的 id 配对 |
| 5 | assistant | content | 模型 | 看到结果后的最终回答 |

多轮工具调用就是序 3-4 重复 N 次再接序 5。两个立刻能用的调试技巧：① 每轮循环结束打印 `len(messages)` 与最后一条的 role，肉眼确认序列形状；② 出错时把完整 messages 落盘成 JSON——90% 的"循环诡异行为"都是消息序列形状不对（漏回填、id 不配、顺序颠倒），看一眼序列就能定位。

> ☕ Java 视角：这张表就是你的"协议报文格式说明"。和排查 RPC 问题先抓包看报文一样，Agent 排障第一动作永远是 dump messages——它是循环的全部状态。

## 练习

1. 给脚本加第二个工具 `calculate(a, b, op)`（含完整 schema 与 dispatch 分支），改造 mock 触发"先查天气再算温差"的链式调用。
   提示：mock 按消息历史分支——上一条是用户就调天气，看到天气结果就调计算器，再看到计算结果才作答。
2. 故意删掉"回填 assistant 消息"那行，运行并记录报错；再把 `tool_call_id` 改错一位，观察配对失败的表现。
   提示：两种错误报在不同环节——一个是请求被拒，一个是结果对不上号。
3. 写一个"循环安全阀"：工具调用总次数超过 N 就强制终止并告知用户。
   提示：在 for 循环外维护计数器，终止时用一条友好消息 break，不要裸退出。

## 小结与自测

- 问：模型会执行工具吗？答：不会，它只输出结构化调用意图；执行永远在你的代码里，这是安全的根基。
- 问：description 为什么重要？答：它是模型选择与调用工具的唯一依据，含糊的描述等于含糊的接口文档。
- 问：tool_call_id 起什么作用？答：把每条 tool 结果消息与它对应的调用请求配对，多工具并行时缺它即乱。
- 问：arguments 需要什么处理？答：它是 JSON 字符串，必须 json.loads，且要防格式伤（L04 的容错思路）。
- 问：循环什么时候终止？答：模型返回的 assistant 消息不再包含 tool_calls（正常完成），或触发轮数上限（安全阀）。

## 下一课预告

第 2 课专门研究"工具怎么写"：参数怎么设计、错误为什么返回文本、幂等与超时怎么保——工具是给模型用的 API，用户从人换成了模型，设计哲学随之全部刷新。
