# 第 2 课：开源模型生态与 Hugging Face Hub
> 🎬 开场钩子：在 HF Hub 上搜一个任务，能蹦出几万个模型，像逛应用商店——但货架上没有评分、一半是空壳。今天教你一套「挑模型」的鉴宝功夫：读模型卡、看参数量、查 license，十分钟内判断一个模型值不值得下载。


> ⏱ 建议学时：1.5~2 小时（含下载实操）

## 本课目标

1. 能说出 Llama / Qwen / DeepSeek / Mistral 四个主流开源家族的定位与差异，选型时不再纠结。
2. 拿到任意一张 HF 模型卡，能在 5 分钟内提取出决策关键字段（参数量/上下文/license/base vs instruct）。
3. 能按「任务 → 显存 → license」三步为给定场景选出合适的模型。
4. 能用三种方式把模型下载到本地，并配置国内镜像加速。
5. 能区分 safetensors / pytorch_model.bin / GGUF 三种权重格式及其适用生态。

## 1. 为什么开源生态是工程师的红利

2023 年之前，「能微调的大模型」只存在于大厂内部。如今：Llama 引爆开源生态，Qwen/DeepSeek
等中国团队把高质量开放权重模型做到了世界第一梯队。对你的直接意义：

- **可控**：权重在你手里，私有部署、离线运行、数据不出域——这在企业场景经常是硬需求。
- **可改**：可以微调（阶段 5 的全部内容）。
- **可比**：开源社区把推理引擎（vLLM）、量化（GGUF/AWQ）、微调工具（LLaMA-Factory）全部
  标准化了，你会一个家族就会全部。

> 注意：开源 ≠ 随便商用。每个家族的 License 条款不同且随版本变化，商用前必查模型卡。

## 2. 主流家族谱系概览

家族与版本迭代很快，下表是「定位直觉」，具体型号以各官方发布页为准。

| 家族 | 出品方 | 一句话定位 | 适合你的点 |
|---|---|---|---|
| Llama | Meta | 开源生态的标杆与「事实标准」，衍生模型最多 | 生态工具最全，教程最多；注意其社区 License 有附加条款 |
| Qwen | 阿里 | 尺寸谱系最全（0.5B~72B），中文能力第一梯队 | **本课程主力**：小尺寸适合实验，中文数据微调效果好 |
| DeepSeek | 深度求索 | 推理与代码能力突出，MoE 架构激进 | R1 系列展示了对齐（推理链）路线的另一种可能 |
| Mistral | Mistral AI（法国） | 「小而精」的欧洲代表，7B 级别成名 | 证明小模型工程优化能打大模型 |

其他你会在社区频繁见到的真实家族：GLM（智谱）、Yi（零一万物）、InternLM（书生）等，
选型思路同本课，不逐一展开。

选型第一直觉（后面反复用）：

- **实验/教学**：Qwen2.5-0.5B / 1.5B（CPU 也能跑，快速迭代）
- **单卡微调主力**：Qwen2.5-7B-Instruct（阶段 5 默认底座）
- **更强效果**：14B / 32B（推理为主，微调显存门槛高）

## 3. 模型卡怎么读（5 分钟提取法）

以 `Qwen/Qwen2.5-7B-Instruct` 的模型卡为例，按顺序抓这六个字段：

### 3.1 参数量与命名

`7B` = 70 亿参数。命名规律：`家族-版本-尺寸[-变体]`。同一个尺寸通常有 base 与
Instruct（chat）两版（见 L01 第 6 节的行为差异）。还有 Code/Math 等专精变体。

### 3.2 上下文长度（Context Length）

模型一次能「看到」的最大 token 数。例如 Qwen2.5-7B-Instruct 支持 128K（以模型卡为准）。
读卡注意两点：

- 「支持 128K」有时依赖 YaRN 等外推技术，原始训练长度可能更短，长文本效果打折。
- 上下文越长，KV cache 显存越大（L03/L04 详述），不是免费的。

### 3.3 License

模型卡右侧栏的 license 标签。常见三档：

- **Apache 2.0 / MIT**：最宽松，可商用可修改（Qwen 多数尺寸为此类，个别型号有附加条款，以卡为准）
- **社区许可**（如 Llama 系）：基本可商用，但有附加条款（如用户规模门槛、命名要求）
- **研究/受限**：只能研究用

### 3.4 base vs instruct

模型名带 `Instruct`/`Chat`/`iT` 等后缀 = 对齐过的对话模型；无后缀 = base 补全模型。
**阶段 5 微调一律选 Instruct 版作底座**（在会聊天的模型上继续教领域行为，比教 base 聊天便宜得多）。

### 3.5 知识截止时间（Knowledge Cutoff）

训练语料截止日期，决定了模型「天生」知道什么。之后的知识要么靠 RAG，要么靠你的微调数据。

### 3.6 Benchmark 表与 Chat Template

模型卡中的评测表用于横向比较（注意各家的评测设置未必一致，看相对位置而非绝对分数）；
仓库内的 `tokenizer_config.json` 里能看到对话模板（L03 详述）。

## 4. 如何选模型：三步决策法

```
第一步：任务定「能力档位」
  只推理/助手 → 7B~14B 常够用；复杂推理/代码 → 14B+ 或专精变体
第二步：显存定「精度与尺寸」（用 L04 公式口算）
  例子：24GB 卡 → 7B fp16 微调(QLoRA 富余) / 14B int4 推理
  8GB 卡 → 7B int4 推理 或 1.5B QLoRA 微调
第三步：license 与语言定「家族」
  商用 → 查 license；中文为主 → Qwen 优先
```

一个常被忽视的点：**微调选型时，底座模型「原本就会什么」几乎决定上限**。微调教行为，不注入
大量新知识（L07 会再对比 RAG）。给 Java 代码助手选底座，就选代码能力本来就强的。

## 5. HF Hub 下载：三种方式 + 国内镜像

Hugging Face Hub 是模型界的 Maven Central。一个模型仓库 = 一个目录（config + 权重 + tokenizer）。

### 5.1 方式一：transformers 自动下载（最常用）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# 首次执行自动下载到 ~/.cache/huggingface，再次执行直接用缓存
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto", device_map="auto")
```

### 5.2 方式二：huggingface-cli 手动下载（推荐，可控可续传）

```bash
# 命令以 huggingface.co 官方文档为准；新版 CLI 也提供 `hf download` 等价命令
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen2.5-1.5b-instruct
```

支持断点续传；大模型仓库靠 git-lfs 承载大文件（CLI 已替你处理好，无需手动操作 git）。

### 5.3 方式三：Python API 精细控制

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct", local_dir="./models/qwen2.5-1.5b-instruct")
```

### 5.4 国内镜像加速（关键工程技巧）

国内直连 HF 经常超时。设置环境变量走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 社区镜像，长期维护
# 建议写进 ~/.bashrc；上面的三种下载方式都会自动走该镜像
```

离线环境（如内网训练机）可提前在有网机器下载，`scp` 整个目录过去，加载时直接传本地路径。
HF 还提供 `HF_HUB_OFFLINE=1` 等环境变量控制离线行为（以官方文档为准）。

## 6. 权重格式概念

下载一个模型仓库，你会看到这些文件——它们是什么、什么关系：

```
Qwen2.5-1.5B-Instruct/
├── config.json               # 模型结构超参（层数/隐藏维度/头数…），加载时按它搭网络
├── tokenizer.json            # 分词器（文本 ↔ token id 的映射表与算法）
├── tokenizer_config.json     # 含 chat template 定义
├── *.safetensors             # 权重本体（可能分片 + 一个 index json 指明分片映射）
└── generation_config.json    # 生成默认参数（temperature 等）
```

三种权重格式的区别（概念级即可，L04 再谈量化细节）：

| 格式 | 本质 | 生态 | 一句话 |
|---|---|---|---|
| `pytorch_model.bin` | PyTorch pickle 序列化 | 旧格式 | 有任意代码执行的安全风险，正被淘汰 |
| `*.safetensors` | 只存张量的安全格式 | HF 生态主流 | 无代码执行风险、mmap 加载快，**下载时优先选它** |
| `*.gguf` | llama.cpp 单文件容器 | Ollama / llama.cpp | 单文件含权重+元数据，通常已量化，本地部署用（L05） |

同一模型常同时提供 safetensors（训练/transformers 推理）与 GGUF（本地工具）两套，按下游工具选。

> 🍿 彩蛋｜Hugging Face 的拥抱：这家公司的 logo 是一个 🤗，名字来自一个滑板术语。它最初想做聊天机器人，后来转型成了「AI 界的 GitHub」——今天去它那下载模型，就像当年第一次用 Maven Central 拉依赖一样自然。

## 练习

1. 打开 `Qwen/Qwen2.5-7B-Instruct` 模型卡，按 3.1~3.6 的清单逐字段记录，写成你自己的「读卡笔记」。
   提示：右侧栏 license、简介中的 context length、Files 里的文件清单三处必看。
2. 用 huggingface-cli + HF_ENDPOINT 镜像把 `Qwen/Qwen2.5-0.5B-Instruct` 下载到本地，记录下载
   总大小，与 L04 的「参数量 × 每参数字节」公式对一下账（该模型为 bf16 存储）。
   提示：0.5B ≈ 4.9 亿参数 × 2 字节 ≈ 1GB 量级，看实际数字差多少并想原因。
3. 场景判断：公司内网要做一个中文合同问答助手，数据绝不能出域，预算一张 24GB 卡。选哪个家族
   哪个尺寸哪个精度的模型？写出三步决策过程。
   提示：内网→必须自托管；24GB→7B fp16 或 14B 量化；中文→Qwen。
4. 打开一个 GGUF 格式的模型仓库（在 HF 上搜 gguf），对比它与 safetensors 仓库的文件列表差异。
   提示：GGUF 通常是单个大文件加量化后缀（如 Q4_K_M）。

## 小结与自测

1. **Q：为什么阶段 5 微调底座选 Instruct 版而不是 base 版？**
   A：Instruct 已会对话与听从指令，微调只需教「领域行为」；从 base 开始等于把 SFT 全流程重做，贵且难。
2. **Q：模型卡上 128K 上下文，意味着部署时就能随便用 128K 吗？**
   A：不能直接等同：长上下文可能靠外推技术、效果打折，且 KV cache 显存随长度线性增长，要算账。
3. **Q：HF_ENDPOINT 环境变量的作用是什么？**
   A：把 transformers/huggingface_hub 的下载请求指向国内镜像站，解决直连超时。
4. **Q：为什么 safetensors 比 pytorch_model.bin 安全？**
   A：pickle 反序列化可执行任意代码，safetensors 只描述张量数据、不含可执行逻辑。
5. **Q：GGUF 和 safetensors 各服务什么场景？**
   A：safetensors 服务 transformers 训练/推理生态；GGUF 服务 llama.cpp/Ollama 本地部署生态（常已量化）。

## 下一课预告

下一课拆开推理的黑盒：自回归循环一次生成一个 token，KV Cache 为什么能省下海量重复计算，
temperature/top-k/top-p 这些你调过的参数背后到底在操作什么分布。
