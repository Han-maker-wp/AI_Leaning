# 第 5 课：QLoRA 实操全流程（第一次微调你的 LLM）
> 🎬 开场钩子：2023 年之前，「一张显卡微调 650 亿参数」是段子；QLoRA 论文把它变成了现实。今天你动手的是同一技术路线的迷你版：小模型、几百条数据、几小时算力——但从今天起，「我微调过大模型」这句话你可以平静地说出来。


> ⏱ 建议学时：4~6 小时（含训练等待时间；本课可照做，命令均需以 LLaMA-Factory 官方 README 为准）
> 配套速查：`code/llamafactory_train_steps.md`（全流程命令速查）

## 本课目标

1. 完成人生第一次 LLM 微调：从模型选择到对话验证的全流程闭环。
2. 会用 LLaMA-Factory 注册自定义数据集（dataset_info.json）。
3. 能读懂并修改 QLoRA 训练配置中的关键参数（cutoff_len/epoch/lr/lora_rank/lora_target）。
4. 会看训练监控（loss 曲线看什么）并完成导出合并与快速验证。
5. 掌握云 GPU 场景的完整操作链（租卡→上传→训练→下载权重）。

## 0. 实操路线设计

两条腿走路：

- **本地/CPU 或小卡**：用 Qwen2.5-0.5B/1.5B + 50 条数据空跑全流程——目的是**走通每个环节**，
  不追求效果（1.5B QLoRA 在 12GB 级即可跑，0.5B 更宽松）。
- **云 GPU（4090 级）**：Qwen2.5-7B-Instruct + 你的毕业项目数据，产出真正可用的模型。

先小后大、先通后优——租着云卡调试脚本是烧钱行为（阶段 4 L06 的 checklist 第 6 条）。

## 1. 选模型

| 圕景 | 推荐 | 理由 |
|---|---|---|
| 空跑流程 | `Qwen/Qwen2.5-0.5B-Instruct` | CPU/入门卡都能训完，几分钟一轮 |
| 毕业项目 | `Qwen/Qwen2.5-7B-Instruct` | 单卡 QLoRA 的甜蜜点，中文强（L02 选型逻辑） |

统一用 **Instruct 版**（阶段 4 L01：在会聊天的模型上教领域行为，最省数据）。

## 2. 准备数据

用 L02 的成果（alpaca 格式 json）。本课先用配套样例数据走流程：

```
05_finetuning/code/alpaca_sample.json     # 含瑕疵样例——实操前先跑 data_clean.py 清洗！
```

把清洗后的文件放入 LLaMA-Factory 的数据目录（源码安装时在仓库 `data/` 下）：

```bash
cp cleaned.json /path/to/LLaMA-Factory/data/java_assistant.json
```

## 3. 注册数据集：dataset_info.json

LLaMA-Factory 通过 `data/dataset_info.json` 认识你的数据集。在 json 中追加一项
（alpaca 格式的典型写法，字段说明详见官方 README 的数据集章节）：

```json
"java_assistant": {
  "file_name": "java_assistant.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

要点：

- 键名（`java_assistant`）即后续命令里 `dataset` 参数的值
- `columns` 把你的字段映射到框架的语义槽位：`prompt`=指令、`query`=补充输入、`response`=答案
  （sharegpt 格式则映射 `messages` 等字段，写法以官方文档为准）
- 改完注册文件不需要重启任何服务

## 4. 训练

### 4.1 路线 A：WebUI（第一次强烈推荐）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
llamafactory-cli webui                     # 浏览器打开提示的地址（默认 7860 端口）
```

在界面中按下面参数表逐项选择/填写，点开始训练。WebUI 里选好的配置可以导出为 yaml，
正好作为路线 B 的起点。

### 4.2 路线 B：CLI + yaml（正式交付形态）

```yaml
# qwen25_7b_qlora.yaml —— 典型示例；参数全集以 LLaMA-Factory 官方 README 为准
model_name_or_path: Qwen/Qwen2.5-7B-Instruct   # 或本地路径
trust_remote_code: true

stage: sft                # 指令微调阶段
do_train: true
finetuning_type: lora     # LoRA 家族
lora_rank: 8              # 秩 r（L01/L03：小 r 足够）
lora_alpha: 16            # alpha = 2r（L01 缩放约定）
lora_target: all          # 挂所有线性层（L03 结论 2 的实践）
quantization_bit: 4       # 关键一行：加上它 = QLoRA（NF4 量化冻结主干）

dataset: java_assistant   # dataset_info.json 里注册的名字
template: qwen            # chat template 名，以官方模型支持表为准——用错模板=灾难（阶段4 L03）
cutoff_len: 2048          # 样本截断长度，须 ≥ 你的数据实际长度分布（L06 详述）
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: saves/qwen25-7b-java-qlora
per_device_train_batch_size: 1
gradient_accumulation_steps: 8    # 等效 batch=8（L06 详述）
learning_rate: 1.0e-4             # LoRA 典型 1e-4~2e-4（L06 详述）
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true                        # 显卡支持才开（阶段4 L04）
logging_steps: 10
save_steps: 500
plot_loss: true                   # 自动绘制 loss 曲线图
```

```bash
llamafactory-cli train qwen25_7b_qlora.yaml
```

### 4.3 关键参数回顾（每个都要能说出「为什么」）

| 参数 | 本课取值 | 为什么（对应理论课） |
|---|---|---|
| `quantization_bit: 4` | QLoRA 的开关 | NF4 冻结主干，显存 14GB→3.5GB（L01 §4） |
| `lora_rank / alpha` | 8 / 16 | r 小即可 + alpha=2r 保持缩放稳定（L03） |
| `lora_target: all` | 全挂 | 预算摊到多个矩阵优于集中（L03 结论 2） |
| `template: qwen` | 匹配模型 | chat template 是格式协议，错了模型直接崩（阶段4 L03） |
| `cutoff_len` | 2048 | 按数据长度分布定，影响显存与样本完整性（L06） |
| `num_train_epochs` | 3 | 指令微调极易过拟合，宁少勿多（L06） |

## 5. 训练监控：loss 曲线看什么

训练日志每 `logging_steps` 打印一次 loss；`plot_loss: true` 会在输出目录生成 loss 曲线图
（训练结束后生成，注意确认生成）。看曲线的三个检查点：

1. **整体趋势**：前 10%~20% 步内明显下降 → 学习正常（模型在「学会你的数据」）。
2. **是否归零/过低**：训练 loss 一路降到接近 0 → **过拟合警报**（模型在背答案，L06/L10 的
   「对话变笨」前兆）。指令微调的合理终值通常在 0.5~1.5 区间（视数据与任务而异，看相对趋势
   而非绝对值）。
3. **是否震荡/爆炸**：剧烈震荡或变 NaN → 学习率过大或数据有坏样本（L06 处置清单）。

同时开着 `nvidia-smi -l 1`（阶段 4 L06）：确认显存峰值离上限有余量、GPU-Util 大体饱满。

## 6. 导出合并与快速验证

### 6.1 用 adapter 直接对话（不合并，最快验证）

```bash
llamafactory-cli chat --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path saves/qwen25-7b-java-qlora \
    --template qwen --finetuning_type lora
```

问几个训练集**之外**的同类问题（不是背答案，是考泛化）。效果满意再导出。

### 6.2 导出合并（merge，部署前置）

```yaml
# export.yaml —— 典型示例
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: saves/qwen25-7b-java-qlora
template: qwen
finetuning_type: lora
export_dir: models/qwen25-7b-java-merged   # 导出目录：W+BA 合并后的完整权重
export_size: 5                              # 分片大小（GB）
export_device: cpu                          # 在 CPU 上合并，省 GPU 显存
```

```bash
llamafactory-cli export export.yaml
```

导出产物 = 一个标准 HF 模型目录（safetensors），可被 transformers/vLLM/Ollama 直接使用
（L09 部署课的主角）。**合并这一步就是 L01 理论「$W_{\text{merged}} = W + BA$」的落地。**

## 7. 云 GPU 场景完整操作链（AutoDL 为例）

```
1. 租卡：4090 24GB 实例 + 预置 PyTorch/CUDA 镜像（阶段4 L06 流程）
2. 环境：pip install llamafactory（或提前打包好 requirements 离线装）
3. 传数据：scp 上传 数据集 json + 训练 yaml（MB 级，秒传）
4. 拉模型：export HF_ENDPOINT=https://hf-mirror.com 后在实例内下载（GB 级，现场下更快）
5. 空跑检查：先用 max_samples: 20 跑 30 个 step，确认数据/模板/显存无误
6. 全量训练：nohup llamafactory-cli train xxx.yaml > train.log 2>&1 &（断网不中断）
7. 收工：下载 adapter（几十 MB）或合并权重（14GB，转网盘/压缩分包）→ 关机
```

第 5 步是省钱关键：**先花 5 分钟空跑验证，再花几小时全量训练**。

## 8. 预计耗时与费用量级（对齐阶段 4 L06 测算表）

| 场景 | 数据量 | 耗时（4090 级参考） | 费用量级 |
|---|---|---|---|
| 1.5B 空跑全流程 | 50 条 | 10~20 分钟 | 云卡 <1 元 |
| 7B 全量（毕业项目规模） | 500~2000 条 × 3 epoch | 1~6 小时 | 每次约 10~30 元 |

> 🍿 彩蛋｜第一次微调的仪式感：训练跑起来后，盯着 loss 曲线下降的那几分钟，是每个入行者的成人礼。也提醒一句：第一次微调大概率会撞上 OOM 或格式报错——报错不是失败，是学费；课末的排坑清单就是前人的学费合集。

## 练习

1. 用 0.5B/1.5B + `alpaca_sample.json`（清洗后）完整走一遍：注册→训练→chat 验证→导出，
   每步截图存档。
   提示：卡住时优先检查 dataset_info.json 的字段映射与 template 两个地方。
2. 故意把 `template` 改成一个错误值（如 `llama2`）再 chat 一次，观察输出崩坏的样子——
   亲手体验「格式即协议」。
   提示：崩坏形态可能是胡言乱语、不停止、或直接吐特殊 token。
3. 把 `lora_rank` 从 8 改成 64 训练对比（小模型快速对比），观察最终 loss 与对话效果的差别
   是否如 L03 预期（r 增大收益有限）。
   提示：同时看训练耗时与 loss 曲线，r=64 明显更慢但 loss 不一定更低。
4. 在云实例上完成一次 7B 全量训练并把 adapter 下载回本地，记录全链路耗时与花费。
   提示：训练用 nohup 挂后台，期间断开 ssh 不影响；结束立刻关机。

## 小结与自测

1. **Q：从「数据文件」到「能被训练」要做什么？**
   A：放入 data/ 目录 + 在 dataset_info.json 注册（声明文件名与 alpaca 字段映射）。
2. **Q：一行什么参数把 LoRA 变成 QLoRA？**
   A：`quantization_bit: 4`（主干按 NF4 量化加载）。
3. **Q：训练 loss 降到接近 0 说明什么？**
   A：过拟合（在背训练集），对话大概率变笨/复读——减 epoch 或补数据，不是「练得好」。
4. **Q：导出合并的产物是什么？为什么部署前要合并？**
   A：$W+BA$ 合并后的完整 HF 模型目录；部署引擎无需理解 LoRA 旁路，与原模型同构零额外延迟。
5. **Q：云上训练的省钱操作链里最关键一步？**
   A：全量训练前先 max_samples=20 空跑验证配置与显存。

## 下一课预告

跑通之后是「调好」：下一课讲微调超参的基线与调节直觉（lr 为什么比全参高一个量级、
epoch 为什么宁少勿多、OOM 排查清单、loss NaN 处置）。
