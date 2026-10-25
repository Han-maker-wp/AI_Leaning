# LLaMA-Factory QLoRA 训练全流程命令速查（L05 配套）

> 版本迭代快：**所有命令、参数名、配置项以 LLaMA-Factory 官方 README（含 README_zh）为准**。
> 本文是「典型流程示例」，用于速查与核对步骤顺序。
> 讲义正文见 `../L05_QLoRA实操全流程.md`。

## 0. 前置条件

- [ ] Python 3.10+ 虚拟环境已建好（conda/uv 均可，见 L04）
- [ ] `torch.cuda.is_available()` 为 True（云平台预置镜像可跳过）
- [ ] 国内网络：`export HF_ENDPOINT=https://hf-mirror.com`
- [ ] 显存账已口算（7B QLoRA ≈ 6~10GB，见阶段 4 L04）

## 1. 安装与环境自检

```bash
# 方式一：pip 安装
pip install llamafactory

# 方式二：源码安装（推荐项目期使用，可锁版本、可查模板源码）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 环境自检：确认 torch/CUDA/显卡识别正常（子命令以官方 README 为准）
llamafactory-cli env
```

## 2. 数据准备与注册

```bash
# 1) 清洗数据（用 data_clean.py，输入输出按需替换）
python data_clean.py --in my_raw.json --out my_dataset.json

# 2) 复制到 LLaMA-Factory 的 data 目录
cp my_dataset.json /path/to/LLaMA-Factory/data/my_dataset.json
```

在 `/path/to/LLaMA-Factory/data/dataset_info.json` 中追加注册（alpaca 格式典型写法）：

```json
"my_dataset": {
  "file_name": "my_dataset.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

sharegpt 多轮格式的注册字段不同（messages/系统提示等映射），以官方 README 数据集章节为准。

## 3. 训练

### 3a. WebUI 路线（第一次跑通用）

```bash
llamafactory-cli webui
# 浏览器打开提示的地址（默认 7860 端口），按 L05 §4.1 参数表填写后开始训练
# WebUI 可导出 yaml，沉淀为下面的 CLI 配置
```

### 3b. CLI 路线（正式交付）

`qwen25_7b_qlora.yaml`（关键项注释见 L05 §4.2）：

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all
quantization_bit: 4          # 加这一行 = QLoRA

dataset: my_dataset
template: qwen               # 模板名以官方模型支持表为准
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: saves/qwen25-7b-mytask-qlora
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500
plot_loss: true
```

```bash
# 空跑检查（强烈建议）：先限制样本量跑几十 step，验证数据/模板/显存
#   临时把 max_samples 改为 20、num_train_epochs 改为 0.1 量级跑一遍
llamafactory-cli train qwen25_7b_qlora.yaml

# 云上长任务：挂后台，断开 ssh 不中断
nohup llamafactory-cli train qwen25_7b_qlora.yaml > train.log 2>&1 &
tail -f train.log
```

## 4. 训练监控要点

```bash
# 另开终端看显存（峰值余量 ≥10% 为健康）
nvidia-smi -l 1
```

loss 曲线（`plot_loss: true` 训练后生成在 output_dir）看三件事（详见 L06）：

1. 前 10~20% 步明显下降 → 学习正常
2. loss 接近 0 → 过拟合警报（减 epoch）
3. 震荡/NaN → lr 过大或坏样本

## 5. 快速验证（不合并，直接挂 adapter 对话）

```bash
llamafactory-cli chat \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path saves/qwen25-7b-mytask-qlora \
    --template qwen \
    --finetuning_type lora
```

用**训练集之外**的同类问题测试泛化，而不是背题能力。

## 6. 导出合并（部署前置）

`export.yaml`：

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: saves/qwen25-7b-mytask-qlora
template: qwen
finetuning_type: lora
export_dir: models/qwen25-7b-mytask-merged
export_size: 5
export_device: cpu        # CPU 上合并，省 GPU 显存
```

```bash
llamafactory-cli export export.yaml
# 产物 models/qwen25-7b-mytask-merged 为标准 HF 模型目录（safetensors）
```

## 7. 云 GPU（AutoDL 等）操作链速记

```
租卡(4090 级+预置镜像) → 装环境(pip install llamafactory)
→ scp 上传数据/yaml → 实例内 HF 镜像下载基座模型
→ 空跑验证 → nohup 全量训练 → 下载 adapter/合并权重 → 关机
```

费用量级：单次 1~6 小时 ≈ 10~30 元；全项目 100~300 元（阶段 4 L06 测算）。

## 8. 常见报错速查

| 报错/现象 | 首查位置 |
|---|---|
| 找不到数据集 my_dataset | dataset_info.json 是否注册、json 文件是否在 data/ 下、键名拼写 |
| 输出胡言乱语/吐特殊 token | `template` 与模型不匹配（换成 qwen 等，查官方支持表） |
| CUDA out of memory | 按 L06 清单：batch↓ → cutoff_len↓ → gradient_checkpointing: true |
| loss 为 NaN | lr 减半、查坏样本、确认 bf16 |
| 下载模型超时 | `export HF_ENDPOINT=https://hf-mirror.com` 后重试 |
