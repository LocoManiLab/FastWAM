# FastWAM 自定义数据微调指南

本文档说明如何在自定义 LeRobot 数据集（下文以 `/path/to/your_dataset` 表示）上微调 FastWAM。

## 数据集概况

| 项目 | 值 |
|---|---|
| 路径 | `/path/to/your_dataset` |
| Episodes | 109 |
| 帧总数 | 66,868 |
| FPS | 50 |
| 任务 | Push the bottle into the drawer and squat and close drawer |

模态定义参考 `meta/modality.json`：

**Video**

| 键 | lerobot key | 分辨率 |
|---|---|---|
| `ego_view` | `observation.images.ego_view` | 480×640 |

**State**（`observation.state`，43 维）

| 部位 | 维度 |
|---|---|
| left_leg | 0–6 |
| right_leg | 6–12 |
| waist | 12–15 |
| left_arm | 15–22 |
| left_hand | 22–29 |
| right_arm | 29–36 |
| right_hand | 36–43 |

**Action**（`action.motion_token`，64 维）

modality.json 中定义的主 action 为 64 维 learned motion latent token。

---

## 新增文件说明

| 文件 | 用途 |
|---|---|
| `configs/data/custom_1cam.yaml` | 数据配置：摄像头/动作/状态维度、采样参数 |
| `configs/task/custom_1cam_1e-4.yaml` | 任务配置：batch size、学习率、训练轮数 |
| `scripts/custom_setup.sh` | 一键完成环境准备（模型下载 + 文本嵌入预计算） |

---

## 步骤详解

### Step 0：安装环境

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e /path/to/FastWAM
```

---

### Step 1：设置 checkpoint 目录

```bash
cd /path/to/FastWAM
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

> 建议将该 `export` 加入 `~/.zshrc`，避免每次手动设置。

---

### Step 2：生成 ActionDiT backbone

此步骤从 Wan2.2-TI2V-5B DiT 权重插值生成 ActionDiT 初始化权重。
**第一次运行会自动从 ModelScope 下载全部所需权重（DiT + VAE + text encoder + tokenizer，共约 21 GB）。**

```bash
python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    --device cuda \
    --dtype bfloat16
```

> 如需使用 HuggingFace 下载，提前设置：`export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`

---

### Step 3：预计算 T5 文本嵌入

训练前必须将任务描述文本编码为 T5 嵌入并缓存到磁盘。
文本来自 `meta/tasks.jsonl`，本数据集只有 1 条 prompt，很快完成。

```bash
python scripts/precompute_text_embeds.py \
    --config-name train \
    data=custom_1cam \
    model=fastwam \
    "data.train.dataset_dirs=[/path/to/your_dataset]" \
    data.train.text_embedding_cache_dir=/path/to/your_dataset/cache_t5 \
    overwrite=false
```

嵌入缓存会写入 `data.train.text_embedding_cache_dir` 指定的目录。
`configs/data/custom_1cam*.yaml` 中的数据集路径只是占位符，必须通过上面两个 Hydra 覆盖项指定（训练时同样需要）。

---

### Step 4：开始训练

单机多卡（推荐）：

```bash
bash scripts/train_zero1.sh <GPU数量> task=custom_1cam_1e-4 \
    "data.train.dataset_dirs=[/path/to/your_dataset]" \
    data.train.text_embedding_cache_dir=/path/to/your_dataset/cache_t5
```

下面的示例省略了这两个数据路径覆盖项，实际运行时请加上。

例如 4 卡：

```bash
bash scripts/train_zero1.sh 4 task=custom_1cam_1e-4
```

单卡调试：

```bash
bash scripts/train_zero1.sh 1 task=custom_1cam_1e-4 batch_size=2
```

训练输出（checkpoint + 日志）保存在 `runs/custom_1cam_1e-4/<timestamp>/`。

---

### 一键执行（Step 2 ~ Step 3）

```bash
bash scripts/custom_setup.sh /path/to/your_dataset
```

---

## 配置参数说明

### 数据配置 `configs/data/custom_1cam.yaml` 关键参数

| 参数 | 当前值 | 说明 |
|---|---|---|
| `global_sample_stride` | 5 | 数据 50fps，stride=5 → 有效采样率 10fps，33 帧覆盖 3.3 秒 |
| `num_frames` | 33 | 每个训练样本的时间步数（32 action + 1 reference） |
| `action_video_freq_ratio` | 4 | 每 4 个 action step 对应 1 个 video frame（共 8+1=9 帧） |
| `video_size` | [224, 224] | 训练分辨率，从原始 480×640 中心裁剪到正方形 |
| `action_output_dim` | 64 | `action.motion_token` 的维度 |
| `proprio_output_dim` | 43 | `observation.state` 的维度 |
| `delta_action_dim_mask` | 全 false | motion token 为 learned latent，不做 delta 归一化 |

> **FPS 调整建议**：如果任务时长较短（< 2 秒），可将 `global_sample_stride` 改为 `2` 或 `3`。

### 任务配置 `configs/task/custom_1cam_1e-4.yaml` 关键参数

| 参数 | 当前值 | 说明 |
|---|---|---|
| `batch_size` | 4 | 每卡 batch。显存不足时改为 2 |
| `learning_rate` | 1e-4 | 与 LIBERO 相同起点 |
| `num_epochs` | 50 | 共 109 个 episode，50 轮约 5450 梯度步 |
| `save_every` | 500 | 每 500 步保存一次 checkpoint |
| `mot_checkpoint_mixed_attn` | true | 梯度检查点，节省显存 |

---

## 模型维度变化

原 FastWAM（LIBERO）与本配置的维度对比：

| 组件 | LIBERO 原版 | 本配置 |
|---|---|---|
| 摄像头数 | 2（image + wrist） | 1（ego_view） |
| video_size | [224, 448] | [224, 224] |
| action_output_dim | 7 | 64 |
| proprio_output_dim | 8 | 43 |

`model.dit` 中 action 相关维度由 `${data.train.processor.action_output_dim}` 自动推断，
无需手动修改 `configs/model/fastwam.yaml`。

---

## 常见问题

**Q: 报错 `Cannot detect model type ... Model hash: ...`**

FastWAM 使用基于文件哈希的模型类型检测。如果权重文件不是从 ModelScope/HuggingFace 按标准方式下载的，哈希会不匹配。解决方法：删除 `checkpoints/` 下对应目录，让代码重新自动下载。

**Q: 显存不足（OOM）**

按以下顺序尝试：
1. 减小 `batch_size`（改为 2 或 1）
2. 增加 `gradient_accumulation_steps`（保持等效 batch size）
3. 确认 `mot_checkpoint_mixed_attn: true`（已默认开启）

**Q: 训练步数太少**

109 个 episode，每 episode 约 614 帧（66868/109），stride=5 后约 123 步，
33 帧窗口下每 episode 约 90 个有效样本，总共约 9810 个样本。
batch=4、50 轮 ≈ 12263 步。如果不够，增大 `num_epochs`。

**Q: 如何恢复训练**

```bash
bash scripts/train_zero1.sh 4 task=custom_1cam_1e-4 \
    resume=./runs/custom_1cam_1e-4/<timestamp>/checkpoints/step_XXXX
```
