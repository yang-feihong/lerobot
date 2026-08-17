# Pi0.5 训练实验记录

本文档只登记需要继续用于方案比较的训练。更早的探索性训练不再追溯。

每次启动新对比训练前，先在本文档写明实验矩阵；结束后补充 W&B、最终 step、本地 checkpoint 和结论。历史配置以 checkpoint 中的 `train_config.json`、`config.json` 和 W&B 为准，不能用当前 `train_vla_pi05.sh` 倒推。

## 2026-08-11：上一轮四组训练

### 实验矩阵

上一轮是以下两个变量的 2×2 网格：

1. B2 action：`velocity` / `local_trajectory`；
2. state/action 输入：
   - `text`：当前 state 使用 Pi0.5 文本编码，不输入历史 action；
   - `continuous`：13 帧连续 state token，并输入同一时间范围的 12 帧历史 action。

上一轮没有启用 MEM，也没有结构化离散输出。

| 运行名 | B2 | state | 历史 action | MEM | 离散训练 | W&B | 最终状态 | 本地最新 ckpt |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| `pi05_ee_delta_local_trajectory_continuous_20260811_132732` | local trajectory | continuous | 开启 | 关闭 | continuous flow | [uxxjdehr](https://wandb.ai/yangfh/b2-z1-vla-ee-delta-grid/runs/uxxjdehr) | finished，20,000 | 16,500、20,000 |
| `pi05_ee_delta_local_trajectory_text_20260811_132732` | local trajectory | text | 关闭 | 关闭 | continuous flow | [80b199jt](https://wandb.ai/yangfh/b2-z1-vla-ee-delta-grid/runs/80b199jt) | finished，20,000 | 16,500、20,000 |
| `pi05_ee_delta_velocity_continuous_20260811_132732` | velocity | continuous | 开启 | 关闭 | continuous flow | [zzu90d3s](https://wandb.ai/yangfh/b2-z1-vla-ee-delta-grid/runs/zzu90d3s) | finished，20,000 | 16,500、20,000 |
| `pi05_ee_delta_velocity_text_20260811_132732` | velocity | text | 关闭 | 关闭 | continuous flow | [xs68gzxg](https://wandb.ai/yangfh/b2-z1-vla-ee-delta-grid/runs/xs68gzxg) | finished，20,000 | 17,000、20,000 |

四个 W&B run 均已通过 API 核验：`state=finished`、summary `_step=20000`。四组最终 checkpoint 已同步到本地数据盘；每组保留最后两个编号版本，`last` 均指向 20,000。完成本地校验后，远程上一轮输出目录已删除。

### 共同配置

| 配置 | 值 |
| --- | --- |
| 数据集 | `/data/datasets/b2_z1_vla_lerobot` |
| train episode | 1,831 |
| val | `eval_split=0.1` |
| Z1 action | `ee_delta` |
| MEM | 关闭 |
| `action_loss_schema` | `auto` |
| bool 平衡 | 开启，权重 4.0 |
| completion 尾段 | 2.0 秒 |
| state 历史 | 13 帧，间隔 0.04 秒，覆盖约 0.48 秒 |
| 每 GPU batch | 2 |
| 每任务 GPU | 2 |
| gradient accumulation | 12 |
| global batch | 48 |
| 学习率 | `2.5e-5` |
| seed | 1000 |
| steps | 20,000 |
| eval/save | 每 500 step |
| W&B project | `b2-z1-vla-ee-delta-grid` |

### continuous flow 的准确含义

这四个 checkpoint 没有 `discrete_action_training_mode` 字段，因为训练发生在 `structured_temporal` 加入之前，所以对应现在的 `continuous_flow`。

它们仍启用了 `action_loss_schema=auto` 的 gate-aware loss：bool 维度按 train split 比例加权，B2/EE 连续 loss 使用真值 mask；但没有独立分类头、CRF 或 completion hazard head。

### 两种 state/action 输入

`text` 两组实际只使用当前一帧 state 的文本 token，`action_history_enabled=false`。虽然配置保存了 `state_num_frames=13`，text 模式不会构造多帧 state。

`continuous` 两组使用当前帧加 12 帧历史 state，间隔 0.04 秒，同时启用 12 帧历史 action。因此上一轮比较的是两个完整输入方案，不是只改变编码器的严格单变量实验。

## 2026-08-14：8 个单卡任务

### 固定配置

```text
b2_action_representation = local_trajectory
z1_action_representation = ee_delta
state_num_frames = 13
state_history_frame_interval_seconds = 0.04
task_complete_sample_tail_seconds = 2.0
```

全部任务保留 bool 类别平衡和基于数据集真值的连续 action loss mask。每个任务一张 GPU，`global_batch_size=48`。非 MEM 使用 `batch_size_per_gpu=2`、`gradient_accumulation_steps=24`；MEM 实测每卡 batch 2 在 RTX 4090 上 OOM，因此使用 `batch_size_per_gpu=1`、`gradient_accumulation_steps=48`，有效 global batch 不变。

### 任务矩阵

这 8 个均为新任务，不重复上一轮 checkpoint：

| GPU | 编号 | MEM | state | 历史 action | 离散训练 | 主要比较 |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | N0 | 关闭 | text | 关闭 | structured temporal | 对比上一轮 E0，验证 structured |
| 1 | N1 | 关闭 | continuous 13 帧 | 开启 | structured temporal | 对比上一轮 E1，验证 structured |
| 2 | N2 | 开启 | text | 关闭 | continuous flow | 对比上一轮 E0，验证 MEM |
| 3 | N3 | 开启 | continuous 13 帧 | 开启 | continuous flow | 对比上一轮 E1，验证 MEM |
| 4 | N4 | 开启 | text | 关闭 | structured temporal | MEM + structured 的 text 方案 |
| 5 | N5 | 开启 | continuous 13 帧 | 开启 | structured temporal | MEM + history + structured 完整方案 |
| 6 | N6 | 关闭 | continuous 13 帧 | 关闭 | structured temporal | 与 N1 比较历史 action |
| 7 | N7 | 开启 | continuous 13 帧 | 关闭 | structured temporal | 与 N5 比较 MEM 下历史 action |

### 2026-08-14 实际运行

数据集共 1,832 个 episode，按固定划分使用 1,648 train / 184 val。训练采样检查发现 episode 1543 没有任何 `task_complete=1` 尾段，因此明确从 train sampler 排除；其余有效 train 起点共 3,689,481 个。

| 编号 | 输出目录时间 | W&B |
| --- | --- | --- |
| N0 | `20260814_135106` | [ksm3bwep](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/ksm3bwep) |
| N1 | `20260814_135106` | [4t36ji9l](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/4t36ji9l) |
| N2 | `20260814_141414` | [ktxr1mcg](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/ktxr1mcg) |
| N3 | `20260814_141414` | [npwgkanq](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/npwgkanq) |
| N4 | `20260814_141414` | [otl4uxbl](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/otl4uxbl) |
| N5 | `20260814_141414` | [6uj89j4o](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/6uj89j4o) |
| N6 | `20260814_135106` | [nmo76roj](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/nmo76roj) |
| N7 | `20260814_141414` | [hs2cwdw4](https://wandb.ai/yangfh/b2-z1-vla-grid8-20260814/runs/hs2cwdw4) |

八组均已完成 step 500 首次验证、保存完整 checkpoint，并继续完成至少一个训练 step：

| 编号 | step 500 val loss | val batches |
| --- | ---: | ---: |
| N0 | 0.7459 | 256 |
| N1 | 0.6735 | 256 |
| N2 | 0.0730 | 512 |
| N3 | 0.0714 | 512 |
| N4 | 0.4826 | 512 |
| N5 | 0.3612 | 512 |
| N6 | 1.4153 | 256 |
| N7 | 0.4456 | 512 |

非 MEM checkpoint 约 4.14 GB，MEM checkpoint 约 7.35 GB。远程保留器依据完整写入后的 `checkpoints/last` 更新，只保留各 run 最新两个编号 checkpoint。本地同步器逐文件校验大小后将 `.partial` 原子改名，同样只保留最新两个；八组 step 500 均已完成本地校验。

上一轮用于直接对照的两个 local-trajectory 基线：

| 编号 | 配置 | W&B |
| --- | --- | --- |
| E0 | text + 无历史 action + 非 MEM + continuous flow | [80b199jt](https://wandb.ai/yangfh/b2-z1-vla-ee-delta-grid/runs/80b199jt) |
| E1 | continuous state + 历史 action + 非 MEM + continuous flow | [uxxjdehr](https://wandb.ai/yangfh/b2-z1-vla-ee-delta-grid/runs/uxxjdehr) |

### 启动后的返回条件

八个任务都必须满足：

1. 使用表中指定的唯一 GPU 和配置；
2. W&B 在线 run 创建成功；
3. 完成 optimizer step，无 OOM、NaN、数据或 processor 错误；
4. 第一次 val 完整结束，W&B 出现有限的 `overview/val_loss` 和 `data/val_batches`；
5. val 后继续完成至少一个训练 step。

## 2026-08-16：修正 EE delta 语义后的下一轮训练计划

### 背景与历史 checkpoint 的使用边界

2026-08-11 和 2026-08-14 的 EE-delta 训练发生在以下问题修正之前：

1. EE delta 的归一化统计量由绝对 EE 工作空间范围近似，而不是在连续 episode 内完成 absolute EE pose → EE delta 变换后统计真实 delta；
2. 旋转 delta 使用 identity-centered rot6d，而不是三维 rotation vector；
3. 连续 EE loss 没有完整要求 delta 两端同时处于 active、非 reset 状态；
4. 旧 continuous-history checkpoint 没有完整保存 `state_memory_proj` 和 `action_memory_proj` 的 weight/bias；
5. 旧 bool 输出没有统一使用结构化时序分类头。

因此，旧训练可以保留作历史实现记录，但不能再用来评价新 EE-delta 表示、continuous history 或结构化门控的模型能力，也不能与新训练的 validation loss 直接横向比较。下一轮必须从基础 PI0.5 checkpoint 重新训练，不能从上述旧 checkpoint resume。

全量数据检查确认位置监督本身是充分的：1,830 个 episode 中识别出 4,218 段有效机械臂操作；操作开始后前 1 秒 EE 位移中位数约为 5.4 cm，完整操作段的 EE 净位移中位数约为 12.5 cm。旧模型没有学会移动并对准门把手，不能解释为数据中没有明确的位置动作。

### 所有新训练固定不变的动作语义

以下参数不是本轮实验变量，所有新 run 必须保持一致：

| 配置项 | 固定值 | 说明 |
| --- | --- | --- |
| `z1_action_representation` | `ee_delta` | 模型预测相邻目标之间的 EE 增量 |
| `ee_delta_rotation_representation` | `rotvec` | 使用三维 rotation vector，避免 identity-centered rot6d |
| EE delta statistics | 变换后的真实统计量 | 必须遍历连续 episode，完成动作变换后计算 q01/q99，并保存到 checkpoint |
| EE delta loss mask | 两端均 active、非 reset、非 padding | t→t+1 只有在两个端点都有效时才参与连续 EE loss |
| `discrete_action_training_mode` | `structured_temporal` | arm mode、gripper 和 task complete 使用结构化离散输出 |
| `finetune_mode` | `lora` | action expert/projection 全量训练，VLM/ViT 使用 LoRA；不使用仅冻结视觉的 expert-only 作为正式模型 |
| `lora_rank` / `lora_alpha` | 16 / 32 | 保持不同 run 的可比性 |
| `action_chunk_size` | 50 | 监督完整 1 秒动作序列 |
| `action_steps_to_execute` | 25 | 部署默认每执行 0.5 秒重新推理；该参数不参与训练 loss |
| `control_frequency_hz` | 50 | 与数据频率和低层控制频率一致 |
| `num_inference_steps` | 10 | 保持 checkpoint 默认流匹配积分步数 |
| `task_complete_sample_tail_seconds` | 2.0 | 保留完成尾段，同时避免尾段样本支配训练 |
| `global_batch_size` | 48 | 不随单卡 micro batch 改变 |
| `seed` | 1000 | 第一轮严格对照固定随机种子 |
| `eval_steps` / `save_freq` | 500 / 500 | 每 500 step 形成同一节奏的验证与 checkpoint |
| task variants | 开启 | 训练 batch 按 episode 从 `meta/task_variants.json` 选择语言改写 |

正式 LoRA 训练必须检查 adapter 中包含以下完整训练模块：

```text
model.state_memory_proj
model.action_memory_proj
model.arm_mode_head
model.gripper_state_head
model.task_complete_head
model.arm_mode_crf
model.gripper_state_crf
```

其中 continuous state 或 action history 未启用时，相应的 memory projection 可以不存在；启用时其 weight/bias 必须同时保存且能够由部署加载入口恢复。

### 第一阶段：三个非 MEM 严格对照

第一阶段只改变 state 表示和历史 action，B2 固定为 `local_trajectory`，MEM 固定关闭。三个训练应顺序执行，不再一次性铺开八组网格。

| 优先级 | 编号 | B2 action | state 输入 | 历史 action | MEM | 离散训练 | 主要问题 |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| P0 | A | `local_trajectory` | `text`，当前帧 | 关闭 | 关闭 | `structured_temporal` | 修正后的传统单帧 VLA 是否已经能学会启动机械臂并移动到门把手 |
| P0 | B | `local_trajectory` | `continuous`，13 帧、0.04 秒间隔 | 关闭 | 关闭 | 在不引入历史 action 的情况下，连续机器人状态是否改善操作阶段判断和 EE 位置预测 |
| P0 | C | `local_trajectory` | `continuous`，13 帧、0.04 秒间隔 | 开启，12 帧严格过去动作 | 关闭 | 与 B 严格比较历史已执行 action 对 RTC、运动滞后和目标连续性的价值 |

三个实验的依赖关系如下：

1. A 是必须存在的干净基线。它最直接回答新 EE-delta 训练是否解决了位置对准问题；
2. B 只相对 A 引入 continuous state history，用于评估连续状态表示本身；
3. C 只相对 B 增加历史 action，用于评估 action history，不把 state 编码和 action history 混成一个变量；
4. C 虽然是当前 `train_vla_pi05.sh` 的默认完整方案，但在 B/C 对照证明收益之前，不能预设 C 一定最好。

如果当前只允许使用一张 GPU，则执行顺序为 A → B → C。每个 run 使用唯一输出目录和 W&B run，不允许从另一个实验的 checkpoint resume。

### 第二阶段：根据第一阶段结果追加的条件实验

第二阶段不与第一阶段同时启动。先从 A/B/C 中选出最佳 state/history 方案，再逐个改变 B2 表示、图像历史和图像增强。

| 优先级 | 编号 | 相对前一最佳方案的唯一主要变化 | 目的 | 启动条件 |
| ---: | --- | --- | --- | --- |
| P1 | D | `b2_action_representation: local_trajectory → velocity` | 判断导航更适合学习实际运动形成的局部轨迹，还是直接学习低层接收的速度命令 | A/B/C 已选出 state/history 方案 |
| P2 | E | `enable_mem: false → true`，6 帧、0.5 秒间隔 | 评估历史图像对门状态、遮挡和开门进度的帮助 | 非 MEM 模型已经能可靠启动机械臂并基本对准门把手 |
| P2 | F | 在胜出模型上增加轻量图像增强 | 专门评估实采图像与仿真渲染之间的域差异 | 真实图像 open-loop 正常，但仿真图像表现显著下降 |

D 必须继承 A/B/C 的胜出 state/history 配置，避免再次把 B2 表示和输入历史同时改变。E 必须继承 D 之后的胜出配置。F 首次引入时不要同时改变 MEM，否则无法判断提升来自图像增强还是视觉历史。

### 暂不作为主线的配置

| 配置 | 暂不优先的原因 |
| --- | --- |
| `continuous_flow` bool 输出 | 已确认不适合四个离散控制语义，不再作为正式主线 |
| `expert-only` | 只适合流程和显存检查；门把手视觉定位需要适配 VLM/ViT |
| full fine-tune | 当前 24 GB GPU 使用 AdamW 无法稳定容纳，且没有必要先承担该成本 |
| rot6d 对照 | 旧 rot6d checkpoint 同时混有错误归一化统计，不能形成干净对照 |
| 首轮直接启用 MEM | 如果当前帧模型尚未学会位置对准，MEM 会增加变量并掩盖根因 |
| 同时训练大量组合 | 新动作语义刚完成修正，应优先获得可归因的单变量结论 |

### 训练过程中的评估与选型规则

不能再仅凭总体 validation loss 选择 checkpoint。每个 run 至少在 step 500、2,000、5,000 以及最终候选 checkpoint 上，对固定的 staff1/staff2 held-out episode 执行操作起始窗口 open-loop 评估。

模型选型指标按以下优先级解释：

1. 预测 active 条件下，前 1 秒 EE 平移 endpoint error；
2. manipulation onset 检出率；
3. arm-mode precision 和 recall；
4. staff1/staff2 wrist-twist 符号准确率；
5. 前 1 秒累计 SO(3) 旋转误差；
6. gripper 和 task-complete 的离散准确率与时序完整性；
7. 总体 validation loss 只作为辅助指标，不作为首要排序依据。

open-loop 评估必须强制包含每个 manipulation onset 前后窗口，不能因为全局 frame stride 或最大帧数限制跳过操作开始阶段。staff1 和 staff2 必须分别报告，不能只给混合平均值掩盖腕部旋转方向错误。

建议每个新 run 先训练到 5,000 step 做阶段决策：如果操作 onset、EE endpoint 和腕部方向均没有随训练改善，应先检查数据、输入和 loss，而不是无条件继续到 20,000 step。只有指标明确改善的候选模型再延长到 20,000 step，并进入固定初始条件的闭环测试。

### 闭环测试准入条件

模型进入 VLA+wholebody 闭环测试前必须满足：

1. checkpoint-local `pi05_transformed_action_stats.json` 存在，且 q01/q99 与 normalizer 完全一致；
2. checkpoint metadata 明确记录 `z1_action_representation=ee_delta` 和 `ee_delta_rotation_representation=rotvec`；
3. continuous-history 模型通过部署实际使用的 PEFT 加载入口恢复，不能出现 memory projection 缺失；
4. staff1/staff2 操作起始窗口指标均已生成；
5. 真实图像 open-loop 中不能稳定误判 task complete；
6. 首轮闭环先关闭 task-complete 提前终止，完整运行固定仿真时长，以便观察门控、位置和旋转的全过程；
7. 闭环视频和日志中同时保留模型输出、实际执行区间、EE 实际状态和所有离散控制状态。

### 2026-08-16：staff1 A/B 实际运行

本轮只使用 staff1 数据集 `/data/datasets/b2_z1_vla_lerobot_staff1`，共 1,229 个 episode、2,902,892 帧、50 Hz；固定划分为 1,106 train / 123 val episode。两个任务均从基础 PI0.5 checkpoint 独立启动，训练 5,000 step，每个任务使用 4 张 RTX 4090，单卡 batch 为 2、gradient accumulation 为 6，有效 global batch 为 48。

| 编号 | GPU | state / history action | 输出目录 | W&B | step 500 val loss | val batches |
| --- | --- | --- | --- | --- | ---: | ---: |
| A | 0,1,2,3 | `text` / 关闭 | `20260816_063343_pi05_b2_z1_vla_staff1_A` | [0zxojvxc](https://wandb.ai/yangfh/b2-z1-vla-staff1-ab-20260816/runs/0zxojvxc) | 0.3927 | 64 |
| B | 4,5,6,7 | `continuous` 13 帧、0.04 秒 / 关闭 | `20260816_063834_pi05_b2_z1_vla_staff1_B` | [ao4u8h7i](https://wandb.ai/yangfh/b2-z1-vla-staff1-ab-20260816/runs/ao4u8h7i) | 0.4608 | 64 |

验收时 A 已在验证后继续到至少 step 510，B 已继续到至少 step 505；两组均无 OOM、NaN、数据读取或 processor 错误。两组 `checkpoints/last` 均已指向完整写入的 `000500`，远程保留器持续执行 latest-2 规则。

checkpoint 检查确认 A 的 adapter 保存 action expert/projection、结构化离散 head 和 CRF；B 在此基础上额外完整保存 `model.state_memory_proj`。两组 checkpoint 均包含 `pi05_transformed_action_stats.json` 和 `pi05_deployment_metadata.json`，W&B 的 model artifact 上传保持关闭。
