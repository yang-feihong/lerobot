# PI0.5 分阶段开环评估

`openloop_vla_eval.sh` 使用 `--stage` 选择与模型训练边界一致的数据集：

| stage | 数据范围 |
| --- | --- |
| `full` | 完整开门 episode |
| `approach` | 机器狗对位阶段 |
| `handle_press` | 机械臂操作门把手阶段 |
| `door_traversal` | 复位、推门和通过阶段 |
| `custom` | 调用者显式提供数据集路径与 repo id |

脚本不再写死历史 episode 编号。`--splits=train|eval|both` 选择训练子集、留出子集或两者；
默认 `both`，分别写入 `<output-root>/train` 和 `<output-root>/eval`，不会互相覆盖。
`--eval-split` 默认 `0.1`，必须与训练配置一致。

未传 `--train-episodes` 或 `--eval-episodes` 时，评估器与训练端共用同一个 episode 划分函数：

- 普通数据集按 metadata 中的任务文本分层；
- 数据集根目录存在 `meta/semantic_views.json` 时自动按物理 `source_group` 分层，并尊重
  `selected_episodes`；
- 侧车不在标准位置时用 `--semantic-views-path` 显式指定；
- 显式传入 episode 列表时，以显式列表为准。

每次结果的 `summary.json` 都记录 `split`、`split_strategy`、`eval_split` 和实际 episode，
因此可以审计该结果来自训练数据还是 held-out 数据。模型动作维度、B2/Z1 表示以及 anchor
画法始终从 checkpoint metadata 读取。

```bash
./openloop_vla_eval.sh \
  --stage=handle_press \
  --policy-path=/data/models/trained/pi05/<run>/checkpoints/050000/pretrained_model \
  --output-root=/data/evaluations/open_loop/b2_z1/<name> \
  --splits=both \
  --eval-split=0.1
```

只评 held-out 子集时使用 `--splits=eval`；复现原有 train 子集评测时使用
`--splits=train`。两者调用同一评测实现，除 episode 集合外，动作解码、归一化、GT 构造和
物理指标口径完全一致。

## 固定物理指标协议

用于跨 checkpoint 报告的 `run_standard_openloop_benchmark.sh` 同样支持
`--dataset-split=train|eval`。三个 `evaluation_protocols/b2_z1_staff1_stage*_full_horizon_v1.json`
都固定保存了一组 train episode 和一组从训练留出区取得的 eval episode；两组各 100 条，
并分别锁定 episode 范围、输入数据哈希、anchor 数和有效动作步数。默认仍为 `train`，保证
历史报告可复现；held-out 评测显式写：

```bash
./run_standard_openloop_benchmark.sh \
  --protocol=evaluation_protocols/b2_z1_staff1_stage2_full_horizon_v1.json \
  --dataset-split=eval \
  --policy-path=/data/models/trained/pi05/<run>/checkpoints/050000/pretrained_model \
  --dataset-root=/data/datasets/b2_z1/benchmarks/staff1_stage2_full_horizon_v1 \
  --output-dir=/data/evaluations/open_loop/b2_z1/<name>/eval
```

输出中的 `protocol_id` 对 train/eval 不同，比较工具不会误把训练集成绩与留出集成绩放进
同一个可比组。
