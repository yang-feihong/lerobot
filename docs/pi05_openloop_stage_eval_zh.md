# PI0.5 分阶段开环评估

`openloop_vla_eval.sh` 使用 `--stage` 选择与模型训练边界一致的数据集：

| stage | 数据范围 |
| --- | --- |
| `full` | 完整开门 episode |
| `approach` | 机器狗对位阶段 |
| `handle_press` | 机械臂操作门把手阶段 |
| `door_traversal` | 复位、推门和通过阶段 |
| `custom` | 调用者显式提供数据集路径与 repo id |

脚本不再写死历史 episode 编号。未传 `--train-episodes` 或 `--eval-episodes` 时，由评估器
依据数据集 metadata 和训练时相同的 task-stratified split 选择 episode。模型动作维度、
B2/Z1 表示以及 anchor 画法始终从 checkpoint metadata 读取。

```bash
./openloop_vla_eval.sh \
  --stage=handle_press \
  --policy-path=/data/b2_z1_vla_pi05_outputs/<run>/checkpoints/050000/pretrained_model \
  --output-root=/data/b2_z1_vla_openloop_eval/<name>
```
