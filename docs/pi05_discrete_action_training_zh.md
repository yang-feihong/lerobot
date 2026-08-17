# Pi0.5 离散动作训练说明

本文档说明当前 B2+Z1 Pi0.5 训练链路如何处理离散输出。内容以仓库现有实现为准，主要涉及：

- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/scripts/pi05_vla_server.py`
- `train_vla_pi05.sh`

## 1. 当前启用的离散输出

当前数据集 action 中有四个具有离散语义的字段：

| 数据集字段 | 真值语义 | 模型中的训练形式 |
| --- | --- | --- |
| `arm_teleop_inactive` | 当前没有机械臂遥操作指令 | 与 `arm_reset` 合并为三分类 `arm_mode` |
| `arm_reset` | 当前机械臂处于 reset 状态 | 与 `arm_teleop_inactive` 合并为三分类 `arm_mode` |
| `gripper_target` | 两种夹爪目标之一 | 二状态线性链 CRF |
| `task_complete` | 任务已经完成，之后持续为真 | 首次完成时刻 hazard loss + 吸收态解码 |

正式入口脚本当前配置为：

```bash
predict_arm_teleop_inactive="true"
predict_arm_reset="true"
predict_gripper="true"
predict_task_complete="true"
discrete_action_training_mode="structured_temporal"
```

`structured_temporal` 模式要求上述四项全部启用；当前实现不支持在该模式中单独关闭其中一项。

## 2. 离散量不再参与 flow matching

官方 Pi0.5 会把所有 action 维度放进同一个连续 flow-matching 输出中。本项目在 `structured_temporal` 模式下改变了这一点：

1. 输入 flow matching 前，把四个离散通道置零。
2. 对应的噪声通道也置零。
3. 每一步去噪时，把对应的 flow 输出速度置零。
4. 最终连续 flow loss 不包含这四个维度。
5. 从 action expert 的逐时刻特征上接独立离散预测头。

因此，W&B 的 `action_dimensions/*` 中，这四个维度不会再显示具有意义的 flow MSE；应查看专门的 `discrete_action/*` 指标。

## 3. 从归一化 action 恢复布尔真值

进入 policy 时，action 已经过数据集统计量归一化。离散标签通过归一化域中的零点划分：

```text
arm_teleop_inactive：normalized > 0 为 true
arm_reset：normalized > 0 为 true
task_complete：normalized > 0 为 true
gripper_target：默认 normalized < 0 为 true 类
```

夹爪使用哪一侧作为 true 由 `action_gripper_target_true_side` 控制，当前默认是 `negative`。这里的 true 只代表二分类中的一个类别，不在训练代码中假定它在物理上是“张开”还是“闭合”。

这种判断使用固定的归一化零点，不依赖当前 minibatch 是否刚好同时包含正负样本。

## 4. 类别比例如何统计

训练开始、构造 policy 之前，`lerobot_train.py` 会扫描完整 train split，预先统计每个已启用离散字段的正负样本比例。val 不参与统计。

统计不是简单地把数据集每一行只计算一次，而是考虑：

- action chunk 的 50 个监督位置；
- 相邻 chunk 对同一个 action 标签的重复使用次数；
- 数据集 FPS 到模型控制频率的采样偏移；
- `task_complete` 尾段允许作为 chunk 起点的范围。

因此统计结果对应训练过程中标签实际进入 loss 的频率。

设某个二值标签在有效监督位置中的 true 比例为 `p`，则基础类别权重为：

```text
true_weight  = 0.5 / p
false_weight = 0.5 / (1 - p)
```

这样 true 和 false 两类对加权损失的期望总贡献各占一半。最终日志中显示的有效权重还会乘以：

```text
action_bool_loss_weight = 4.0
```

如果 train split 中某个启用的离散字段只有一个类别，训练会直接报错，不会悄悄退化成永远预测多数类。resume 时，当前 train split 的类别比例必须和 checkpoint 保存的比例一致，否则也会报错。

## 5. 机械臂模式：三状态 CRF

`arm_teleop_inactive` 和 `arm_reset` 并不是两个独立的 sigmoid。这两个真值先被合并为：

| `arm_mode` | `arm_teleop_inactive` | `arm_reset` | 含义 |
| ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 正常 EE 遥操作 |
| 1 | 1 | 0 | 没有遥操作指令 |
| 2 | 0 | 1 | reset |

如果数据中同一时刻两者同时为 true，训练立即报错。

模型为 action chunk 的每个时刻输出 3 个 emission logits，并使用一个可训练的三状态线性链 CRF。机械臂模式损失为：

```text
arm_loss = CRF sequence NLL + class-balanced per-step cross entropy
```

- CRF 学习三种模式之间的转移倾向，减少逐帧独立分类产生的 `0/1/0/1` 抖动。
- per-step cross entropy 使用 train split 的全局类别比例进行平衡，避免稀有 reset 被多数状态淹没。
- 三种状态之间目前没有硬编码的禁止转移；包括 reset 是否只出现一次，都不是模型规则。
- 推理时使用 Viterbi 对完整 action chunk 联合解码，然后还原成两个互斥字段。

## 6. 夹爪：二状态 CRF

夹爪头为每个时刻输出 2 个 emission logits，并使用二状态线性链 CRF：

```text
gripper_loss = CRF sequence NLL + class-balanced per-step cross entropy
```

其中：

- CRF 学习保持状态和切换状态的相对代价。
- per-step cross entropy 根据完整 train split 的夹爪类别比例加权。
- 没有硬编码“最多切换几次”或“切换后不得切回”。
- 推理时使用 Viterbi 联合解码 50 步，再按 checkpoint 中记录的两个物理目标值输出。

所以夹爪虽然最终只有两个输出值，但不是逐帧独立 threshold，也不是官方 Pi0.5 的连续 gripper flow loss。

## 7. 任务完成：hazard + 平衡 BCE

`task_complete` 的语义是状态而不是单帧脉冲：

```text
完成前：false
首次完成及之后直到 episode 结束：true
```

训练前会检查每个 episode：

- 必须存在显式的 `task_complete=true` 尾段；
- true 后不能再次出现 false；
- completion 不是从 episode padding 或最后一帧临时推导出来的。

模型为每个时刻输出一个 completion logit。损失由两部分组成：

```text
completion_loss = first-onset hazard NLL + class-balanced BCE
```

### First-onset hazard NLL

它监督“第一次进入完成状态”的时刻：

- 首次完成之前的每一步都惩罚错误的正 logit；
- 首次完成位置惩罚错误的负 logit；
- hazard 项不重复计算首次完成之后的每个尾段位置。

### Class-balanced BCE

BCE 对所有有效位置监督完成/未完成状态，并使用 train split 的全局正负比例平衡。这一项让首次完成之后的输出继续保持为 true。

推理解码首先以归一化 logit `> 0` 判断完成，然后进行累积 OR：一旦某一步为 true，该 chunk 后续全部变成 true。因此 completion 在推理结果中是吸收态，不会出现 `0/1/0/1`。

## 8. 离散量与连续量的 loss mask

所有 mask 都由数据集真值生成，不使用模型预测值生成，也不会通过 mask 反向传播。

定义：

```text
valid           = 不是 action padding
execution_valid = valid 且 GT task_complete 为 false
ee_valid        = execution_valid
                  且 GT arm_teleop_inactive 为 false
                  且 GT arm_reset 为 false
```

各输出的有效监督范围为：

| 输出 | 有效范围 |
| --- | --- |
| B2 速度或局部轨迹 | `execution_valid` |
| EE 位姿或 EE 增量 | `ee_valid` |
| `arm_mode` | `execution_valid` |
| `gripper_target` | `execution_valid` |
| `task_complete` | `valid`，包括完成尾段 |

这意味着模型提前预测 `task_complete=true`，不会屏蔽当前样本的 B2、EE、arm mode 或夹爪损失。只有数据集真值已经完成时，这些执行动作才不再计入 loss。

同理，只有真值 `arm_teleop_inactive=true` 或 `arm_reset=true` 时，EE 连续 loss 才被屏蔽；模型自己预测 inactive/reset 不能逃避 EE loss。

## 9. 总 loss 的组合

每个样本先分别得到：

- `continuous_loss`：所有有效 B2 与 EE 连续元素的加权平均；
- `arm_loss`：三状态机械臂模式损失；
- `gripper_loss`：二状态夹爪损失；
- `completion_loss`：完成 hazard 与状态 BCE。

最终为：

```text
total_loss = continuous_loss
           + action_bool_loss_weight
             * (arm_loss + gripper_loss + completion_loss) / 3
```

虽然原始数据有四个离散字段，但 `arm_teleop_inactive` 和 `arm_reset` 合并成了一个三分类任务，所以这里是三个离散任务取平均。

## 10. 完成尾段采样

当前配置：

```bash
task_complete_sample_tail_seconds="2.0"
```

它限制“当前输入时刻已经处于完成状态”的 chunk 起点最多保留到首次完成后的 2 秒，避免很长的 ROS bag 尾段支配训练。它不删除 episode 尾段，也不妨碍较早启动的 chunk 使用后续帧作为标签上下文。

因此训练中确实会存在输入画面已经处于任务完成后的样本，用于教模型稳定输出完成；但这类 chunk 起点的数量受到限制。

## 11. W&B 中应该查看什么

当前训练会记录：

```text
continuous_action/loss
continuous_action/val_loss

discrete_action/loss/arm_mode
discrete_action/loss/gripper_target
discrete_action/loss/task_complete

discrete_action/accuracy/arm_mode
discrete_action/accuracy/gripper_target
discrete_action/accuracy/task_complete

discrete_action/val_loss/*
discrete_action/val_accuracy/*

discrete_action/target_fraction/arm_mode_ee
discrete_action/target_fraction/arm_mode_inactive
discrete_action/target_fraction/arm_mode_reset
continuous_action/mask_fraction/ee_pose
```

此外，W&B run config 的 `runtime/action_bool_balance` 保存 train split 的正负数量、true 比例和有效类别权重。

逐帧 accuracy 只能作为基础检查，尤其不能单独评价 `task_complete`。部署时一次过早完成就可能终止整个任务，后续评估还应重点关注：

- premature completion rate；
- 未完成序列的 completion false-positive rate；
- 首次完成时刻的提前量和延迟量；
- gripper 状态准确率与切换时刻误差；
- arm mode 的混淆矩阵，特别是 reset recall。

## 12. 部署解码

部署端使用 checkpoint metadata 中保存的协议：

- `arm_mode` 经 Viterbi 解码后还原为互斥的 inactive/reset 输出；
- 夹爪经 Viterbi 解码后映射为配置中记录的两个目标值；
- `task_complete` 在第一次 true 后锁存；
- 启用 `stop_on_model_task_complete` 时，从完成位置起停止后续可执行运动输出。

训练、open-loop 和闭环 server 必须使用同一个 checkpoint metadata，不能仅凭物理 action 反归一化后的数值重新猜测离散类别。
