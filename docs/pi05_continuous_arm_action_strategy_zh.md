# 大小脑协同：机械臂全连续动作方案

## 困难

机器狗行走时，“关节保持”会让末端随机身运动；抓住门把手时，又必须让 height-invariant EE 保持稳定。此前用 `inactive/reset` 离散量切换两种模式，但 PI0.5 的离散输出训练不稳定。

现有 LeRobot 数据已记录阶段标志、机械臂关节状态和 EE target，足够区分 reset 延续与真正 inactive，不需要重新读取 rosbag。

## 第一版方案

| 阶段 | 连续监督 |
| --- | --- |
| 遥操 | 保留现有 LeRobot EE target |
| reset | 保留现有 LeRobot reset target；关节仍在运动的尾段也归入 reset 延续 |
| 真正 inactive | 等机械臂关节稳定后，用 SE(3) 插值连接该段首尾 target |
| 夹爪 | 直接回归物理位置 `0 / -1.047 rad`，不再作为 bool |

模型统一输出 `B2 local trajectory(3) + EE delta rotvec/xyz(6) + gripper(1)`，共 10 维；不再预测 `inactive/reset/task_complete`。低层始终消费 EE delta，并只积分 RTC 实际执行的帧。

## 数据接口与验证

- 数据集只保留四个业务 key：`observation.state`、两路图像和 `action`。LeRobot 自身要求的
  timestamp/index 等 bookkeeping 字段不属于模型业务输入。
- `observation.state` 为 `原49维 + 实测EE 9维 + continuous实测EE 9维 = 67维`；`action`
  为 `原16维 + continuous EE target 9维 = 25维`，不再维护独立 EE key。
- 当前训练从 `observation.state[58:67]` 形成 EE delta 监督，语义标识为
  `inactive_endpoint_interpolated_state`。原始段仍留在同一向量中用于历史 checkpoint replay
  和后续 ablation，但不会作为额外模型输入。
- reset 从采集消息 `reset=true` 开始，以机械臂关节进入稳定后缀为结束；修正后的 inactive
  从该稳定点开始。
- `command_only_inactive_interpolated` 仅作为未来可选消融，不是当前训练前置条件。
- 新训练使用 `continuous_flow + uniform_valid loss`，50 帧监督、4 Hz RTC、50 Hz 执行。
- 数据落盘前检查 NaN、逐帧 EE 平移/旋转上限和夹爪取值；训练前后检查 10 维 schema、每维 delta 分布、梯度、checkpoint metadata、open-loop replay 和闭环执行。
