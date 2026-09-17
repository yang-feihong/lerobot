# Pi0.5 机械臂模式训练说明

当前正式训练链路不增加独立分类头。机械臂模式与 B2、EE 和夹爪动作一样，由 Pi0.5 的统一 flow-matching 动作头输出。

## 三态联合编码

数据集中的两个字段必须同时启用或同时关闭：

| 模式 | `arm_teleop_inactive` | `arm_reset` |
| --- | ---: | ---: |
| `TELEOP` | 0 | 0 |
| `INACTIVE` | 1 | 0 |
| `RESET` | 0 | 1 |

`(1, 1)` 是非法组合。配置只启用其中一个字段会直接报错，训练数据中出现非法组合也会直接报错。

正式入口使用：

```bash
action_semantics_profile="joint_control_arm_mode_v2"
predict_arm_teleop_inactive="true"
predict_arm_reset="true"
discrete_action_training_mode="continuous_flow"
action_loss_schema="auto"
```

旧的 `joint_control_ee_v1` 保留用于复现历史训练，不输出机械臂模式。

## 监督和 loss

两个模式字段保留在普通 action vector 中，参与原生 flow matching；它们没有独立的线性头、交叉熵或 CRF。

训练开始前会根据 action chunk 的实际标签复用次数扫描完整 train split，分别统计两个字段的正负样本比例。flow residual 按固定的全局类别比例加权，避免少数类 RESET 被多数类淹没。25维 control-extended 数据集和旧16维 action 数据集都支持该统计。

连续动作 mask 使用数据集真值，而不是模型预测：

```text
TELEOP   : 监督 B2、EE、夹爪和两个模式维度
INACTIVE : 监督 B2、夹爪和两个模式维度；不监督 EE
RESET    : 监督 B2、夹爪和两个模式维度；不监督 EE
```

因此模型不能通过预测 INACTIVE 或 RESET 逃避 EE loss；只有数据集真值决定当前 EE 是否有效。

## checkpoint metadata

checkpoint 会记录：

- 两个模式字段是否启用；
- `paired_flow_channels` 编码；
- 三个合法组合和一个非法组合；
- 完整训练集的两个字段类别先验；
- B2、Z1、夹爪等其余动作语义。

这保证后续 open-loop 与部署端可以从 checkpoint 恢复协议，而不依赖入口脚本中的隐含默认值。

部署端在归一化域内把两个 flow 输出联合投影到 `(-1,-1)`、`(+1,-1)`、`(-1,+1)` 三个合法原型，再转换为 IK Bridge 已有协议：

| 模式 | `arm_active` | `arm_reset` | IK Bridge 行为 |
| --- | ---: | ---: | --- |
| `TELEOP` | 1 | 0 | 使用 height-invariant EE 目标做 IK |
| `INACTIVE` | 0 | 0 | 不发布机械臂目标 |
| `RESET` | 1 | 1 | 执行 IK Bridge 的 `VLA_RESET_PRESET`（当前默认 middle） |

部署端不锁存、不延长、不去抖模式输出；每个执行帧都直接采用模型当前帧的联合解码结果。IK Bridge 仍保持 `arm_reset` 优先于 `arm_active` 的既有处理顺序。没有 `arm_mode_encoding` metadata 的历史 checkpoint 继续使用历史独立阈值解码。

## 历史结构化模式

代码仍保留 `structured_temporal`，用于读取和复现已经采用独立 CRF/离散头的历史实验。它不是当前正式训练配置，也不会由 `joint_control_arm_mode_v2` 启用。
