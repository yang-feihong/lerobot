# VLA 训练编排

用户可调配置统一维护在 `training_plan/.env`；镜像、模型和数据版本的精确哈希
维护在 `training_plan/artifacts.lock`。入口不读取隐藏的机器默认值，也不依赖
远程工作区的未提交修改。

## 稳定训练/验证划分（recording_hash_v1）

训练和开环评估共用 `datasets/episode_split.py`，不保留按任务/数据来源末尾截取的旧算法。
分组单位是不可变的原始录制身份：优先语义 sidecar 的 `recording_id`，否则使用
标注中保留的原始 `source_path`。该路径是采集 provenance，不是当前数据集挂载位置；
搬迁数据不应改写它。同一录制的不同阶段、clean/converted 副本及语义视图必须保留同一身份。
不要用 episode 编号、basename、语言、有效阶段、数据集目录或追加位置代替录制身份。
没有语义侧车的数据必须显式提供 `meta/episode_split_groups.json`：

```json
{"format":"lerobot.episode_split_groups","version":1,
 "recording_ids":{"0":"robot/session/immutable-recording-uuid"}}
```

固定版本命名空间 + 录制身份经 SHA256 得到均匀数，低于 `eval_split` 的整组进入验证。
新增、重排、重编号、合并、筛掉坏阶段、修改语言都不改变已有录制的归属。
不把物体或阶段标签加入身份 hash，避免补标注导致原样本换边。
这避免采集顺序与物体/场景分布系统性关联，给各物体、阶段、批次同样的验证概率；
不声称有限小样本一定精确 90/10。稀有分组需要审查覆盖和比例，不能通过挪动已有样本
或挑选更好看的 seed 强行配额。无训练/验证录制时明确报错，而不是暗中换边。

实际录制身份、episode 映射和两侧列表保存为输出及每个 checkpoint 的 `dataset_split.json`。
普通 resume 校验协议、比例和录制成员，变化必须显式 fork 更新数据集；
老 checkpoint 没有新协议凭据时，不能自动声称其新 hash 验证集是历史 held-out 数据。
历史实验诊断使用其原始代码快照/明确保存的 episode 列表，不新增旧切分算法兼容分支。

## 相机模型槽位

新训练统一将机身 `observation.images.base_0_rgb` 放在腕部相机之前，其余相机按 key
确定性排序，空相机放最后；不继承数据集 JSON 对象的插入顺序。
已有任务 checkpoint 的 resume、继承 schema 的分支训练和推理，均恢复其自身
`pi05_deployment_metadata.json → observation.camera_keys_in_model_order`，因此历史
grasp 的腕部优先顺序仍保留。相机集合变化或 metadata 缺失/重复会报错，不能静默换序。
portable artifact 路径重写保留 JSON key 顺序；模型执行顺序另由显式列表保障。
仅更新代码即可正确加载历史 checkpoint，无需修改原 checkpoint。

## 本轮编排入口

`TRAINING_JOBS_FILE` 可选择本目录维护的 `object_grasp_training_jobs.json`、
`object_grasp_camera_fixed_jobs.json` 或
`all_scenes_simulation_fork_jobs.json`，支持 `staff3_grasp`、`all_scenes_simulation` 数据源
。正式任务共用唯一 `training_profile.json`，当前为 full MEM；任务文件禁止覆盖运行模式。
`resume-task` / `resume-dry-run-task` 可额外传明确的
checkpoint 路径；`run_dir_name` 固定原 run，避免误选最新同名任务。
历史任务语义或 artifact 身份的接纳必须显式设置对应 `allow_resume_*_adoption`，
仍校验动作、状态和记忆协议。`sync_full_checkpoint.sh` 与
`sync_full_checkpoint_chunks.sh` 用于包括 optimizer/RNG 的完整恢复点传输并校验哈希。

## 可迁移 checkpoint 基座

新 checkpoint 通过 `pretrained_model/base_artifacts.json` 记录基础 policy、MEM-ViT 和
tokenizer 的逻辑 ID 与 SHA256，不把训练机器的绝对路径当作模型身份。各训练节点用
`training_plan/base_artifact_registry.json` 把这些 ID 映射到本机 `/data` 路径；拉到
本地部署时则使用部署目录自己的 registry。加载前会校验实际文件内容，路径不同可以
正常运行，内容不一致会明确失败。

MEM-ViT 使用 LoRA 或 frozen 模式时，新 checkpoint 不再重复保存冻结的 MEM 基座权重；
真正 full fine-tune MEM-ViT 时仍会保存已训练权重，避免丢失更新。没有
`base_artifacts.json` 的历史 checkpoint 继续使用原有嵌入权重/路径逻辑。

## 训练语义与单次任务配置

两种配置具有不同的生命周期，因此分开维护：

- `training_plan/semantic_training_presets.json` 是长期维护的快捷模板库，只定义可复用的
  `state_weights` 和完整 `state_instruction_matrix`；
- `training_plan/training_jobs.json` 是本轮训练任务配置，选择数据集、记忆模式和训练语义；
- `training_plan/.env` 保存本轮任务共享的步数、batch、模型路径等运行参数。

模板库目前提供 `grasp`、`stage1_only`、`stage2_only`、`stage3_only`、
`three_stage_joint`、`three_goal_conditioned`、`three_goal_plus_full`，以及拉门专用的
`pull_stage1`、`pull_stage2`、`pull_stage3`、`pull_joint`。
每个快捷项自身包含完整矩阵。

后续正式训练统一按以下分层配置，由 `cluster.sh → launch.sh → train_vla_pi05.sh`
执行：可复用的训练语义放在 preset，任务 JSON 引用 `training_preset` 并选择数据源、
MEM、sidecar、训练步数、机器/GPU；共享参数在 `.env`，内容身份在 `artifacts.lock`。
新训练组织方式优先新增或复用 preset，不在任务文件中重复矩阵或手工拼接训练命令。
内联矩阵保留为兼容能力，与 preset 使用同一解析和验收链路。

`grasp` 定义 `object_grasp × object_grasp` 的 1×1 矩阵，使用 demonstrated 专家动作和
`goal_end` 完成点；两份正式 grasp 任务均引用 `"training_preset": "grasp"`。
数据来源 Staff3 和 GPU 分配属于任务配置，不写入 preset。grasp 与开门使用同一
`training_profile.json`：图像历史 1–6 帧、连续状态历史 13 帧，禁止任务级 `memory_mode`。
旧 grasp 的 image-MEM checkpoint 保持历史身份，不能无条件 resume 成 full MEM。
实际模型创建并应用 PEFT 后、任何 optimizer update 前，统一入口核验全部模型配置、
共享训练参数、数据处理配置和 adapter 参数名/形状指纹；只有任务语义、数据身份及
数据相关标签比例等显式列出的差异可以保留。失败直接退出，不会继续训练。
输出及 checkpoint 的 `training_profile_audit.json` 保存核验结果。
未来 grasp 与开门混合训练同样通过数据/sidecar 和 preset/任务配置组织；需先准备兼容的
混合数据和采样语义，本次不新增该混合方案。

### 拉门配置

| preset | 物理阶段 × 指令 | 用途 |
|---|---|---|
| `pull_stage1` | `approach × goal_approach` | 接近门 |
| `pull_stage2` | `handle_press × goal_open` | 操作门把手；`handle_press` 沿用统一三阶段 schema 的阶段键 |
| `pull_stage3` | `traversal × goal_enter` | 拉门通过 |
| `pull_joint` | 上述三条关系，阶段权重 1:1:1 | 分阶段联合训练，各阶段只用对应指令 |

这四项均采用 demonstrated 动作和 `goal_end`。任务指定
`"dataset_family": "pull_door"`、`"training_preset": "pull_joint"`（或某个单阶段），
由 `.env` 的 `PULL_DOOR_DATASET_ROOT` 选择拉门训练集，仍走统一入口。
矩阵中的阶段/goal 是统一语义键，推/拉方向必须来自拉门自己的同 episode 标注及动作；
不复用推门语言，也不将独立录制的三个阶段拼成伪全流程 episode。

2026-10-21 本地三份独立拉门数据均已完成清洗：`cleaned/pull_door_approach`
保留 319 条、`cleaned/pull_door_handle` 保留 206 条、`cleaned/pull_door_traversal`
保留 102 条。三者均有 v2 `stage_boundaries.json`（含完成点），且
`diagnostics/independent_door_cleaning_audit.json` 状态为 ok；对应原始标注各有
322/295/105 条 ok 记录。`pull_door_handle` 是历史 `complete_pull_door` 的正式名称，
仅表示独立操作门把手录制；`full_episode_pull` 标注 stage 不代表跨越三阶段。
统一 `training/pull_door` 已生成并验收：627 条、556489 帧，三份 cleaned 与统一集均有
`semantic_views.json` 和四档 `task_variants.json`。语言按原始录制路径精确匹配，禁止使用
清洗前的 episode 编号回退；1254 个视频为源数据硬链接，state/action 不变。
正式准备入口为 `data_postprocessing/lerobot/prepare_pull_door_training.sh`（已有输出拒绝覆盖），
验收结果在统一集 `diagnostics/pull_door_training_audit.json`，四个 preset 及全部完成点均通过。
本次仅准备本地数据，未新增远程任务或启动训练；选择资源后仍须执行
`dry-run-task` 再训练。`pull_joint` 不等同于跨状态目标条件训练，本次不新增
拉门的跨状态或全流程 command 预设。

`three_goal_plus_full` 保留 `three_goal_conditioned` 的全部关系，并在三个物理状态中
加入 `full_episode_command`。完整流程语言来自同一 episode 的四档独立标注；在
approach/handle_press 中以 `future_goal_end` 延续到入室完成点，在 traversal 中使用
`goal_end`。默认按允许的状态×指令组合等权采样，四档语言在选中组合内采样。

### 所有 preset 共用的采样规则

先选择合法的“物理状态×指令”池，再在池内采样物理帧；sampler 选中的指令会随索引
传到 batch，语言物化不再重新抽另一个指令。默认 N 个组合各占 1/N：
`three_goal_conditioned` 为 9×1/9，`three_stage_joint`/`pull_joint` 为 3×1/3，
`three_goal_plus_full` 为 12×1/12，单阶段与 grasp 为 1×1。不同阶段时长、episode 数、
可用指令数量或语言改写数量，不改变组合的目标概率；每 epoch 配额仅有整数取整差异。
`state_weights` 是组合权重的显式状态系数，默认全为 1；`semantic_source_weights` 如配置，
只调整各组合内部的数据来源比例。某个配置组合在整个训练集中没有合法样本时明确报错，
不虚构标签，也不静默把概率分给其他组合。

不要求整个 episode 或三个 goal 全部有效：保留独立有效的物理阶段和已有语言关系。
独立 Stage2 可加入有依据的 Approach/complete、Open/active，不补造 Enter 完成点。
`stage_boundaries.json` 是物理有效性依据，语言缺失不等于该阶段的观测无效。
只有当前样本实际请求的相机/MEM、state/action 历史及未来动作窗口涉及的物理阶段需要有效；
相邻有效阶段可以跨越，无效间隔不能跨越。随机 MEM 间隔在合法历史范围内采样，
不使用最大可能历史长度一刀切；固定窗口不足的边界样本不进入池。
目标已完成后的 hold 是生成的监督，不再读取无效阶段的未来专家动作。
标签先验按同一组合采样分布计算，归一化统计遍历合法指令配对而非只取第一条指令。

Semantic 训练默认排除“当前指令未 complete，且动作首帧 B2 三维速度精确为 0、
机械臂 inactive”的帧×指令配对。只看首帧，不检查夹爪、reset 或后续是否运动；
complete 依当前指令的完成边界判定，不使用其他阶段的原始 complete 标记。
同一物理帧可保留 Open/complete 而排除 Enter/未完成；没有合法指令的物理帧不会进入
采样池。组合的目标权重保持不变，只在各组合的合法配对中重采样。
验证集不做这项首帧静止过滤，但仍检查物理窗口合法性。
配置保存为 `dataset.semantic_exclude_incomplete_static_starts=true`；启动脚本支持
`--exclude-incomplete-static-starts=false` 以复现旧规则。旧 checkpoint resume 若因此
改变标签比例，仍须显式使用原有“更新数据集”续训入口，不绕过一致性校验。

Staff1 三目标筛选版分支任务维护在 `staff1_goal_filtered_fork_jobs.json`：wholebody6
四卡、full MEM、`three_goal_conditioned`，从与 `20261015_180800` 相同的
`20261008_105154_mem_pi05_b2_z1_vla_training_job_staff1_joint_full_mem_arm_mode_complete/checkpoints/050000`
完整恢复点训练到 100000。使用 `TRAINING_JOBS_FILE` 选择该计划，通过
`cluster.sh fork-dry-run-task` / `fork-task staff1_goal_conditioned_filtered_full_mem CHECKPOINT`
执行，建立新 W&B run 和输出目录，保留 optimizer/scheduler，不覆盖历史 run。
筛选开关沿用当前 `DatasetConfig` 的 true 默认值；启动后核对实际保存配置。
2026-10-21 已从统一 `training/staff1/meta/semantic_views.json` 重建专用 goal sidecar，
保留 2284 个有有效阶段的 episode（含 1064 个独立 Stage2）；当时已启动任务的旧划分
为 2055 条训练、229 条验证，九个组合各占 1/9。新任务改用下面的 recording_hash_v1，
不会通过修改本地代码改变已运行的远程快照。
旧的“全部 goal 齐全”侧车会被 preflight 拒绝。远程启动仍必须通过上述统一入口验收。

新任务同时指定以下两个字段（不会自动更换现有任务的 sidecar）：

```json
{
  "training_preset": "three_goal_plus_full",
  "semantic_views_file": "semantic_views_three_goal_plus_full.json"
}
```

专用 sidecar 使用 `data_postprocessing/lerobot/build_goal_conditioned_semantic_views.py`
构建，支持重复 `--full-episode-annotations` 来传入多个来源的 `annotations.json`：

```bash
python3 ../data_postprocessing/lerobot/build_goal_conditioned_semantic_views.py \
  --source DATASET/meta/semantic_views.json \
  --full-episode-annotations ANNOTATIONS/full_episode/full_episode_command/staff1/annotations.json \
  --output DATASET/meta/semantic_views_three_goal_plus_full.json
```

构建器精确匹配 `episode.source_path` 和标注 `video_path`；逐阶段保留可用的 goal，
不要求三个 goal 或全流程标注齐全。完整流程指令复用同一 episode 的 `goal_enter` 区间和完成边界，但保留
完整流程自身的语言，不从其他 episode 或 `goal_enter` 语言补缺。原始 sidecar 和视频
均不改写。输入必须是当前训练根目录的 canonical sidecar，不应只输入 cleaned Staff1
完整录制而遗漏统一集中的独立 Stage2。旧构建策略或 canonical 来源哈希不匹配时拒绝启动。

单次任务支持两种互斥、等价的配置形式。使用快捷配置时，在
`training_jobs.json` 中只引用名称：

```json
{
  "dataset_family": "staff1",
  "training_preset": "stage1_only",
  "resources": {
    "node": "f",
    "gpu_ids": [0, 1, 2, 3],
    "main_process_port": 29500
  }
}
```

任务默认读取数据集的 `meta/semantic_views.json`。当一种训练语义需要经过正式构建器
筛选出的专用候选视图时，可在任务中显式填写同一 `meta/` 目录下的文件名，例如
`"semantic_views_file": "semantic_views_three_goal_conditioned.json"`。该字段只选择语义
侧车，不复制或改写物理数据；入口仍会对专用侧车执行逐 episode 的完整矩阵验收。

不使用快捷配置时，在 `training_jobs.json` 的该任务内直接填写完整权重和矩阵：

```json
{
  "dataset_family": "staff1",
  "resources": {
    "node": "f",
    "gpu_ids": [0, 1, 2, 3],
    "main_process_port": 29500
  },
  "state_weights": {"approach": 1.0},
  "state_instruction_matrix": {
    "approach": {
      "goal_approach": {
        "status": "active",
        "action_supervision": "demonstrated",
        "completion_boundary": "state_end"
      }
    }
  }
}
```

同一任务不得同时写 `training_preset` 和内联矩阵。增加普通训练任务只修改
`training_jobs.json`；只有新增值得长期复用的语义组合时才修改 preset 库。两种形式最终
经过同一个解析、sidecar 校验和训练入口，不存在两套训练语义。

数据侧的全部候选语言视图仍由数据集的 `meta/semantic_views.json` 提供；训练矩阵是
其严格的运行时白名单。运行时直接按照“物理状态 × 规范化语言意图”选择视图，不使用
任何视图分类简称。`launch.sh` 在开始训练前会逐 episode 校验
sidecar；每个已有关系的状态、监督方式和完成点必须匹配，重复监督直接报错。
允许某条 episode 缺少部分关系，只采其合法配对；全局配置组合没有监督则报错。
sidecar 中未被矩阵选择的其他候选
语言不会进入训练。

查看某个任务的完整语义而不启动训练：

```bash
bash training_plan/launch.sh describe staff1_stage1_full_mem
```

当前任务计划直接配置三条关系：`approach × goal_approach`、
`handle_press × goal_open`、`traversal × goal_enter`。即使这些语言视图在不同 episode
的 sidecar 中来自不同历史分类，训练选择也只取决于这里明示的状态—指令关系。

每个任务可在 `training_jobs.json` 里通过 `train_steps` 独立指定训练终点；省略时才使用
`.env` 的共享默认值。入口按任务的 `gpu_ids` 数量自动计算梯度累积，不同 GPU 数量的
任务仍保持相同的全局 batch size。Staff1 统一数据集同时包含完整 Staff1 episode 与
单独采集的 Stage2 episode。

## 一键流程

```bash
# 查看机器、GPU 与本轮6个任务的固定分配
bash training_plan/cluster.sh describe

# 校验镜像、代码快照、数据集、基础模型和 MEM 权重
bash training_plan/cluster.sh prepare all

# 训练运行期间只发布新代码快照，不执行GPU探针或大文件哈希
bash training_plan/cluster.sh sync-code all

# 首次启动本轮6个任务
bash training_plan/cluster.sh dry-run-plan
bash training_plan/cluster.sh start-plan

# 异常中断后，从各任务最新完整 checkpoint 续训
bash training_plan/cluster.sh resume-dry-run-plan
bash training_plan/cluster.sh resume-plan

# 只验收并恢复一个任务；用于不同任务独立到达迁移点的场景。
bash training_plan/cluster.sh resume-dry-run-task TASK_ID
bash training_plan/cluster.sh resume-task TASK_ID

# 前台检查训练进程、日志错误、checkpoint 和磁盘空间
bash training_plan/cluster.sh status
```

`prepare` 会生成带内容哈希的只读版本标识，传输当前工作区内所有已跟踪文件，
以及本目录的正式编排脚本。训练采用“一任务一容器”，训练进程是容器主进程，
不使用云平台上不生效的 `docker exec`。容器入口会等待训练结束，并在容器停止时
把终止信号转发给完整训练进程组，Docker会保留120秒的优雅退出窗口；普通手工
入口仍保持原有后台启动语义。

`resume-plan` 只接受包含完整 `train_config.json` 和原 W&B `run_id` 的 checkpoint。
续训不创建新 run，而是使用 `wandb resume="must"` 接回原 run。续训校验复用首次
`prepare` 已验证的CUDA环境，不会在其他任务运行期间额外启动GPU探针。

两个 `dry-run` 命令使用一次性、无 GPU 的容器，逐项覆盖本轮全部6个任务。它们复用
正式镜像、挂载点、代码快照、任务映射和入口参数，但不会载入模型或启动训练。
`resume-dry-run-plan` 还会逐项读取真实的最新完整 checkpoint，验证优化器状态、原
W&B `run_id`、`resume="must"` 参数，以及续训时不会被入口默认值改写训练总步数。

## 监控与 F 盘保护

监控由 Codex 前台低频执行 `status`，不安装后台脚本、定时器或自动重试服务。
发现 wholebody_F 可用空间低于 `.env` 中的阈值后，再显式执行：

```bash
bash training_plan/cluster.sh archive-f-checkpoints
```

该命令优先处理非最新两份的整万 checkpoint；若空间仍低于目标且整万点已耗尽，
再处理每个任务次新的完整滚动 checkpoint。每个任务的远端最新完整恢复点永远保留。
所有候选会先在 F 的同一文件系统内建立原子硬链接快照，再从该稳定快照同步到本地。
这样训练进程即使在传输期间按滚动保留策略删除原 checkpoint，也不会造成半份传输。
远端快照和本地文件会逐文件做 SHA256 比对；完全一致后，才精确删除对应原 checkpoint
及其临时快照。
