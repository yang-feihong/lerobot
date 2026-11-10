# VLA 训练编排

用户可调配置统一维护在 `training_plan/.env`；镜像、模型的精确哈希
维护在 `training_plan/artifacts.lock`，数据身份由发布物自身的 READY 凭据锁定。入口不读取隐藏的机器默认值，也不依赖
远程工作区的未提交修改。

## 数据配方与不可变发布

任务用 `data_recipe` 选来源组合，用 `training_preset` 选监督矩阵，两者独立。
配方唯一维护在 `data_postprocessing/config/pipeline_sources.json`，训练不另维护目录映射。
每个逻辑配方只有一个 `training/<recipe>` 入口，指向 `release_history/` 中有验收凭据的不可变内容，包括单组件配方。训练解析真实内容路径并锁定；代码更新和监督 preset/权重变化不复制数据。数值、语言或阶段语义确实变化才产生历史内容版本。

```bash
bash training_plan/prepare_dataset.sh --task TASK \
  --datasets-root /data01/yangfeihong/MEM_videos/datasets/b2_z1
```

可用 `--jobs training_plan/某计划.json` 选择维护中的计划。该入口使用数据处理容器，
不需要 GPU、不调用标注模型：展开 preset、登记配方请求、验收全部组件和矩阵覆盖、
用已有 merger 复用视频硬链接，最后原子发布唯一 canonical sidecar、READY 凭据及索引。
缺失/过期组件须回数据流水线 reconcile 或补标注，不允许换路径绕过。
已登记配方在后续标注绑定成功后自动刷新；失败时索引保持 STALE，新训练拒绝使用。
发布后不原地更新。数值、metadata 的内容校验与接收视频的路径/大小检查写入凭据；
这不等于每次启动重新解码/计算所有视频 SHA。

`launch.sh` 按配方和展开后的 preset 自动解析 `training/releases.json` 并复验凭据与矩阵，
仅使用发布物的 `meta/semantic_views.json`。不能手填平行 sidecar 或直接读 cleaned。
原 `dataset_family` 只保留旧配置读取兼容，正式计划统一使用 `data_recipe`。
远程使用前同步相应完整发布目录和生成索引，再走现有 `cluster.sh check/dry-run`；
本地准备不自动传输数据，`cluster.sh prepare` 也不替代本地数据发布。
远程发布解析预检使用已锁定训练镜像中的 Python，以只读挂载保留宿主路径；
不依赖训练节点可能较旧的系统 Python，解析失败仍阻止启动。

普通 resume 从 checkpoint 的 `dataset.root` 锁定原不可变版本，不随索引换数据；
变更配方、监督矩阵或采用新数据必须显式 fork。旧 checkpoint 指向 cleaned 或历史可变
training 目录时不自动迁移，也不修改 checkpoint；应保留历史代码快照续跑或显式 fork。
以下历史实验的路径、专用 sidecar 和数据数目仅为当时记录，不是现行准备步骤。

## 稳定训练/验证划分（recording_hash_v1）

### 仿真 / 实采比例（训练配置，不是数据配方）

同一正式发布物可重复用于任意比例，无需重建或复制数据。在 job 中增加
`"domain_weights": {"simulation": 3, "real": 1}` 即为仿真 3:实采 1；
也支持在 `semantic_training_presets.json` 的某个 preset 中设置同名默认值，job 显式值优先。
`{"simulation": 1}` 为仅仿真，`{"real": 1}` 为仅实采；未写的域或权重为 0 的域不参与。
新训练不设置 `domain_weights`（配置中保存为 `null`）时，默认每个状态×指令组合内
仿真∶实采为 **3∶1**；组合内只有仿真或只有实采合法样本时，全部采该来源。
可用域在历史窗口、静止帧等合法性筛选后确定，不依据整个数据集的来源总数。
显式 `{}` 保持历史按合法帧自然混合，不强制域比例；显式数值权重仍严格要求所选域覆盖。
所有数值须有限、非负且总和为正。

采样顺序是：状态×指令组合 → 来源域 → 可选 source_group → 合法帧。
任务可用 `instruction_domain_weights` 显式覆盖某指令的来源域策略，例如
`"domain_weights": {"simulation": 3, "real": 1}` 配合
`"instruction_domain_weights": {"full_episode_command": {"simulation": 1}}`，
表示 full 组合只采仿真，其余组合仍为 3:1。组合等权不变；六组合时整体为 7:1。
覆盖项必须引用已配置指令、非空且不能启用全局排除的域；不自动容忍缺失域。
策略写入 config/runtime，普通 resume 禁止悄悄变更，需显式 fork。
因此默认组合等权不变，3:1 是**每个组合内部**的比例，而非整个数据集 episode 数配额；
小域会重复采样，有限 batch 允许取整误差，不承诺每个 batch 恰好 3:1。
显式权重下，任一启用组合在所需域中无合法帧（包括历史过滤/窗口检查之后）就报错。
只有上述自动默认策略允许单域组合自动使用唯一来源；显式逐指令来源限制不自动放宽。
某个域根本不存在时，不应选择要求该域的混合策略。

数据处理负责保存逐 episode 不可变来源证据：转换 manifest 的 `data_domain=simulation|real`；
历史发布物无需改写：`source_type=accepted_simulation_shard` 是仿真，ROS bag 原始录制后缀
`.bag/.bag.active` 是实采。训练加载时以 episode 编号和原始 `bag_path` 与 sidecar 精确连接并验冲突；
不读取原始挂载、不根据目录名或网页 tag 猜测。现有发布物已具备这些字段，因此没有迁移数据实体的必要。

零权重域在数据加载、hash 划分和动作统计前排除；其余录制保持原 hash 归属，不重洗牌。
训练 complete/布尔 loss 先验使用实际联合采样质量计算。连续动作归一化仍是所选录制/合法关系的
确定性数据统计，不随正权重比率重复物化样本。已有发布物中的 state 等固定归一化统计
保持原发布契约，不因比例切换改写。验证沿用启用域中的 held-out 录制自然分布，
不按训练比例重采样；跨比例对比指标时须核对验证域组成。
运行 `semantic_view_runtime.json` 记录域策略、各组合/域实际权重及合法配对数，checkpoint config 保存策略。
普通 resume 不允许改变域权重，调整比例应显式 fork，保持数据发布身份不变。
旧配置缺少域策略或保存 `{}` 时继续自然混合，不因新默认值变化而改写；普通 resume
沿用已保存策略，新训练/fork 才采用新默认。运行审计中的 `domain_sampling_policy`
明确标记自动 3∶1，`sampling_groups` 保存每个组合/来源实际权重。

这与 `dataset.image_source/mixed_sim_probability` 的图像替换增强完全独立，不能用图像增强概率代替录制来源比例。

多场景推门 Stage2 的 150k→200k 分支续训维护在
`all_push_stage2_resume_jobs.json`：锁定 10908 条正式发布物，使用 `stage2_only`，
仿真∶实采为 3∶1，wholebody6 四卡。用 `fork-dry-run-task` 验收后再用
`fork-task`，显式指定 20261019 的 `checkpoints/150000`；保留模型、优化器和训练状态，
新建输出与 W&B run，旧模型的 reset selector 结构保持不变。

11589 条新版多场景推门 Stage2 的四卡 0→100k 任务维护在
`all_push_stage2_fresh_100k_jobs.json`，锁定 READY artifact `cca3f6e…`，使用 `stage2_only`。
从基座全新初始化，不读取上轮模型、配置或优化器；使用当前共享 full-MEM profile、
默认 physical 工具轴、默认组合内仿真∶实采 3∶1（单域自动适配），历史指令输入关闭。
通过 `TRAINING_JOBS_FILE=training_plan/all_push_stage2_fresh_100k_jobs.json`
选择计划，再执行 `cluster.sh prepare wb6`、`dry-run-task`、`start-task`；
reset 监督放在发布物之外，并仅屏蔽已审计未知目标的分类 loss。

2026-11-08 已停止上述误配的 Stage2 新训。当前推门全流程任务改用
`all_push_joint_plus_full_fresh_100k_jobs.json`，任务
`all_push_joint_plus_full_sim3_real1_fresh_100k`：沿用同一 11589 条 READY 发布物，
`three_stage_plus_full` 的六个状态×指令组合等权，组合内默认仿真∶实采 3∶1，
单域组合自动使用唯一来源。wholebody6 四卡从基座重新训练 0→100k，
不加载已停止的 Stage2 权重/优化器，不启用历史指令输入，新建输出和 W&B run。
旧 Stage2 计划与产物仅保留作历史记录，不能冒充全流程六组合训练。

远程 dry-run 包含正式发布物解析和大数据集的语义前置检查；编排等待上限由
`.env` 的 `CLUSTER_DRY_RUN_TIMEOUT_SECONDS` 配置，默认 300 秒。
超时会输出诊断容器日志后停止，不启动训练，也不会绕过任何数据验收。

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

### W&B 本地代理与私有反向隧道

正式 `start/resume/fork` 在创建训练容器前自动调用 `wandb_proxy.py`，检查本地代理，
建立本地 systemd 用户服务维护的 SSH 反向隧道，并实测远程 W&B API；失败则不启动训练。
`.env` 中 `CLUSTER_WANDB_PROXY_ENABLED=true` 启用，`CLUSTER_WANDB_PROXY_LOCAL_PORT=18000`
是本地 HTTP CONNECT 代理，`CLUSTER_WANDB_PROXY_REMOTE_PORT=18001` 是每台远程的 loopback 监听端口。
容器使用 host 网络并注入大小写 HTTP(S)_PROXY，localhost 绕过代理；HF 仍保持离线。
这不开放公网代理。host 网络下同一节点的 DDP 端口须唯一，现有计划校验负责检查。

隧道独立于提交命令存活，SSH 每 15 秒保活，失联后退出，systemd 每 5 秒重连；
同一节点重复提交复用服务。首次接入会对当前本地用户启用 systemd lingering，使隧道在登出后仍存活；
若系统不允许启用则启动报错，不降级为随会话消失的后台进程。
日志用 `journalctl --user -u lerobot-wandb-proxy-<identity>` 查看，
`proxy-up NODE` 会显示精确服务名，也可独立补建隧道。停止服务用 `systemctl --user stop 服务名`，
但该节点仍有训练时不要停止。服务不依赖当前终端，但依赖本地机器、代理与用户服务管理器；
当前 transient unit 不保证本地重启后自动恢复，重启后执行 `proxy-up NODE`，无需重启已配好代理的训练。

既有无代理的训练进程不能通过修改环境文件热切换。迁移使用：

```bash
TRAINING_JOBS_FILE=training_plan/原计划.json bash training_plan/cluster.sh proxy-resume-task TASK_ID
```

入口先验隧道和完整恢复点，再停止该任务，从最新完整 checkpoint 用原源码快照的
`launch.sh resume` 恢复；配置、优化器、RNG、W&B run 和目标步数沿用，不能换成当前工作区训练实现。
可能丢失最新 checkpoint 后尚未保存的更新（正常小于一个保存周期）。原容器日志归档至远程
`lerobot/logs/`。原快照不能删除；此命令不触碰其他任务。普通新提交依然使用当前工作区内容快照。

### 只复用已训练模型，重新建立训练配置

任务可声明 `model_init_checkpoint`，再用普通 `start-task` 启动：该模式仍由统一入口创建
当前 profile、优化器、scheduler、RNG、数据处理器和 W&B run，只加载源 checkpoint 的
PEFT 模型 tensor；旧训练配置和优化器不加载。起点读取源 `training_step.json`，
`train_steps` 是全局终点，例如 100k 模型到 200k 表示本轮新增 100k 次更新，学习率重新 warmup。
输入/输出语义、基座身份和所有旧 tensor 的名称/形状/恢复值逐项核验；只允许显式新增的
reset selector 保持新初始化，不可静默丢失旧权重。普通 `fork-task` 仍是另一种语义：继承旧配置和优化器。

`dataset_artifact_id` 可显式锁定已发布版本；同一数据支持多种矩阵，训练启动独立检查所选矩阵覆盖，
不修改原发布物、接收凭据或全局发布索引。未显式 pin 时仍要求请求与发布物严格一致。
附加 reset 监督可通过任务的 `reset_preset_sidecar` 放在发布物之外，仍绑定数据 SHA256；
正式生成器 `prepare_reset_presets.sh --output ... --write` 支持此方式，不破坏不可变数据。

本次 Staff1 六组合八卡任务维护在 `staff1_joint_full_fresh_jobs.json`：
`TRAINING_JOBS_FILE=training_plan/staff1_joint_full_fresh_jobs.json bash training_plan/cluster.sh dry-run-task staff1_joint_plus_full_200k`，
验收后使用同计划的 `start-task`。训练输出保存 `model_initialization.json`、
`training_profile_audit.json`、`semantic_view_runtime.json` 和由真实训练 batch 采集的
`semantic_input_audit.json`（自然语言原文、tokenizer prompt、实际 token IDs 及解码结果）。
状态文本格式属于 checkpoint 的预处理契约，独立于连续状态历史输入。
部署和 resume 保留 `policy_preprocessor.json` 的状态 tokenizer 配置；历史空配置 `{}`
按原默认保留 `State: ...`，不能因 `state_action_encoding=continuous` 删除该文本。
新训练（非 resume）显式按配置选择状态文本格式，并随 processor 保存。
热切换比较保存的实际文本格式；文本格式不同则完整重载，不隐式改写旧模型输入。
数据传输用 `sync_training_release.py`，只发送接收凭据列出的训练文件及确实缺失的视频；
发布物中的本地软链接在传输时解引用为内容，远端不保留指向本地挂载的绝对链接。
远端相同视频须 SHA256 一致才硬链接，最终复验所有训练文件及视频内容，不传 raw_nonvisual 或诊断图片。
可用 `--parallel-transfers 16` 分组传输；每个文件只分配一个发送端，连接错峰建立、失败有限重试，
中断文件保留在 `.rsync-partial`，全部发送完成且 SHA256 验收通过后才发布 READY 入口。
已有 VERIFIED 接收凭据时，可用 `sync_training_release.py --expected-video-identities FILE`
传入按发布物相对路径索引的 `{path: {"size": N, "sha256": "..."}}` 子集，复用已验收的
期望哈希以减少本地视频重复预读；未列出的视频仍现场计算。此选项不跳过远端已有视频
或传输后内容的完整 SHA256 校验，任何差异仍拒绝发布。清单来源凭据须随任务证据保留。
代码同步通过校验和及 `--link-dest` 复用同节点已发布快照，仍生成独立的内容身份和文件校验清单。

拉门和外部抓取的四卡 50k 新训练维护在 `pull_external_grasp_50k_jobs.json`，分别引用
`pull_joint`（3×1/3）与 `external_grasp_joint_plus_full`（4×1/4），共用 full-MEM profile。

所有拉门 5122 条的四卡 0→50k 任务维护在 `all_pull_joint_full_50k_jobs.json`，
使用 `pull_joint_plus_full` 六组合等权。full 组合只采仿真，其余组合内仿真∶实采 3∶1；
这是显式任务采样策略，不补造独立单阶段实采的全流程监督。

2026-11-08 的 767 条外部抓取三组四卡对比计划为
`external_grasp_raw_3way_50k_jobs.json`，锁定发布物 `e6554faa…`，
沿用 `external_grasp_joint_plus_full` 四组合等权，全部从基座 0→50k。
三组均设置 `EE_SUPERVISION_SOURCE=recorded_action`、`MODEL_EE_FRAME=physical`、
`GLOBAL_BATCH_SIZE=48`、`HISTORY_PAUSE_MAX_STEPS=0`；每卡 batch=2，四卡累积=6。
每个任务必须使用**独立源码快照中的 `.env`** 固化下表配置，不能只切 TASK_ID 而共用同一环境：

| 任务 | 节点 / GPU | ACTION_HISTORY_ENABLED | EE_DELTA_REFERENCE |
|---|---|---|---|
| `external_raw_nohist_state_delta_50k` | wb2 / 0–3 | false | current_state |
| `external_raw_history_state_delta_50k` | f / 0–3 | true | current_state |
| `external_raw_history_command_delta_50k` | f / 4–7 | true | previous_command |

本轮源码快照与传输、预检、验收记录存于
`MEM_videos/workspace/codex_scratch/external_raw_3way_20261108.rglx38/`。
三份 `src` 内容哈希相同；差异只来自 `.env` 的历史输入开关和 EE delta 基准。
关闭历史输入时禁止 previous_command；有历史时指令基准取模型可见的最后一个历史槽，
保持/reset/启动 padding 时按统一契约回退到当前实测 EE，不使用未来标签。

EE 来源由一个配置同时控制动作监督与历史动作输入（包括对应的归一化统计）：
`EE_SUPERVISION_SOURCE=control_action`（默认）选择存储的 `action[16:25]`；
`EE_SUPERVISION_SOURCE=recorded_action` 选择存储的 `action[5:14]`。

已有 checkpoint 切换 EE 监督来源必须使用显式 `fork-task`，设置
`EE_SUPERVISION_SOURCE` 后会通过新 run、更新数据及更新训练目标三个条件验收，
继承模型/优化器/scheduler/RNG与累计步数，但重新统计新来源的动作归一化。
张量布局、工具轴、delta 基准和历史窗口不得同时改变；普通 resume 保持 checkpoint 来源，
不会被当前环境变量覆盖。禁止仅换 run 名而继续使用旧监督来源。
直接训练脚本对应 `--ee-supervision-source=recorded_action`，Python/CLI policy 配置对应
`--policy.ee_supervision_source=recorded_action`。它只切换 EE 的 9 个通道，
不改底盘、机械臂模式、夹爪或其他监督，也不自动开启历史输入。
`recorded_action` 指数据集已有列，并不保证每个来源都是 IK 前的 VR 请求；
已核对当前门/抓取仿真生产者：两段列均写入下发关节目标的 FK 结果，
不是两套独立的 IK 前/后目标；后处理可能仅改变控制列，使数值不同但不改变来源身份。
因此该开关不等于已统一实采与仿真的 IK 前原始指令语义。
历史 EE 的 inactive/reset 屏蔽规则仍保留；语义 hold 对两套 EE 列分别保持，不跨来源替换。
部署 metadata/config 保存来源；动作差分与精确动作统计均使用所选来源，
不同来源的统计不可混用。普通 resume 继承 checkpoint，不切换该语义。
部署的历史输入也应提供与所选来源对应的目标；不能将上游请求和下游控制目标视为等价。

精确动作统计仅由主进程执行。该 CPU 预处理使用独立的 Gloo 同步组（最长 6 小时），
不让其他 rank 在训练 NCCL barrier 中空等并触发 watchdog；训练 collective 原有超时保持不变。
主进程失败会通知其他 rank，不能继续读取不存在的统计文件。日志按 episode 进度及精确分位数
完成情况输出；本机制不改变统计样本、数值算法或普通 resume 的统计一致性检查。
新增的预处理运行模块已显式纳入源码快照清单（不依赖本地未跟踪文件），远程 dry-run
还会先导入实际快照中的训练模块，避免本地测试通过而部署漏文件后才在模型加载末尾失败。

固定基准 EE delta 支持 `EE_DELTA_REFERENCE=current_state|previous_command`，默认
`current_state` 保持原实测 EE 基准。设为 `previous_command` 时，必须同时设置
`Z1_ACTION_REPRESENTATION=ee_state_delta`、`ACTION_HISTORY_ENABLED=true`，且历史字段包含完整 EE 9D；
否则配置/启动直接报错。直接脚本参数为 `--ee-delta-reference=previous_command`，
policy CLI 为 `--policy.ee_delta_reference=previous_command`。
此模式用最后一个输入历史采样槽中的 EE 指令，作为整个未来块共同的绝对姿态基准，
不是上次推理输出的 50 帧，也不是旧 `ee_delta` 的逐帧累加；默认历史间隔 40ms，
因此该槽不一定等于 50Hz 控制器上一拍。基准来源跟随 `EE_SUPERVISION_SOURCE`。
启动历史不足、该槽未知或 EE 被 inactive/reset feature mask 屏蔽时，训练与部署
均回退到当前实测 EE，不使用被遮住的目标。暂停增强后的历史也使用同一规则。
动作统计按同一历史槽/来源计算并锁定参考模式与采样间隔；不同模式不能复用统计。
checkpoint 保存该配置，SYNC 自动按对应基准解码；RTC 还会将旧前缀的绝对目标
重新表达为本次基准下的 delta，避免换基准造成轨迹跳变。现有模型和普通 resume 不切换基准。

历史动作输入默认关闭。新训练可在统一入口的环境配置中显式设置
`ACTION_HISTORY_ENABLED=true`；普通 resume 始终继承 checkpoint，不在恢复时增删模型结构。
20261106 wholebodyF 的 `external_grasp_history_50k_jobs.json` 已用独立代码快照开启历史输入，
暂停增强为 0；当前 `.env` 已为下一轮无历史指令训练关闭历史输入，不影响运行中的 F 任务。
该 F 任务沿用 20261105 外部抓取任务的同一数据 artifact、四组合、八卡及
50k 目标，从基座 0 步新训，不加载上一轮权重或训练状态；旧任务计划仍保留。
`ACTION_HISTORY_FEATURE_NAMES` 可指定 JSON 字符串数组，默认 `null` 表示标准已执行命令：
按顺序为底盘速度 3D、inactive/reset 模式 2D、控制目标 EE 9D、夹爪命令 1D、
reset 最终关节目标 6D，共 21D；reset 六维具名为 `arm_reset_joint_target_1..6`。
历史 checkpoint 已保存的显式字段列表继续沿用（包括旧的 13D/15D 列表），不静默迁移；
切换字段属于新输入契约，不能按普通 resume 或热切换处理。不包含生成的 complete/blocked 标签，
也不随输出通道开关变化。EE 按字段名读取 `control_*`，统计量使用同一映射；不靠 16/25 维猜测。
新默认历史采用 `mode_reset_target_v2`：只有主动 Cartesian EE 控制（inactive=0 且 reset=0）
时使用 EE 目标；保持/reset 的九维 EE 历史在归一化后 mask 成零，模式、底盘/夹爪仍保留。
reset 六维仅在 reset 时有效：训练按历史采样帧读取已审计 `reset_presets.json`，
将对应 preset 映射为该发布物/catalog 的六关节目标；已审计未知目标仅屏蔽六维，
未审计漏标仍报错，不默认 middle。历史目标在暂停增强前绑定，随后与指令一起移动；
不含当前/未来帧的 preset。部署读取执行端下发/采用的最终目标，不取预测的未来类别序列。
模式固定归一化到 [-1,1]，reset 关节目标固定以 [-pi,pi] 作为归一化边界，不依赖输出维数。
这是“不提供该模式下的 EE 指令”，不是实际反馈或零位置目标；保持与 reset 由显式模式区分。
已保存的显式字段列表若缺语义版本，按 `legacy_control_v0` 保留旧行为。新自定义字段列表可显式
设置 `ACTION_HISTORY_EE_SEMANTICS=mode_reset_target_v2`；已有 `active_only_v1` 保留其原行为。
该语义版本随 checkpoint/input-spec 固化，
普通 resume 不切换；已有历史权重禁止用同形状掩盖语义变化。未来动作监督不受此 mask 影响。
`MAX_ACTION_HISTORY_DIM`（默认 32）是独立输入投影容量，不使用输出的 `max_action_dim`。
12 帧历史（默认）与 state 共享 25Hz 时钟，仅覆盖当前之前；语义 complete/hold 仅修改未来监督。
`HISTORY_PAUSE_MAX_STEPS` 默认 `0`，完全关闭暂停增强；正整数要求开启历史动作输入。
训练每个样本均匀抽取 `0..最大值` 个历史步（每步默认 40ms，设为 5 即最多 0.2s），
将短期状态与指令历史一起向更早槽位移动，截掉溢出历史，末端补理想保持。
状态重复当前快照（含原字段值，不模拟物理停稳）；指令底盘零速、机械臂 inactive、reset=0，
夹爪保持最后历史目标，EE 与 reset 目标组按保持模式屏蔽。padding 跟随来源，不把未知命令变成已知。
图像、当前状态/anchor、未来动作及全部输出监督不变。超过窗口的移动使历史全部保持；
验证和部署不执行增强，普通 resume 继承 checkpoint 配置。此增强不模拟接触/回弹动力学。
模型保存输入字段、容量、投影和独立归一化统计，部署按该契约选择实际执行记录；
缺字段报错，缺少启动历史则 padding/mask，不以实际状态冒充动作。
历史字段或容量变化须重建模型，不能热切换相同形状但不同含义的历史输入。
仅复用旧模型权重的新训练可以新增历史投影，保留原权重、新投影单独初始化；
已有历史模型禁止静默换字段语义。关闭开关时不新增投影、不改变旧部署 metadata/schema。

筐内抓取四卡 50k→100k 的权重复用任务维护在 `bin_grasp_fresh_50k_to_100k_jobs.json`，
使用 `grasp`（1×1）与共享 full-MEM profile，在 wholebody2 运行。
数据锁定逻辑入口 `training/bin_grasp_all_real` 当次的 READY 发布物；任务中的配方标识
与该发布物接收凭据一致，不按目录后缀推测内容。用 `start-task bin_grasp_full_mem_50k_to_100k`
进入：仅加载 20261022 抓取模型的 50k 权重，从全局 50k 开始，100k 为终点；
配置、优化器、scheduler、RNG、W&B run 和输出目录全部新建，不用普通 `resume-task` 继承旧配置。
任务固定数据 artifact 而非可重命名的发布入口；拉门仅对审计侧车列明的未知 reset 目标
屏蔽分类 loss，保留原动作与 complete 监督。

`TRAINING_JOBS_FILE` 可选择本目录维护的 `object_grasp_training_jobs.json`、
`object_grasp_camera_fixed_jobs.json` 或
`all_scenes_simulation_fork_jobs.json`，支持 `staff3_grasp`、`all_scenes_simulation` 数据源
。正式任务共用唯一 `training_profile.json`，当前为 full MEM；任务文件禁止覆盖运行模式。
`resume-task` / `resume-dry-run-task` 可额外传明确的
checkpoint 路径；`run_dir_name` 固定原 run，避免误选最新同名任务。
历史任务语义或 artifact 身份的接纳必须显式设置对应 `allow_resume_*_adoption`，
仍校验动作、状态和记忆协议。`sync_full_checkpoint.sh` 与
`sync_full_checkpoint_chunks.sh` 用于包括 optimizer/RNG 的完整恢复点传输并校验哈希。
仅拉取模型与配套 metadata 使用 `sync_eval_checkpoint.sh`，支持
`SYNC_PARALLEL_PARTS`；并行连接错峰建立，SSH 传输失败重试最多五次，
任一分片最终失败或 SHA256 不一致均拒绝报告同步成功，保留远程原模型。
同一 run 追加指令列须显式设置任务字段 `allow_resume_instruction_extension=true`：
状态集合不变、原有每条关系不变，只能追加新指令；不允许删除或改写旧关系，也不放宽
MEM、动作表示、输出结构和 artifact 身份校验。入口仍显式启用 updated-dataset 和
updated-training-objective，重新计算采样先验与动作统计，并保存新语义侧车。

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
`three_stage_joint`、`three_goal_conditioned`、`three_goal_plus_full`、`external_grasp_goal_plus_full`，以及拉门专用的
`pull_stage1`、`pull_stage2`、`pull_stage3`、`pull_joint`。
每个快捷项自身包含完整矩阵。

后续正式训练统一按以下分层配置，由 `cluster.sh → launch.sh → train_vla_pi05.sh`
执行：可复用的训练语义放在 preset，任务 JSON 引用 `training_preset` 并选择数据源、
MEM、训练步数、机器/GPU；共享参数在 `.env`，模型/镜像身份在 `artifacts.lock`，数据身份在发布凭据。
新训练组织方式优先新增或复用 preset，不在任务文件中重复矩阵或手工拼接训练命令。
内联矩阵保留为兼容能力，与 preset 使用同一解析和验收链路。

`grasp` 定义 `object_grasp × object_grasp` 的 1×1 矩阵，使用 demonstrated 专家动作和
`goal_end` 完成点；两份正式 grasp 任务均引用 `"training_preset": "grasp"`。
数据来源 Staff3 和 GPU 分配属于任务配置，不写入 preset。grasp 与开门使用同一
`training_profile.json`：图像历史 1–6 帧；连续状态历史上限 13 帧，训练时逐样本随机保留
1–13 帧有效后缀（始终保留当前帧、不越过真实 padding），验证和部署使用全部可用历史。
禁止任务级 `memory_mode`。
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
| `pull_joint_plus_full` | 对应阶段目标 3 条＋全流程 command 列 3 条，共 6 条关系，各占 1/6 | 三阶段联合＋全流程；与推门 `three_stage_plus_full` 相同矩阵 |

前四项均采用 demonstrated 动作和 `goal_end`。`pull_joint_plus_full` 的全流程列在靠近/把手阶段
使用 `future_goal_end`、通过阶段使用 `goal_end`；6 个状态×指令组合等权，不按时长或数据量混合。
任务指定
`"data_recipe": "pull_door"`、`"training_preset": "pull_joint"`（或某个单阶段），
由发布索引自动选择该配方的 READY 训练集，不维护拉门专用路径变量。
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
正式准备入口为 `data_postprocessing/run_pipeline.sh`（共享验收及不可变发布），
验收结果在统一集 `diagnostics/pull_door_training_audit.json`，四个 preset 及全部完成点均通过。
本次仅准备本地数据，未新增远程任务或启动训练；选择资源后仍须执行
`dry-run-task` 再训练。`pull_joint` 不等同于跨状态目标条件训练。

完整三阶段拉门录制使用后处理预设 `full_pull_door_episode`，与完整推门共享分段、清洗及完成点验收，
标注来自同一个 episode 的三个阶段 goal＋一个拉门全流程 command，每组四档语言。
独立采集的阶段仍使用原来的三个 pull profile，仅具有本阶段 goal，不能加入 full-command 列。
有无效阶段时保留其他有效阶段，但不生成全流程 command；不得拼接三个独立 episode 冒充完整录制。
在网页人工验收并合入新数据后，将它追加到拉门训练组合，再以
`"data_recipe": "pull_door", "training_preset": "pull_joint_plus_full"` 准备训练发布物。
新矩阵必须在全局存在 6 个合法配对；只有旧的独立阶段数据时会明确报缺少全流程样本，
不能伪造指令或静默退化为三组合。这里仅准备支持，不启动训练。

`three_goal_plus_full` 保留 `three_goal_conditioned` 的全部关系，并在三个物理状态中
加入 `full_episode_command`。完整流程语言来自同一 episode 的四档独立标注；在
approach/handle_press 中以 `future_goal_end` 延续到入室完成点，在 traversal 中使用
`goal_end`。默认按允许的状态×指令组合等权采样，四档语言在选中组合内采样。

### 分阶段联合＋全流程（不展开阶段目标的跨状态配对）

`three_stage_plus_full` 在 `three_stage_joint` 的三条对应关系上增加全流程列，共 6 个组合：

| 当前状态 | 对应阶段指令 | 全流程指令 |
|---|---|---|
| `approach` | `goal_approach` | `full_episode_command` |
| `handle_press` | `goal_open` | `full_episode_command` |
| `traversal` | `goal_enter` | `full_episode_command` |

`external_grasp_joint_plus_full` 使用靠近、抓取各自的阶段目标，再增加全流程列，共 4 个组合：

| 当前状态 | 对应阶段指令 | 全流程指令 |
|---|---|---|
| `object_approach` | `object_approach` | `full_episode_approach_grasp` |
| `external_object_grasp` | `external_object_grasp` | `full_episode_approach_grasp` |

每个单元格是一条允许关系：阶段目标模仿专家至本阶段目标完成点；全流程在前面的状态中
延续至全流程完成点（`future_goal_end`），末状态使用 `goal_end`。不加入其他阶段目标的
跨状态配对，也不加入已越过阶段目标的 complete＋hold 配对；选中关系内部到达完成点后的
hold＋complete 仍按统一机制生成。完成点复用已验收的 sidecar，不重新定义完成语义。
默认分别按 6 个组合各 1/6、4 个组合各 1/4 采样，四档语言在组合内采样。
两者走同一训练入口、采样和发布物覆盖检查；选择任务 JSON 中的 `training_preset` 即可。
数据需具备所选关系及全流程语言/完成监督；部分有效阶段和独立阶段仍遵循共享合法窗口规则。
既有 `three_goal_plus_full` 的 12 个组合、`external_grasp_goal_plus_full` 的 6 个组合保持不变。
新增 preset 不自动修改任何既有任务、发布数据或启动训练。

### 所有 preset 共用的采样规则

外部物体抓取使用 `"training_preset": "external_grasp_goal_plus_full"`，
与推门 `three_goal_plus_full` 同一解析/采样/监督链路，定义 2 类状态 × 3 类指令：

| 物理状态 | 靠近物体 `object_approach` | 抓取物体 `external_object_grasp` | 全流程 `full_episode_approach_grasp` |
|---|---|---|---|
| 靠近 `object_approach` | 专家动作，到靠近完成点 | 专家动作，延续至抓取完成点 | 专家动作，延续至全流程完成点 |
| 抓取 `external_object_grasp` | 全段 `complete=1` + hold | 专家动作，到抓取完成点 | 专家动作，到全流程完成点 |

hold 复用现有训练侧确定性保持逻辑：底盘速度为零，保持当前控制 EE 目标和夹爪开度，
不是把绝对位姿或夹爪清零；已完成目标不继续模仿当前正在进行的抓取动作。
状态/指令名称与数据侧抓取标注保持一致；本 preset 不改变原有筐内 `grasp` 1×1 定义。
默认六个状态×指令池各占 1/6，四档语言在选中的池内采样。
该模板不指定数据源/GPU，也不自动创建训练任务。正式训练需先提供通过完成语义审查、
覆盖这六个关系的 semantic views（包括跨阶段 active 和 complete/hold 视图）；
仅有语言标注或 stage_boundaries 不能直接替代。范围内 complete 遵守
`../../data_postprocessing/COMPLETION_SEMANTICS.md`，不以切段事件或阶段末帧伪造完成点。

先选择合法的“物理状态×指令”池，再在池内采样物理帧；sampler 选中的指令会随索引
传到 batch，语言物化不再重新抽另一个指令。默认 N 个组合各占 1/N：
`three_goal_conditioned` 为 9×1/9，`three_stage_joint`/`pull_joint` 为 3×1/3，
`three_goal_plus_full` 为 12×1/12，`external_grasp_goal_plus_full` 为 6×1/6，单阶段与 grasp 为 1×1。不同阶段时长、episode 数、
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
四卡、full MEM，最初使用 `three_goal_conditioned`，从与 `20261015_180800` 相同的
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

多场景实采＋仿真的跨阶段耦合任务维护在 `all_scenes_goal_mixed_jobs.json`：
wholebody F 八卡、共享 full MEM，最初使用 `three_goal_conditioned` 从基座初始化训练至 100000，
建立 W&B `5ic4dyp1` 和 `20261022_214058` 输出目录。使用混合集
`all_scenes_plus_simulation_auto_open_v1` 的专用语义侧车；
该侧车由上面的正式构建器从混合集 canonical sidecar 生成，不筛掉独立 Stage2 或部分有效阶段。
采用 `recording_hash_v1` 划分；每卡 batch 2、梯度累积 3，
八卡全局 batch 仍为 48。通过 `TRAINING_JOBS_FILE=training_plan/all_scenes_goal_mixed_jobs.json`
选择任务，再执行 `cluster.sh prepare f`、`dry-run-task all_scenes_goal_mixed_full_mem` 和
`start-task all_scenes_goal_mixed_full_mem`。

2026-10-23，上述两个任务均切到 `three_goal_plus_full`，显式选择
`semantic_views_three_goal_plus_full.json` 和 instruction-extension 许可，从各自最新完整
checkpoint 通过 `resume-task` 继续，原 W&B run、输出目录和 100000 目标不变；
Staff1 原 run 为 `20261021_205355` / W&B `cu9o541l`。
12 种状态×指令组合各占 1/12。两份侧车均从 canonical 重建，分别保留 Staff1 的
2284 条和混合集的 6236 条，不沿用要求全 goal 齐全的旧文件；其中 1088/4654 条分别具有
同 episode 全流程语言及有效 Enter 监督（仿真 2074 条全部覆盖），其余仍保留可用三目标。
全流程四档语言精确绑定原录制，完成点复用同 episode 的入室完成点；不补造独立 Stage2 的
全流程指令。两条任务均采用当前统一采样、时序合法性、静止首帧筛选及 recording hash 划分。

2026-10-24，Staff1 `three_goal_plus_full` 的后续目标调整为 150000，维护在
`staff1_goal_filtered_fork_jobs.json`。等待原任务完成 100000 并保存完整恢复点后，
通过统一 `resume-task` 从该点继续；保留原 W&B run、输出目录、12 组合和四卡资源。

Staff1 三阶段联合的 100k→200k 续训维护在 `staff1_joint_resume_jobs.json`：
wholebody2 四卡、`three_stage_joint`、共享 full MEM，恢复 `20261008_105154` 的完整 100k
checkpoint（模型、optimizer、scheduler、RNG），沿用 W&B `3cmk2ai6` 与原 run 目录。
数据划分和采样使用最新统一代码：`recording_hash_v1`、三个组合各 1/3、合法时序窗口和
未 complete 静止首帧筛选。通过 `TRAINING_JOBS_FILE=training_plan/staff1_joint_resume_jobs.json`
选择任务，用 `resume-dry-run-task` 验收后执行 `resume-task staff1_joint_full_mem CHECKPOINT`；
统一入口显式开启 updated-dataset / updated-training-objective，不使用旧数据顺序的复现语义。
新验证集可能包含前 100k 已训练过的录制，不能声称它是全程未见数据的独立 held-out 集。

新任务指定来源配方与监督 preset：

```json
{
  "data_recipe": "staff1",
  "training_preset": "three_goal_plus_full"
}
```

语言与全流程指令由数据流水线绑定，训练准备只消费验收后的组件，不手动拼标注路径。
仍按独立有效阶段保留关系，不要求每个 episode 三个 goal 齐全；全流程自身的语言和完成
证据必须齐全，不从其他 episode 或 goal_enter 文字补缺。发布器检查 preset 所需的全局组合覆盖。

单次任务支持两种互斥、等价的配置形式。使用快捷配置时，在
`training_jobs.json` 中只引用名称：

```json
{
  "data_recipe": "staff1",
  "training_preset": "stage1_only",
  "resources": {
    "node": "f",
    "gpu_ids": [0, 1, 2, 3],
    "main_process_port": 29500
  }
}
```

任务只读取自动解析发布物的 `meta/semantic_views.json`，不配置 `semantic_views_file`。

不使用快捷配置时，在 `training_jobs.json` 的该任务内直接填写完整权重和矩阵：

```json
{
  "data_recipe": "staff1",
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

## Reset 目标选择（front / middle / back）

新训练由统一入口默认启用 `action_predict_reset_preset=true`；原有 13D flow 动作、
`arm_reset` gate 和 complete 语义不变。目标选择使用独立的共享候选评分头与 CE loss，
只在语义视图物化后的有效 reset 帧计算，排除 padding 和 RTC 已知前缀。
无 reset 的抓取片段不产生此项 loss，也不会按任务名强塞 front/back 标签。

训练前在本地准备监督（正式入口，默认只审计，加 `--write` 才写入并刷新数据台账）：

```bash
bash ../data_postprocessing/lerobot/prepare_reset_presets.sh --dataset /绝对路径/数据集
bash ../data_postprocessing/lerobot/prepare_reset_presets.sh --dataset /绝对路径/数据集 --write
```

生成器从已有 `raw_nonvisual` 的 `/arm_target_pos` 控制目标识别完整 reset 操作，
仿真则从 manifest 明示的 accepted shard `raw_audit` 中读取
`arm_joint_target_dispatched`，先验证逐帧关节反馈与训练 episode 对齐，再识别原始
完整 reset 操作的最终下发目标，映射到清洗后的 reset 范围。反馈仅用于对齐，不用于猜目标。
可用 `--workers 4` 并行审计；原始证据仍只在本地读取，远程只同步生成侧车。
排除启动 latched 消息，复现转换采样原点；不读 rosbag、不使用实际关节状态或任务名猜目标。
末端命令需唯一匹配 preset（最多 0.03 rad），中断、未知目标或无法区分的切换保留为 unresolved。
训练若实际采到既无已识别目标、也未列入审计 `unresolved` 的 reset 会报错，不回退 middle。
统一默认 `reset_preset_unknown_target_policy: mask_audited_unknown`：生成器无法唯一辨认的
reset 目标记录到侧车 `unresolved`，训练自动仅屏蔽这些帧的 preset 分类 loss，
原动作、padding 和 complete 监督均不变，无需按数据集逐次确认。
未审计的缺失目标仍报错，不默认 middle、不屏蔽动作 loss，也不丢弃整个 episode。
显式 `error` 可用于严格审计；旧 checkpoint 显式保存的策略仍按普通 resume 契约恢复。
策略随训练配置保存，侧车随 checkpoint 保存。
`meta/reset_presets.json` 绑定转换 manifest 与 action parquet 的 SHA256；数据变化必须重新审计，
生成器拒绝覆盖旧侧车。数据不需重转、不传视频或 raw_nonvisual；远程只需同步该侧车及原训练数据。
checkpoint 保存侧车与 hash；普通 resume 不允许偷偷更换监督。

catalog 位于模型 config 的 `reset_preset_catalog`，ID 只在该 catalog 内作类别索引，
不进入归一化或 flow MSE。共享评分器按候选关节目标打分，候选数可扩展，参数量与 flow 维度不变。
扩展目标仍需真实监督与训练验证，不承诺未见过的新 preset 可以零样本可靠执行。
主线 profile 继续严格校验原参数结构，并独立校验新增选择头的六组权重。

旧 checkpoint 缺少选择头时仍可加载，reset 固定使用 middle；普通 resume 保持其旧结构，
不会静默创建随机选择头。新 checkpoint 声明有选择头却缺权重、缺 catalog 或输出非法 ID 时拒绝部署。
推理结果通过独立 `arm_reset_preset_id` 和 checkpoint catalog 传给执行端，不改变物理动作归一化。
切换 catalog/是否启用选择头会触发完整重载，不沿用不匹配的热切换。
ROS `VLAWholeBodyTarget` 当前为 v3（物理工具姿态 + preset 名称及六关节目标），
部署前需重建两侧消息与 Bridge/客户端；v1/v2 旧命令拒绝执行，不能与 v3 混用。
旧模型没有 selector 时，由新版客户端明确选择 middle，不能靠旧版 ROS 命令兼容层实现。
模型服务 TCP v2 统一处理历史模型的输入／输出工具轴转换，执行端不再重复补角。
提交代码不自动重启现有服务或训练，实际部署状态须单独核验。

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
# 模型工具轴选择与历史兼容

新训练默认 physical，无需手动选择。可在 `.env` 用 `MODEL_EE_FRAME=legacy|physical`、
直接脚本入口用 `--model-ee-frame=legacy|physical` 或 Python 的 `policy.model_ee_frame`
显式覆盖。留空时，新训练解析为 physical；普通 resume 始终继承 checkpoint，
不会把新训练的默认值应用到旧模型。
工具轴不属于数据配方或监督 preset；同一混合来源数据可供两种模型约定使用。

数据读取以 `meta/ee_frames.json` 的逐录制声明适配完整状态和动作历史，
在相对动作生成与归一化之前统一为本轮模型工具轴。直接遍历 HF 数值的统计入口采用
同一个适配器；需要变化的姿态统计从转换后数值计算，不套用另一工具轴的分位数。
数据文件不被改写，也不复制视频或派生另一套 training 数据。

新 checkpoint 的部署 metadata v11 必须包含 `model_ee_frame` 和
`ee_frame_contract=same_tcp_tool_rx_minus_90_v1`，并与 config/processor 一致。
历史 v10 无该字段的 checkpoint 在 resume/部署时按 legacy 解释；普通 resume
不能改变工具轴，显式不一致会报错。新格式缺失契约或非法值不按历史规则吞掉。
公共机器人服务始终输入输出 physical，服务按模型 metadata 决定是否补偿；
相对动作须在模型约定中先解码成绝对目标再转换，RTC/历史/anchor 使用相同约定。
历史发布物补齐全 legacy 元信息不改变原发布身份或 checkpoint 的锁定路径。
