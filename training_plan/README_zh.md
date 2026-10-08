# VLA 训练编排

用户可调配置统一维护在 `training_plan/.env`；镜像、模型和数据版本的精确哈希
维护在 `training_plan/artifacts.lock`。入口不读取隐藏的机器默认值，也不依赖
远程工作区的未提交修改。

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

模板库目前提供 `stage1_only`、`stage2_only`、`stage3_only`、
`three_stage_joint` 四个快捷项。每个快捷项自身包含完整矩阵。

单次任务支持两种互斥、等价的配置形式。使用快捷配置时，在
`training_jobs.json` 中只引用名称：

```json
{
  "dataset_family": "staff1",
  "memory_mode": "full_mem",
  "training_preset": "stage1_only",
  "resources": {
    "node": "f",
    "gpu_ids": [0, 1, 2, 3],
    "main_process_port": 29500
  }
}
```

不使用快捷配置时，在 `training_jobs.json` 的该任务内直接填写完整权重和矩阵：

```json
{
  "dataset_family": "staff1",
  "memory_mode": "full_mem",
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
sidecar，要求所选状态每一帧恰好对应一个矩阵允许的语言关系；该关系缺失、重复，
或其状态、监督方式、完成边界不一致都会直接报错。sidecar 中未被矩阵选择的其他候选
语言不会进入训练。

查看某个任务的完整语义而不启动训练：

```bash
bash training_plan/launch.sh describe staff1_stage1_full_mem
```

当前6路计划直接配置三条关系：`approach × goal_approach`、
`handle_press × goal_open`、`traversal × goal_enter`。即使这些语言视图在不同 episode
的 sidecar 中来自不同历史分类，训练选择也只取决于这里明示的状态—指令关系。

其中四个4卡任务使用 Staff1 统一数据集和完整 MEM 输入：Stage1、Stage2、Stage3
单模型，以及三阶段分阶段联合模型。另外两个2卡任务使用所有场景统一数据集和完整
MEM 输入，分别训练 Stage1 与 Stage2。入口按任务的 `gpu_ids` 数量自动计算梯度累积，
六个任务保持相同的全局 batch size。Staff1 统一数据集同时包含完整 Staff1 episode 与
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
