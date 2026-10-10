# B2-Z1 标准开环评测协议

跨机器、跨 checkpoint 的定量表格只能读取
`standard_full_horizon_metrics.json`（schema v2）。该文件必须由
`run_standard_openloop_benchmark_container.sh` 启动容器，并由容器内的
`run_standard_openloop_benchmark.sh` 生成；旧的 `metrics.csv`、
`trajectory_metrics.csv` 和手工拼接命令都不是标准结果。

固定条件包括：数据集身份、episode、锚点步长、语言、完整 50 帧时域、
样本/GT 指纹，以及按 episode/frame 独立生成的显式 flow noise。这样结果不再
依赖 batch 划分或进程 RNG 状态。

宿主机入口已固定以下容器约束，禁止再手工拼接 `docker run`：

- `--shm-size=16g`（正式协议默认 4 个 DataLoader worker；Docker 默认 64 MB
  会在首批数据时报 shared-memory 错误）；
- 显式选择唯一 GPU；
- checkpoint、基座模型、MEM-ViT 和数据目录均通过 artifact registry/挂载解析；
- 前台运行时完整日志直接进入调用端；需要后台运行时必须同时传
  `--detach --container-name=...`，失败容器与日志会保留到人工确认后再清理。

宿主机示例：

```bash
LEROBOT_EVAL_DATA_ROOT=/absolute/host/data/root \
  ./run_standard_openloop_benchmark_container.sh \
    --gpu-id=0 --container-name=stage1-standard-eval -- \
    --protocol=/workspace/lerobot/evaluation_protocols/b2_z1_staff1_stage1_full_horizon_v1.json \
    --policy-path=/data/models/.../pretrained_model \
    --dataset-root=/data/datasets/... \
    --output-dir=/data/evaluations/... \
    --batch-size=4 --num-workers=4
```

`--gpu-id` 在宿主机入口只出现一次；入口将选中的物理 GPU 映射为容器内 GPU 0。
用户、HOME/cache、Python 环境、代码与数据挂载均由入口统一设置。

比较结果时使用：

```bash
python -m lerobot.scripts.compare_openloop_standard_metrics RESULT_A RESULT_B
```

比较器会拒绝 schema v1、缺少协议 ID、非确定性 flow noise、不同语言、不同
样本指纹或不同 GT 指纹的结果。
