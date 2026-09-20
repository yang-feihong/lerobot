# MEM 未推送提交测试审查与修复（2026-09-20）

## 当前状态：问题已修复，关联测试通过

用户确认修复后，已修改 `src/lerobot/policies/pi05/modeling_pi05.py` 的两处检查点调用：通用 `_apply_checkpoint`（含视觉编码）和联合 Transformer 层重算，均显式使用 `preserve_rng_state=True`，确保 LoRA dropout 在重算时复用前向的随机掩码。最终归一化不含随机运算，保持原样。

本轮按 TDD 顺序执行，没有使用上一轮的进程内 monkeypatch 代替生产代码修复：

1. **Red**：先扩展回归测试，覆盖视觉、语言、两者同时注入 LoRA，分别设置 dropout=0/0.2。将 LoRA B 初始化为非零值，模拟已训练的 adapter。修复前结果 **3 failed、3 passed**；三个非零 dropout 用例均复现梯度不一致，见 [red](logs/fix_20260920_red.log)。
2. **Green**：修改上述两处生产代码后，完整关联测试 **128 passed、2 skipped**，见 [green](logs/fix_20260920_green.log)。梯度对照同时验证 loss、所有实际产生的梯度和 backward 后的 CPU RNG 状态；保存/恢复后下一步更新的 12 种组合继续通过。
3. **回归与检查**：双进程 CPU DDP 覆盖 MEM LoRA/frozen × dropout=0/0.2，每种配置执行 2 次优化器更新，每次累积 8 个 microbatch，各 rank 参数严格一致，见 [DDP](logs/fix_20260920_ddp.log) 和 [复现脚本](logs/fix_20260920_ddp.py)。Ruff lint、新增测试文件格式、`git diff --check` 均通过，见 [lint](logs/fix_20260920_lint_final.log)。生产文件完整格式检查另检出两处原有的无关换行差异，未扩大修复范围。

验证环境仍为现有部署容器的 PyTorch 2.6.0+cu124，低于项目声明的 >=2.7；本轮未升级环境，也未执行真实大模型 GPU/BF16/FSDP 测试。未创建 commit 或 push。

## 修复前审查记录（历史结果，不代表当前代码状态）

审查对象：`origin/MEM..HEAD`，仅提交 `ea9fc1a135d19900118e4bf0f20d1426ac7336bb`（LoRA for MEM）。已执行 `git fetch origin MEM`，远端基线为 `5a7c7b0e`。开始时工作区干净。审查覆盖提交中的 8 个文件；没有修改生产代码、创建提交或推送。

结论：默认零 dropout 的 MEM LoRA 路径及主要保存/恢复行为通过本次验证，但不能认定所有支持的配置都正确。发现 1 个可复现的非默认配置问题。下面的结果基于实际执行，不以仓库已有历史日志替代验证。

## P2：MEM 视觉 LoRA 的非零 dropout 与梯度检查点不兼容

- 触发条件：`mem_vit_enabled=True`、`mem_vit_finetune_mode="lora"`、`gradient_checkpointing=True`，通过自定义 PEFT 配置/API 设置 `lora_dropout=0.2`。启动脚本没有暴露 dropout 参数，默认值为 0，因此默认启动配置不受此复现影响。
- 影响：相同权重、输入和随机种子下，打开梯度检查点会改变视觉 LoRA 的反向梯度；训练不会因此必然报错，但优化器使用的梯度不对应原始前向使用的 dropout 掩码。
- 原因：[modeling_pi05.py:1070](src/lerobot/policies/pi05/modeling_pi05.py#L1070) 的 `_apply_checkpoint` 使用 `preserve_rng_state=False`；[图像编码调用](src/lerobot/policies/pi05/modeling_pi05.py#L1200) 经过该函数。本次提交允许 MEM vision tower 注入 LoRA，将随机 dropout 引入此路径。检查点辅助函数中的问题此前已存在，本次新增的 MEM LoRA 模式也受到影响。
- 回归测试：[test_vision_lora_checkpointing_preserves_gradients](tests/policies/pi0_pi05/test_pi05_mem_peft_review.py)。只对视觉 `q_proj/v_proj` 注入 LoRA，排除语言 LoRA 干扰。dropout=0 对照通过，dropout=0.2 失败，首个不一致参数为视觉第 0 层 `v_proj.lora_B.default.weight`。
- Red：未修改的实现上，回归文件结果为 **13 passed, 1 failed**。
- Green：仅在独立测试进程中将 `_apply_checkpoint` 改为 `preserve_rng_state=True`，同一梯度对照测试 **2 passed**。没有把该修正写入生产源码。
- 建议：修正包含随机运算的 checkpoint 的 RNG 保存策略，保留回归测试；若同时支持语言 LoRA dropout，还应一并检查 joint transformer 的 checkpoint 调用。

复现（在已有容器的 `/workspace/lerobot` 中，用 `/lerobot/.venv` 环境运行）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=/tmp/review-20260920-uv UV_PROJECT_ENVIRONMENT=/lerobot/.venv \
PYTHONPATH=/workspace/lerobot/src:/workspace/lerobot \
uv run --no-sync pytest -q -p no:cacheprovider \
  tests/policies/pi0_pi05/test_pi05_mem_peft_review.py
```

## 已执行验证

| 验证范围 | 结果 | 证据 |
| --- | --- | --- |
| 提交自带的 MEM PEFT 测试 + train_utils 测试 | 50 passed | [baseline](logs/review_20260920_baseline.log) |
| 新增 checkpoint/优化器续训 + 梯度一致性测试 | 13 passed，1 failed | [regressions](logs/review_20260920_regressions_offline.log) |
| 启动参数矩阵及关联 PI05 测试 | 60 passed，2 skipped | [related](logs/review_20260920_related2.log) |
| 进程内候选修正，复跑梯度一致性测试 | 2 passed | [green](logs/review_20260920_green.log) |
| 双进程 CPU DDP：LoRA/frozen，各 2 次更新，每次累积 8 个 microbatch | 全部通过，各 rank 更新后参数逐元素一致 | [DDP](logs/review_20260920_ddp.log) |

去重后的未修正实现测试结果为 **123 passed、1 failed、2 skipped**，不把候选修正后的通过数重复计入。新增两个测试文件合计 44 个用例；最终版本复测记录见 [final_new_tests](logs/review_20260920_final_new_tests.log)。

新增保存/恢复测试使用真实 PEFT 保存与 `save_checkpoint`、`load_training_state`，覆盖 `full/lora/frozen × 提供/不提供完整 state_dict × active-only 开/关` 的 12 种组合。恢复 AdamW 状态后再更新一次，loss 和全部模型张量与不中断训练严格相同（`rtol=atol=0`）。提供完整 state_dict 的测试只验证序列化路径，不等同于真实 FSDP 集合通信测试。

新增脚本测试覆盖 LoRA/expert/full、MEM 模式、冻结优先级、非法参数和 resume 不覆盖已保存策略。全部使用 `--dry-run=true` 与临时输出目录，不启动训练。

`bash -n train_vla_pi05.sh`、`git diff --check` 已通过。新增测试文件的 Ruff 检查见 [lint](logs/review_20260920_lint_final.log)。

## 验证边界

- 环境：Python 3.12.3、PyTorch 2.6.0+cu124、PEFT 0.19.1、Transformers 5.5.4，见 [versions](logs/review_20260920_versions.log)。这是当前部署容器；PyTorch 版本低于 `pyproject.toml` 声明的 `>=2.7`，没有为此次审查升级或修改运行环境。
- 模型测试使用真实 PI05 forward 的微型 CPU 配置，不代表真实模型的 GPU/BF16、显存峰值、长程训练质量或真实 FSDP 已验证。两个 GPU/HF 条件测试跳过。
- 额外尝试的 tokenizer 对齐测试 `test_pi05_processor_inputs_match_openpi_reference` 因离线环境缺少对应缓存而无法运行，记录在 [首次关联测试日志](logs/review_20260920_related.log)，不作为代码缺陷计入。该轮脚本测试最初遇到默认输出目录无权限，改用临时目录后全部通过。
- 工作区保留两个新增测试文件和本报告；已提交的生产代码保持原样。非零 dropout 回归测试有意保持失败，便于后续修正时完成正式的 Red → Green。
