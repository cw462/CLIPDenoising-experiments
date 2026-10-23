# 门控+C：按原主线方式进行普通训练

用户明确要求去掉新增的数据一致性检查，直接按原训练方式执行。本次使用独立入口 `tools/run_g1c_standard.py` 和配置 `configs/plan_g1c_standard_server.yaml`，旧故障目录和证据保留。

实验为原 G1 门控加 `L1 + 0.05 × 平滑梯度结构损失`。seed 0、1、2 各自从 RN50 与对应原始随机初始化开始，使用完整 300000 步学习率计划、真实 batch 16、FP32、原 AdamW 和全局 PFA 随机数路径。历史 G1 的相同三个 seed 作为已完成的对照，不再重训。

数据读取、Mixup 与优化器每步各执行一次，不再做双读、逐步输入字节比较、副本比较、数据/Mixup 重放或输入保护证明。损失非有限值检查、原梯度处理、冻结编码器检查、普通训练日志和完整断点恢复保持。运行元数据明确记录 `training_integrity_mode: standard_user_requested_v1` 和 `per_update_input_consistency_checks: false`，不把未执行的检查写成通过。

初始分配为 GPU 0 运行 seed 0，GPU 1 并行运行 seed 1 和 seed 2。每个任务有独立模型、优化器和随机状态，不使用 DDP。每 5000 步保存完整断点，保留最近两个；每 100 步更新进度。普通训练直接开始，不再运行旧的输入一致性验收或单独的 200 步试验。

已配置独立自动迁移任务，等待 seed 0 完成 300000 步、训练进程正常退出，并确认 GPU 0 空闲后，再将 seed 2 从 GPU 1 迁至 GPU 0。迁移只暂停 seed 2，seed 1 保持运行。迁移使用 seed 2 的完整常规检查点，经文件身份与完整性校验后恢复模型、优化器、学习率调度、随机数及采样器状态；科学源码和训练配置保持固定，继续同一 seed 的 300000 步计划。

自动等待任务于北京时间 2026-10-22 10:36 在服务器启动，见[启动快照](../reproducibility/g1c_standard/migration_armed_snapshot.json)。当日 15:12:57，seed 0 完成 300000 步。15:19:32 首次自动迁移因旧协调器暂停超时失败：该协调器是 tmux 窗格的直接子进程，tmux 收到其停止状态后会发送 `SIGCONT`，因此不能用 `SIGSTOP` 持续暂停它；这一行为可在官方 [`server_child_stopped` 实现](https://github.com/tmux/tmux/blob/master/server.c)中核对。首次失败发生在操作 seed 2 之前，seed 1、2 继续在 GPU 1 训练，GPU 0 保持空闲。

修复后的单次重试于 17:08:23 部署，等待 seed 2 保存 240000 步常规检查点后开始交接。修复核对进程身份后仅终止旧协调器，再由独立控制器接管调度；迁移期间 seed 1 未暂停。控制器通过 40 项模拟测试，17:10 的 Linux 实际自有进程测试也验证了 `STOP`、`CONT`、`TERM` 行为，测试未操作训练进程。

17:18:04 已确认实际迁移成功：seed 2 从 240000 步完整检查点在 GPU 0 恢复，并记录到 240200 步；检查点之后原已执行的 6 步被重算。随后检查到 240500 步，`nvidia-smi` 确认 seed 2 新进程位于 GPU 0、seed 1 原进程位于 GPU 1。模型、优化器、学习率、随机数和采样器状态均通过原训练入口恢复，训练配置与科学源码保持固定。见[恢复快照](../reproducibility/g1c_standard/migration_recovered_snapshot.json)和[本次验证记录](../reproducibility/g1c_standard/migration_retry_validation.json)。这证明迁移完成，三组最终训练与评测仍在继续。

首次失败的审计、计划和状态原样归档于 `reproducibility/g1c_standard/migration_controller_attempts/`，当前[迁移审计](../reproducibility/g1c_standard/migration_audit.json)记录本次重试及归档引用；另见[迁移计划](../reproducibility/g1c_standard/migration_plan.json)和[修复后的控制器](../reproducibility/g1c_standard/gpu_migration_controller_retry.py)。后续报告保留首次失败和实际重试结果。

检查点之后尚未保存的尾步会从该检查点重算，不增加最终有效训练步数。迁移计划、状态和审计保存在 `reproducibility/g1c_standard/migration*`，记录实际源/目标 GPU、检查点步数与 SHA256、恢复确认及重算尾步数；无法准确确认的尾步数会标为未知。实际跨卡恢复是否完成以审计记录为准，也不承诺跨卡恢复后的整段数值轨迹逐位相同。最终结果导出会依据实际审计记录说明迁移情况。

```bash
.venv/bin/python tools/run_g1c_standard_pipeline.py --paths configs/plan_g1c_standard_server.yaml
```

运行目录为 `runs/g1c_standard`，报告目录为 `reports/g1c_standard`。三组训练完成后，队列自动评测 CBSD68、Kodak24、McMaster、Urban100 的原有 15 个噪声条件，汇总三个 seed 的均值、样本标准差、与历史 G1 的配对差值及最高 OOD 水平的 Table 1。完成状态由真实运行日志和 checkpoint 记录决定，本文不是训练已完成的证明。

此前独立测试发现的主机读写异常保留在[诊断记录](../results/g1c_host_integrity_20261022/README.md)。本次按用户指定关闭额外训练输入检查，结果不代表该主机问题已经修复。最终报告会注明这一运行条件。


## Completed evaluation (2026-10-22)

All three seeds completed 300,000 training updates and 3,150 evaluation items each. [Final results and execution disclosures](../results/g1c_standard_300k_20261022/README.md) include the report, per-image measurements, Table 1 and OOD summaries. At the user request, recovery checks and seed0/seed2 noisy-input consistency blocking were skipped; seed0 also skipped training-asset and historical-checkpoint identity checks. Historical noise identity is not established for the combined results. Seed1 results were reused and original failures retained.
