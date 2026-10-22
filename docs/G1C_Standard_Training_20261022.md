# 门控+C：按原主线方式进行普通训练

用户明确要求去掉新增的数据一致性检查，直接按原训练方式执行。本次使用独立入口 `tools/run_g1c_standard.py` 和配置 `configs/plan_g1c_standard_server.yaml`，旧故障目录和证据保留。

实验为原 G1 门控加 `L1 + 0.05 × 平滑梯度结构损失`。seed 0、1、2 各自从 RN50 与对应原始随机初始化开始，使用完整 300000 步学习率计划、真实 batch 16、FP32、原 AdamW 和全局 PFA 随机数路径。历史 G1 的相同三个 seed 作为已完成的对照，不再重训。

数据读取、Mixup 与优化器每步各执行一次，不再做双读、逐步输入字节比较、副本比较、数据/Mixup 重放或输入保护证明。损失非有限值检查、原梯度处理、冻结编码器检查、普通训练日志和完整断点恢复保持。运行元数据明确记录 `training_integrity_mode: standard_user_requested_v1` 和 `per_update_input_consistency_checks: false`，不把未执行的检查写成通过。

GPU 0 运行 seed 0；GPU 1 并行运行 seed 1 和 seed 2，每个任务有独立模型、优化器和随机状态，不使用 DDP。每 5000 步保存完整断点，保留最近两个；每 100 步更新进度。普通训练直接开始，不再运行旧的输入一致性验收或单独的 200 步试验。

```bash
.venv/bin/python tools/run_g1c_standard_pipeline.py --paths configs/plan_g1c_standard_server.yaml
```

运行目录为 `runs/g1c_standard`，报告目录为 `reports/g1c_standard`。三组训练完成后，队列自动评测 CBSD68、Kodak24、McMaster、Urban100 的原有 15 个噪声条件，汇总三个 seed 的均值、样本标准差、与历史 G1 的配对差值及最高 OOD 水平的 Table 1。完成状态由真实运行日志和 checkpoint 记录决定，本文不是训练已完成的证明。

此前独立测试发现的主机读写异常保留在[诊断记录](../results/g1c_host_integrity_20261022/README.md)。本次按用户指定关闭额外训练输入检查，结果不代表该主机问题已经修复。最终报告会注明这一运行条件。
