# A/C 服务器续训协议（2026-10-21）

以 `luhanting` 分支 `49696d96dc614d0aa98b41e2d5ceb915113db211` 为实现基础。此次只改变 PFA（A）或训练损失（C），两项均独立复验，不组合，不在查看本轮结果后改变超参数。

共同 C0 为上一轮 B0 seed0 的 300000 步完整 checkpoint，SHA256：`123b9091a9c13b17da0e40f25f706efc8e5044ce5d8295bf5701eb1dd8f99675`。与原计划提及的未提供官方 synthetic checkpoint 不混同。B0/B1/B3 每次均从该权重重新加载全部参数和 BN 状态，重新创建 AdamW 和余弦学习率计划；不恢复 C0 的旧优化器。

| 阶段 | 方法 | 训练种子 | 每组更新 | 用途 |
|---|---|---|---:|---|
| smoke | B0、B1(A)、B3(C) | 100 | 200 | GPU 稳定性和配对输入验证，不计正式结果 |
| screen | B0、B1(A)、B3(C) | 100 | 5000 | 20 张开发图筛选，参数不再调整 |
| confirm | B0、B1(A)、B3(C) | 101、102 | 20000 | 两项各自复验，共用匹配的 B0 |

正式 screen+confirm 共 135000 次 optimizer update，另有 600 步 smoke。所有组采用 CBSD432 全部 432 张训练图、128×128 patch、真实 batch16、FP32、原配对 Mixup、Gaussian sigma15 图像训练噪声。冻结编码器参数并沿用原 BN 更新行为；AdamW lr1e-5→1e-6，weight_decay1e-4，保留原梯度裁剪。使用已验证的只读整图缓存和独立 PFA RNG，并逐步比较同训练种子的输入及目标哈希。

A 固定 F3/F4 的 rho=.5、3×3 circular 随机场、单位边际方差；C 固定 lambda=.05、eps=.001、Gaussian 平滑再 Sobel 的结构项。B0 保持独立 PFA 和像素 L1。

评测固定 7 条件（Gaussian15/25/50、空间相关 Gaussian45/55、Poisson2.5/3.5），每个条件 3 个噪声种子。筛选仅用固定 CBSD68 前20张；最终用其余48张及 Kodak24。上一轮研究已经评测过完整 CBSD68/Kodak24，本轮保留集仅表示不参加本轮筛选，不能宣称是整个研究从未见过的数据。

所有 confirm 训练完成后才访问本轮保留集。C0 只评测一次；6个续训模型加C0共7组，最终应有10584条逐图结果。噪声重复先在图像内平均，再按图像做配对bootstrap；两个续训种子的波动单独报告，不能当作独立从头训练重复。ID单列，6个OOD条件等权；分别报告两个数据集及明示的等权汇总。

初步有效判据为相对同预算 B0 的整体 OOD 平均 PSNR 提升至少约0.10 dB、两个复验种子方向一致、ID下降不超过约0.05 dB；同时记录单条件退化和所有负结果。现阶段不预设 A/C 有效，也不将短续训结果表述为 SOTA。

服务器入口：

```bash
.venv/bin/python tools/run_ac_pipeline.py --plan-a configs/plan_a_server.yaml --plan-c configs/plan_c_server.yaml
```

调度器只向没有其他计算进程的 GPU 分配任务，最多每卡4个训练进程；环境变量为每个子进程选择独立 GPU。配置中的 `gpu_index: 0` 表示子进程可见卡编号。当前已有其他作业的 GPU 会等待空闲，不停止任何其他进程。

检查点每1000步及阶段结束原子保存，完整校验后才替换 latest，保留最新1个可恢复检查点。完成信息只有在参数冻结检查、summary、manifest 全部落盘后写入。原项目、原始权重、数据及结果保持独立，新队列输出在 `runs/ac`，评测与报告在 `reports/ac`。
