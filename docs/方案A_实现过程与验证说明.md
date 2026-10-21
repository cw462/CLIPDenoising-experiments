# 方案 A：空间相关 PFA 实现与验证说明

## 目标与实现过程

本次只实现文档中的方案 A，实验组称为 **B1**，不叠加 RGB 校准模块 G1，也不实现候选 B、C。

1. **构造相关随机场。** 新增 `basicsr/models/archs/spatial_pfa.py`。对两个独立标准高斯场计算 `Z = sqrt(1-rho)*epsilon_a + sqrt(rho)*R(epsilon_b)`；`R = 3*AvgPool3x3(circular_pad(epsilon_b))`。默认 rho=0.5，每个位置的理论方差仍为 1。circular padding 仅作用于随机场，不改变图像补齐。
2. **接入原 PFA。** `CLIPDenoising_arch.py` 增加可选的 `pfa_rho`、`pfa_seed`。只有 F³、F⁴，即 idx=3、4，使用相关场；F¹、F² 及 `[0.025, 0.05, 0.075, 0.1]` 的扰动系数保留。默认 rho=0、seed=None 时保留原随机数调用行为。无新增参数或权重键；eval 不执行 PFA，推理结构不变。
3. **隔离随机流并支持恢复。** B0/B1 使用相同的独立 PFA seed，各层及两个随机分量分别使用 generator。新增抽样不会改变数据噪声、增强和 Mixup 的序列。`tools/run_experiment.py` 的 checkpoint 同时保存、恢复 PFA RNG；原默认基线的旧断点仍兼容。
4. **实现公平续训入口。** `tools/run_pfa_experiment.py` 配合 `configs/plan_a.yaml`，严格加载同一个 C0，为两组分别创建新的 AdamW。统一 batch=4、128×128、Gaussian σ15、FP32、L1、原 Mixup/BN/梯度裁剪；编码器参数继续冻结。学习率按实际更新次数从 1e-5 余弦下降至 1e-6，weight decay=1e-4。筛选与复验分别从 C0 开始。
5. **固定评测协议。** `tools/evaluate_pfa.py` 支持 7 个条件：Gaussian 15/25/50、Spatial Gaussian 45/55、Poisson 2.5/3.5，默认噪声 seed=0/1/2。统一 RGB float、输出 clamp、data_range=1、crop_border=0，固定 Gaussian-window RGB SSIM。CBSD68 排序前 20 张用于开发，后 48 张留出，清单位于 `splits/plan_a_{dev,holdout}.txt`。先在每张图内平均噪声重复，再计算图像配对差异及 bootstrap 区间。

C0 为 `pretrained/synthetic_baseline.pth`，SHA256：`09c1809b5dbb41222f423d99b545573f0c2061c59363d4e516414b63d196f7dd`。两组输出分别保存在 `runs/plan_a/<stage>/{B0,B1}/seed<seed>/`，包含有效配置、逐步 loss/耗时/输入哈希及完整 checkpoint。

## 已完成的验证

设备为 RTX 3070 8GB，Python 3.9.20、PyTorch 2.2.2+cu121，CUDA 可用。

| 检查 | 结果 |
| --- | --- |
| 方案 A 专用测试 | 15 项通过，含 CPU/GPU 统计、边界、独立 RNG、默认兼容、评测协议、GPU 反向和完整恢复 |
| 现有数据/调度测试 | 12 项通过，与新增测试合计 27 项通过 |
| 完整回归入口 `run_checks.cmd` | 通过，另含 4 项原模型结构检查、CPU 恢复及缓存一致性检查 |
| 推理等价性 | 相同权重、BN 状态下，GPU eval 输出逐元素完全相同，无新增参数 |
| 稳定性 | B0/B1 各完成 200 步，batch=4、FP32，全部 loss 有限 |
| 跨进程恢复 | B1 第 25 步保存并退出，重新启动后恢复至 200 步 |
| 训练公平性 | 两组 200/200 步的输入和目标哈希完全一致，包括 B1 恢复后的步骤 |
| 评测入口 | C0/B0/B1 × 7 条件 × 3 seed × 1 张开发图像，共 63 条结果；对应噪声输入哈希一致 |
| 留出集 | 未评测，保留给固定方法后的最终检验 |

GPU 随机场实测：rho=0 时标准差 1.0006、相邻相关系数 -0.0024；rho=0.5 时标准差 0.9995、相邻相关系数 0.3305，符合理论 1/3。

| 200 步短测 | B0 | B1 |
| --- | ---: | ---: |
| 去掉前 10 步后的平均每步耗时 | 0.039 秒 | 0.071 秒 |
| PyTorch allocated 显存峰值 | 384 MiB | 385 MiB |
| 首步 / 末步学习率 | 1e-5 / 1e-6 | 1e-5 / 1e-6 |

耗时受同时运行的回归检查及断点恢复影响，仅作本机运行估计；显存数字不是整个系统的 GPU 占用。验收记录见 `reports/plan_a/verification.json`、`reports/plan_a/pfa_statistics.json`、`reports/plan_a_tests.log`、`reports/plan_a_regression.log`。单图评测 CSV 与可视化见 `reports/plan_a/evaluation_smoke/`。

这些结果验证了实现与短程运行，**不证明 OOD 收益**。尚未执行 5k 筛选或双种子 20k 复验；单图短测不作为方法有效性结论。

## 运行与后续实验

双击 `run_plan_a.cmd` 可运行/恢复两组 200 步 GPU smoke；已完成的运行会提示完成。5k 筛选命令：

```powershell
.\run_plan_a.cmd --group B0 --stage screen --seed 100 --resume
.\run_plan_a.cmd --group B1 --stage screen --seed 100 --resume
```

筛选完成后，以相同口径重测 C0/B0/B1 的 20 张开发图像：

```powershell
.venv\Scripts\python.exe tools/evaluate_pfa.py --split dev --b0 runs/plan_a/screen/B0/seed100/checkpoints/update_005000.pth --b1 runs/plan_a/screen/B1/seed100/checkpoints/update_005000.pth
```

支持 `confirm`（20k）与 `confirm_short`（10k）阶段。开发筛选后固定方法，B0/B1 分别用 seed 101、102 从 C0 重新开始 `confirm`，再评测 `--split holdout` 和可选的 `--split kodak`。bootstrap 区间仅反映图像差异，不替代训练种子波动。

当前短测权重：`runs/plan_a/smoke/{B0,B1}/seed100/checkpoints/update_000200.pth`。`--stop-after N` 可在指定更新处保存退出，不缩短原阶段学习率计划；用 `--resume` 继续。修改源码、配置或 C0 后，已有断点会拒绝混用，应使用新的输出目录。

重跑全部检查：`.\run_checks.cmd --no-pause`。本次仅执行必要的稳定性与回归测试，未自动启动长训。
