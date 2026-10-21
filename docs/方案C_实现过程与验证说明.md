# 方案 C：L1 加平滑梯度结构损失的实现与验证

## 实现内容

方案 C 的实验组为 **B3**：保留原始网络、原始 PFA 和像素 L1，只在训练损失中增加与干净真值的平滑梯度匹配项。未叠加方案 A 的空间相关 PFA，也未加入候选 B 的门控。

总损失为 `L_total = L1 + 0.05 * L_structure`。先对预测与真值逐 RGB 通道执行 3×3 Gaussian 平滑，再用归一化 Sobel 求水平、垂直梯度；每个方向的梯度误差使用 `sqrt(delta² + eps²) - eps`，两个方向取平均，eps=0.001。Gaussian 核为 `[[1,2,1],[2,4,2],[1,2,1]]/16`，Sobel-x 为 `[[-1,0,1],[-2,0,2],[-1,0,1]]/8`，Sobel-y 为其转置。每次卷积前使用 reflect padding。

## 实现过程

1. 在 `basicsr/models/losses/losses.py` 新增 `L1StructureLoss`，在 `__init__.py` 导出。滤波核用 `register_buffer` 保存，无新增可训练参数；模型前向、编码器和推理网络不需要修改。
2. 保持训练输出未经 clamp；结构项监督目标使用 `feed_train_data` 配对 Mixup 后的 `model.gt`。当 structure_weight=0 时跳过结构运算，并与原 L1 的数值、梯度保持一致。
3. 在训练日志中增加 `pixel_l1`、`structure`、`weighted_structure`，分别记录原始像素 L1、未加权结构项和加权结构项；`l_pix` / `loss` 表示总损失。不同目标的总 loss 不直接用于判断质量优劣。
4. 新增 `configs/plan_c.yaml`、`tools/run_structure_experiment.py` 和 `run_plan_c.cmd`，复用已有断点、随机状态、调度及数据工具。B0/B3 从同一 C0 严格加载权重，分别使用新优化器、相同采样和独立 PFA 随机流；PFA 均为 rho=0 的原独立扰动。
5. 新增 `tools/evaluate_structure.py`，复用方案 A 的 7 条件评测协议，以 B3 对比 B0。两种方案的训练、报告目录独立，不复用旧方案的训练轨迹。

默认配置：RTX 3070、FP32、真实 batch=4、128×128 patch、CBSD432 全部 432 张训练图、仅 Gaussian σ15 训练噪声、原 Mixup/BN/梯度裁剪、AdamW、weight decay=1e-4、学习率从 1e-5 余弦下降至 1e-6。

损失通过配置选择：

```yaml
pixel_opt:
  type: L1StructureLoss
  loss_weight: 1.0
  reduction: mean
  structure_weight: 0.05
  eps: 0.001
```

此实验只定义 mean reduction；未定义的元素加权和非法参数会直接报错。

## 验证结果

| 验证项 | 结果 |
| --- | --- |
| 方案 C 专项测试 | 19 项通过：CPU/GPU 完美匹配零损失、λ=0 的值与梯度等价、边缘错位/模糊检测、RGB 通道隔离、核归一化、双精度 autograd 检查、非法参数及真实 GPU 恢复 |
| 原有数据、调度、方案 A 加新方案 C | 合计 46 项 pytest 测试通过 |
| 完整 run_checks.cmd 回归 | 通过，另含 4 项原模型结构检查、CPU 恢复和缓存一致性检查 |
| 实际训练 | B0/B3 各完成 200 步，所有 loss 与分项均为有限值 |
| 训练公平性 | 200/200 步的输入、Mixup 后目标和学习率完全一致 |
| 分项日志 | 200/200 步满足 weighted_structure=0.05×structure、loss=pixel_l1+weighted_structure |
| 跨进程恢复 | B3 第 25 步保存并退出，重新启动恢复至第 200 步；另有逐状态 GPU 恢复测试 |
| 编码器/推理 | 编码器参数保持冻结；损失核不进入 net_g 权重，原推理网络可严格加载训练权重 |
| 评测入口 | C0/B0/B3 × 7 条件 × 3 个噪声 seed × 1 张开发图像，共 63 条记录；各方法对应噪声输入哈希一致 |
| 留出集 | 未评测，保留用于方法固定后的最终检验 |

| 200 步短测 | B0：L1 | B3：L1+结构项 |
| --- | ---: | ---: |
| 去掉前 10 步的平均每步耗时 | 0.076 秒 | 0.082 秒 |
| PyTorch allocated 显存峰值 | 384 MiB | 385 MiB |
| 首步 / 末步学习率 | 1e-5 / 1e-6 | 1e-5 / 1e-6 |

以 B3 最后一步为例：pixel_l1=0.01744221、structure=0.00408755、weighted_structure=0.00020438、loss=0.01764659。上述耗时仅为本机短测结果，显存不是系统总占用。

这些检查验证数学实现、可训练性与当前环境的短程稳定性，**不证明结构损失已提高 OOD 或纹理恢复质量**。5k 筛选与双种子 20k 复验尚未执行；若后续 SSIM 提高但 PSNR 降低，应报告质量权衡。当前没有对“平滑优于裸 Sobel”作独立有效性结论。

## 运行与结果位置

双击 `run_plan_c.cmd` 可运行/恢复两组 200 步 smoke。已经完成的运行会直接提示完成。5k 筛选：

```powershell
.\run_plan_c.cmd --group B0 --stage screen --seed 100 --resume
.\run_plan_c.cmd --group B3 --stage screen --seed 100 --resume
```

开发集统一评测：

```powershell
.venv\Scripts\python.exe tools/evaluate_structure.py --split dev --b0 runs/plan_c/screen/B0/seed100/checkpoints/update_005000.pth --b3 runs/plan_c/screen/B3/seed100/checkpoints/update_005000.pth
```

支持 confirm（20k）、confirm_short（10k）阶段；正式复验分别用 seed 101、102 从 C0 重新开始，不能接续筛选权重。`--stop-after N` 可保存退出而不缩短学习率计划；`--resume` 恢复匹配断点。源码或配置变化后不能混用旧断点，应使用新的输出目录。

开发集沿用 CBSD68 排序前 20 张，后 48 张留出；这种开发方式属于 OOD 辅助选型。默认噪声 seed=0/1/2，先在图像内平均重复噪声，再对图像做配对分析。方法固定后才使用 `--split holdout` 或 `--split kodak`。原始像素 L1、各条件 PSNR/SSIM、局部纹理和平坦区误差应共同检查。

文件位置：

- 配置：`configs/plan_c.yaml`。
- 短测权重：`runs/plan_c/smoke/{B0,B3}/seed100/checkpoints/update_000200.pth`。
- 逐步分项与输入哈希：对应运行目录的 `train.jsonl`；有效配置与资源统计：`effective_config.json`、`summary.json`。
- 验收记录：`reports/plan_c/verification.json`、`reports/plan_c_tests.log`、`reports/plan_c_regression.log`。
- 单图诊断 CSV 与可视化：`reports/plan_c/evaluation_smoke/`。
- 全部检查入口：`.\run_checks.cmd --no-pause`。
