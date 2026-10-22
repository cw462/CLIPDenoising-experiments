# CLIPDenoising：门控＋结构损失实验

> 当前运行安排：按用户要求，门控+C 已启动普通训练，关闭新增逐步输入一致性检查。seed 0、1、2 各训练完整 300k，初始分配为 GPU 0 跑 seed 0、GPU 1 并行跑 seed 1、2。**已配置自动迁移：等待 seed 0 满 300k、正常退出且 GPU 0 空闲后，从 seed 2 的完整常规检查点迁至 GPU 0 接续；seed 1 不暂停。**此处不表示迁移已经发生，实际状态、恢复步数和重算尾步数以迁移审计记录为准。三组完成后自动评测与汇总。见[启动记录](results/g1c_standard_launch_20261022/README.md)和[运行说明](docs/G1C_Standard_Training_20261022.md)。已知主机异常尚未确认修复，最终报告保留这一运行条件；以下停止状态与故障证据属于较早尝试。


**状态快照（2026-10-21 23:18）：** 当前方法为G1C（原G1门控＋L1＋0.05×平滑梯度结构损失），固定seed **0、1、2**，每种子完整300000步，分别对照历史G1相同seed。独立NumPy环境候选通过三seed短程GPU检查，但长程验证在第761步再次读取不一致；GPU1并发探针seed2在第30步也触发保护。两项均停止，候选退役，三seed正式300k尚未启动，没有后台正式训练或有效最终结果。详见[新故障与完整证据](results/g1c_numpy_isolation_failure_20261021/README.md)及[三种子协议](docs/G1C_Multiseed_20261021.md)。

此前单独C已按用户要求在1028步停止；旧环境G1C成功完成1634步，在第1635步因同一CPU输入连续读取不一致自动停止。异常根因未定位，旧失败完整保留。尚无本轮完整训练或最终评测结果。详见[旧运行快照与故障证据](results/c_full_20261021/STATUS.zh-CN.md)。

用户要求独立单卡训练并利用另一张卡运行额外种子，三种子训练、调度、评测和汇总代码已完成。当前真实一致性故障尚未解决，不能通过重试、放宽检查或减少并发假装修复；需要进一步定位读写路径或使用经验证稳定的计算环境。双卡DDP短测完整步耗时0.4472秒，单卡0.4178秒，当前实现双卡慢约7%，因此不采用DDP。原两组执行协议与下面A/C内容保留为工作记录。

**Table 1 对应条件：** [查看前轮六个条件的实际 PSNR/SSIM 与截图参考值](results/ac_20261021/table1/README.md)。B0/G1 为前轮三个训练种子的完整训练结果。当前分支新增[严格配对重跑与完整历史基准协议](docs/AC_Table1_Recovery_20261021.md)，A/C 覆盖相同的四数据集、15 个噪声条件，并单列六项最高噪声强度结果。新一轮的完成状态与有效结果见 [A/C Table 1 执行记录](results/ac_table1_20261021/README.md)。

**当前状态：** r2 通过实际 RN50/C0、batch16、三模型的短程恢复检查（连续4步对照2步＋恢复2步，完整状态逐位一致）及200步试跑；历史210张原图、3,150个噪声输入匹配。随后5k筛选在第3,855步被输入副本一致性检查中止，正式队列已停止，尚无本轮筛选评测、20k确认或A/C Table 1完成结果。独立长程恢复诊断又发现少数步骤的数据与原记录不同，短程检查不能证明整段训练轨迹可逐位复现。最新正式状态与证据索引见执行记录。

**前次 A/C 服务器实验（2026-10-21，失败证据保留）：** 46 项 GPU 验收通过，三组 200 步试跑、5k 筛选和 1,680 条筛选评测完成。六组 20k 续训已跑满，但最终逐步输入/目标哈希配对验收未通过，原队列因此停止；该批次没有有效的最终评测结论。详见[原运行状态与结果](results/ac_20261021/README.md)、[执行协议](docs/AC_Server_Run_20261021.md)及[30 万步完整训练的必要性分析](docs/AC_Full_Training_Assessment_20261021.md)。完整 A/C 300k 训练尚未启动。

实现说明：[A：空间相关 PFA](docs/方案A_实现过程与验证说明.md)、[C：平滑梯度结构损失](docs/方案C_实现过程与验证说明.md)。本分支服务器入口为 `tools/run_ac_table1_pipeline.py`（或 `bash launch_ac_pipeline.sh`），配置为 `configs/plan_a_server.yaml`、`configs/plan_c_server.yaml`。本轮从同一个前轮 B0 检查点续训，不能与从原始初始化训练 300k 混称。

**本机 Windows 已配置完成：** 使用方法见 [本地运行说明](LOCAL_SETUP.zh-CN.md)。双击 `run_demo.cmd` 运行已有基线模型的单图去噪；`run_checks.cmd` 验证环境、真实数据、模型和断点恢复。本机默认 CPU 配置，正式 CUDA 运行配置另见 `configs/local_paths.cuda.yaml`。

本仓库基于 [CLIPDenoising（CVPR 2024）](https://github.com/alwaysuu/CLIPDenoising)，保存末级 RGB 旁路校准的代码、验收工具、训练与评测入口，以及 2026 年 10 月完成的实验记录。上游基准 commit 为 `2647bf815df0299a03d16df34b896fa266ceefda`；下方保留上游 README 与论文引用。

## 已完成的实验

主线于北京时间 **2026-10-21 07:30:01** 正常完成。B0/G1 分别训练 seed 0、1、2，G2/G3 分别训练 seed 0，共 **8 组 × 300,000 次 optimizer update**。CBSD432 的 432 张图用于正式训练；CBSD68、Kodak24、McMaster、Urban100 共 210 张测试图，每图评测 15 个条件，得到 **25,200 条逐图结果和 48 张固定样本对比图**。最终测试在八组正式训练全部完成后开始。

| 实验 | 结构 | 正式训练种子 |
|---|---|---|
| B0 | 原冻结 CLIP RN50、原解码器和 PFA 基线 | 0、1、2 |
| G1 | 末级 RGB 旁路门控残差校准，包含解码特征指导 | 0、1、2 |
| G2 | 去掉门控的残差校准 | 0 |
| G3 | 去掉解码特征指导，保留门控校准 | 0 |

| OOD 范围 | B0 PSNR / dB | G1 PSNR / dB | G1 − B0 / dB |
|---|---:|---:|---:|
| 官方三类噪声 | 28.679 | 28.792 | **+0.113** |
| 扩展五类噪声 | 28.388 | 28.468 | **+0.079** |

表中为三个训练种子的均值。官方三类为 Gaussian、Spatial Gaussian、Poisson；扩展条件另含 Speckle 与 Salt-and-Pepper。OOD 聚合排除训练分布 Gaussian σ15，并对图像、族内强度、噪声族和数据集逐层等权汇总。

G1 的总体改善较小，不能表述为全面或统计显著提升：Spatial Gaussian 和 Salt-and-Pepper 的平均 PSNR 分别下降约 0.0115、0.1998 dB。**seed 0 的 G3 得分高于完整 G1**，官方三类与扩展五类分别高 0.0200、0.0743 dB；现有结果不能证明解码特征指导有收益，单种子消融也不能证明 G3 稳定更优。Urban100 使用原作者 SelfExSR 的 SRF2 HR 版本，与 CLIPDenoising 作者具体测试副本的逐像素等同性尚未确认。

- [完整实验报告](final_results_20261021/reports/experiment_report.md)
- [详细结果、配对差值与消融边界](final_results_20261021/reports/results_interpretation_zh.md)
- [48 张对比图索引](final_results_20261021/reports/figures/index.md)
- [逐图结果 CSV](final_results_20261021/reports/per_image_results.csv)、[OOD 汇总](final_results_20261021/reports/ood_aggregate_results.csv)、[全部负差值案例](final_results_20261021/reports/failure_cases.csv)

仓库提供源码、报告、指标表、对比图和运行元数据。完整训练 checkpoint 与原始诊断数组未随仓库上传。以下命令用于在自备数据下载和 GPU 环境中**重新训练**；仅克隆本仓库无法直接加载本轮已训练权重进行最终评测。

## 环境与配置

本轮正式执行环境为 Linux、Python 3.10.18、PyTorch 2.2.1 / CUDA 12.1、torchvision 0.17.1、NumPy 1.26.4、SciPy 1.15.3，以及两张 RTX 3090。实际采用已有 CUDA Python 环境创建 `--system-site-packages` 隔离 venv，再在 venv 内补齐依赖，没有重新安装基础 CUDA 环境。

[远程依赖快照](environment/requirements.remote.snapshot.txt)用于审计，含原环境的本地构建地址和可编辑安装路径，**不是可移植的 `pip install -r` 文件**。根目录 `requirements.local.lock.txt` 是 Windows CPU 控制端快照，也不能代替正式 GPU 环境。本仓库未声称已验证从空白操作系统一键安装。

已有相应 CUDA Python 及快照中基础依赖时，在仓库根目录执行以下实际使用的环境复用方式。`CUDA_PYTHON` 应替换为该环境的 Python 路径；后续命令均使用项目 venv：

```bash
CUDA_PYTHON=/path/to/existing/cuda-python/bin/python
"$CUDA_PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install lmdb==1.5.1 scikit-image==0.24.0
.venv/bin/python -c "import torch, torchvision, numpy, scipy, cv2, PIL, yaml, psutil, lmdb, skimage; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

配置文件 [configs/local_paths.yaml](configs/local_paths.yaml) 已采用仓库相对路径，可直接编辑；也可从 [示例配置](configs/local_paths.example.yaml) 复制：

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

检查 `project_root`、数据/权重路径、`output_root`、`reports_dir` 和 `gpu_indices`。新一轮复现应使用独立输出目录，勿将输出指向归档交付目录 `final_results_20261021/`。当前主队列要求两个不同的可见 CUDA 设备，默认 `[0, 1]`；控制进程需同时看到两张卡，子进程由队列各自分配 GPU。

本轮设置 `jobs_per_gpu: 4`、`cpu_threads: 4`、`cache_decoded_images: true`，每卡并行四个正式实验；smoke/pilot 每卡一个。每组保持独立模型、优化器、随机状态、batch 16 和 FP32，未使用 DDP。若资源较少，可在新队列开始前降低 `jobs_per_gpu`；已经创建的队列不接受中途改变 GPU 分配或并发配置。

## 数据、验收与完整主线

数据准备工具按固定上游 commit 下载、核验文件与 RN50 SHA256，生成 400/32 开发划分。该工具**不接受 `--paths` 参数**，目标固定为当前仓库根目录的 `data/`、`pretrained/`、`splits/`；配置应与这些目录对应。Urban100 单独准备：

```bash
.venv/bin/python tools/prepare_data.py
.venv/bin/python tools/prepare_data.py --urban100-only
```

可先执行实际数据与 CPU 恢复验收。下面的诊断更新不计入正式训练：

```bash
.venv/bin/python -m pytest tests/test_data_contract.py
.venv/bin/python tests/test_calibration_contract.py --weights pretrained/RN50.pt --device cpu --report reports/calibration_contract_cpu.json
.venv/bin/python tests/test_resume_contract.py --paths configs/local_paths.yaml --device cpu --report reports/resume_contract_cpu.json
.venv/bin/python tests/test_cache_contract.py --paths configs/local_paths.yaml --report reports/cache_contract.json
```

随后启动完整主线：

```bash
.venv/bin/python tools/run_pipeline.py --mode main --paths configs/local_paths.yaml
```

主队列依次完成资源预检、调度/缓存/结构/恢复验收、B0/G1 的 100 步 smoke 和 5,000 步 pilot、八组 300,000 步正式训练、最终评测及汇总；无需逐阶段确认。进度写入配置的 `reports_dir/progress.md` 与 `output_root/pipeline_state.json`。中断后，在源码和配置未变时重跑同一主队列命令，可接续匹配的运行或断点。

只在主队列没有监测该组、该组进程已结束时，才手工恢复单组：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_experiment.py --exp B0 --stage full --seed 0 --paths configs/local_paths.yaml --resume
```

最终评测与汇总已由主线自动调用。需要手工重算时，先确保八组完整 checkpoint、effective config、数据与源码对应一致，再运行：

```bash
.venv/bin/python tools/evaluate_experiment.py --paths configs/local_paths.yaml --visuals
.venv/bin/python tools/summarize_experiments.py --paths configs/local_paths.yaml
```

这些命令读取配置的训练输出，生成本轮的新报告；`final_results_20261021/` 是已完成实验的归档。源码、配置、checkpoint、数据与噪声标识在运行和评测时检查；报告中的 CUDA 恢复验收包含 FP32 容差，不代表整段 CUDA 训练轨迹逐位确定。

---

# Transfer CLIP for Generalizable Image Denoising (CVPR2024)

> **Abstract:** *Image denoising is a fundamental task in computer vision. While prevailing deep learning-based supervised and self-supervised methods have excelled in eliminating in-distribution noise, their susceptibility to out-of-distribution (OOD) noise remains a significant challenge. The recent emergence of contrastive language-image pre-training (CLIP) model has showcased exceptional capabilities in open-world image recognition and segmentation. Yet, the potential for leveraging CLIP to enhance the robustness of low-level tasks remains largely unexplored. This paper uncovers that certain dense features extracted from the frozen ResNet image encoder of CLIP exhibit distortion-invariant and content-related properties, which are highly desirable for generalizable denoising. Leveraging these properties, we devise an asymmetrical encoder-decoder denoising network, which incorporates dense features including the noisy image and its multi-scale features from the frozen ResNet encoder of CLIP into a learnable image decoder to achieve generalizable denoising. The progressive feature augmentation strategy is further proposed to mitigate feature overfitting and improve the robustness of the learnable decoder. Extensive experiments and comparisons conducted across diverse OOD noises, including synthetic noise, real-world sRGB noise, and low-dose CT image noise, demonstrate the superior generalization ability of our method.* 
<hr />


## Data preparation

- For synthetic noise removal, we used clean images from CBSD432 and added i.i.d. Gaussian noise with sigma=15
- For sRGB noise removal, we utilized the ISP/inverse ISP pipelines from CBDNet and clean images from DIV2K, and synthesized noisy images based on Poisson-Gaussian noise with the fixed noise level. We use the following code:
```
python basicsr/data/ISP_implement.py
```
- For LDCT image noise removal, we used 1mm-thickness normal-dose CT images from nine patients (except for patient L506, which was used for test) of [Mayo2016-AAPM dataset](https://aapm.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h) and added i.i.d. Gaussian noise with sigma=5.

## Training and evaluation

Training and test commands for synthetic noise removal:
```
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_SyntheticDenoising_GaussianSigma15.yml
python Denoising/test_synthetic_denoising.py
```

Training and test commands for sRGB noise removal:
```
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_sRGBDenoising_FixedPoissonGaussian.yml
python Denoising/test_real_denoising_sRGB.py
```

Training and test  commands for LDCT image noise removal:
```
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_LDCTDenoising_GaussianSigma5.yml
python Denoising/test_real_denoising_CT.py
```

### Pre-trained models
Pre-trained models can be found in https://drive.google.com/drive/folders/1F8md2zen0iPlhGnI5-VdjdBOLY8DBIiF?usp=drive_link

## Citation
Please consider citing this paper if it helps you:

    @inproceedings{Jun2024Transfer,
        title={Transfer CLIP for Generalizable Image Denoising}, 
        author={Jun Cheng and Dong Liang and Shan Tan},
        booktitle={CVPR},
        year={2024}
    }



**Acknowledgment:** This code is based on the [Restormer](https://github.com/swz30/Restormer) 
