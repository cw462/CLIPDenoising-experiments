# Windows 本地运行说明

更新：RTX 3070 显卡坞已连接，CUDA 与方案 A 的 GPU 训练测试已通过。方案 A 请使用 `run_plan_a.cmd`，详见 [实现与验证说明](docs/方案A_实现过程与验证说明.md)。下文保留初次配置时的 CPU 运行说明；GPU 不可用的记录指当时尚未连接的状态。`run_checks.cmd` 现已增加 15 项方案 A 测试。

项目目录：`E:\E_MyProjects\school\CLIPDenoisingV1`。
仓库：<https://github.com/cw462/CLIPDenoising-experiments>，拉取版本 `41a97fd63a7a722f288e831e5f4ba945e4167dad`。

## 直接运行

双击 **`run_demo.cmd`**，即可运行一次真实图像去噪。默认读取 CBSD432 开发验证集第一张图，添加固定 seed=0、σ15 的高斯噪声，再用已有的上游 synthetic 基线模型推理。CPU/CUDA 自动选择；当前机器检测不到可用 CUDA，因此采用 CPU。

输出在 `runs/local_demo/`：

- `comparison.png`：原图、含噪图、恢复图横向对比。
- `clean.png`、`noisy.png`、`restored.png`：三张独立图像。
- `metrics.json`：输入、模型哈希、设备、PSNR、SSIM 和耗时。

默认输出可重复覆盖。命令行可指定另一张**干净参考图像**或独立输出目录：

```powershell
.\run_demo.cmd --device cpu --input "实际干净图像路径.png" --sigma 25 --output runs/demo_sigma25
```

此入口用于有参考图像的加噪/去噪演示，使用 `pretrained/synthetic_baseline.pth`，对应上游原始 B0 结构。它不代表本仓库归档实验中的 B0/G1/G2/G3 正式 checkpoint。

## 虚拟环境和数据

`.venv` 是从相邻 `CLIPDenoising/.venv` 复制的独立环境（其 `venv` 目录没有完整深度学习依赖）。已重新生成激活脚本和 24 个命令行启动器，修正复制后的绝对路径，并补齐 psutil、pytest；原项目环境未修改。

- Python 3.9.20、PyTorch 2.2.2+cu121、torchvision 0.17.2+cu121。
- NumPy 1.26.4、SciPy 1.13.1、scikit-image 0.22.0。
- 双击 `activate_env.cmd` 打开激活的命令行，或直接使用 `.venv\Scripts\python.exe`。
- 本机可移植依赖入口为 `requirements.windows.txt`；实际安装快照在 `environment/requirements.windows.verified.txt`。
- 原仓库的 `requirements.local.lock.txt` 是另一环境的控制端快照，不用于本机安装。

已准备 CBSD432（432 张）、CBSD68（68 张）、Kodak24（24 张）、McMaster（18 张）、Urban100（100 张），共 **642 张**。CBSD432/CBSD68 和 RN50 先从原项目复制，再用仓库工具对上游固定版本进行哈希核验；其他数据由仓库工具下载并核验。400/32 开发划分与仓库已有划分一致。

RN50 位于 `pretrained/RN50.pt`，官方 SHA256：`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`。基线去噪模型复制自原项目 `weights/synthetic/net_g_latest.pth`，SHA256：`09c1809b5dbb41222f423d99b545573f0c2061c59363d4e516414b63d196f7dd`；已通过严格完整加载和实际推理。

数据来源、文件哈希和核验记录见 `configs/data_sources.json`、`data/manifest.json`、`data/Urban100_manifest.json`、`pretrained/RN50.manifest.json` 与 `reports/data_audit.md`。

## 验证环境和训练代码

双击 `run_checks.cmd`，或执行：

```powershell
.\run_checks.cmd --no-pause
```

脚本依次执行依赖一致性检查、12 项数据/调度测试、4 项模型结构测试、真实开发图像上的训练/保存/恢复一致性检查、缓存一致性检查。临时文件固定在项目 `tmp/`，避免沙箱系统临时目录的权限问题。检查顺序执行，以减少内存占用。恢复测试包含真实前向、反向和 optimizer update；这些诊断更新独立于正式训练。

JSON 结果在 `reports/calibration_contract_cpu.json`、`reports/resume_contract_cpu.json`、`reports/cache_contract.json`，本次检查的控制台日志在 `reports/local_checks.log`。

本次完整 `run_checks.cmd --no-pause` 已返回 0：依赖检查、12 项数据/调度测试、4 项结构测试、两种累积次数的真实训练恢复测试、缓存一致性测试全部通过。单张开发图像演示的 PSNR 为含噪 24.619 dB、恢复 32.153 dB，SSIM 为 0.6881、0.9380；这是单图运行验证，不是完整基准平均分。

## 启动训练

双击 **`train_local.cmd`** 会按 `configs/local_paths.yaml` 启动/恢复 G1、seed=0 的 100 步 smoke 训练。默认 CPU、micro_batch=1、梯度累积 16 次、FP32、关闭图像缓存。CPU 训练较慢，需要较长时间；本地配置以降低峰值内存为目的。

本地 micro-batch 的 BatchNorm 和 Mixup 行为与一次 batch=16 不同，因此本地 smoke 用于运行验证，不应作为论文或归档实验的严格复现结果。

也可明确指定实验与阶段：

```powershell
.\train_local.cmd --exp B0 --stage smoke --seed 0 --paths configs/local_paths.yaml --resume
```

支持 B0/G1/G2/G3，阶段为 smoke（100 步）、pilot（5,000 步）、full（300,000 步）。输出在 `runs/<阶段>/<实验>/seed<种子>/`。`--resume` 支持首次运行和匹配断点恢复；已有运行会核验代码、配置、数据和权重，拒绝混用不同配置的断点。

具有可用 CUDA 且显存足够时，单个实验可使用保留 batch=16 的 CUDA 配置：

```powershell
.\train_local.cmd --exp G1 --stage smoke --seed 0 --paths configs/local_paths.cuda.yaml --resume
```

CUDA 配置输出在 `runs_cuda/`、`reports_cuda/`，与 CPU 目录独立。本次未验证 GPU 运行：PyTorch 检测到 0 个 CUDA 设备，`nvidia-smi` 返回权限不足。

上游完整主队列仍要求**两张可见 CUDA GPU**，其调度、训练预算和验收要求保持原样。具备该条件后运行：

```powershell
.venv\Scripts\python.exe tools/run_pipeline.py --mode main --paths configs/local_paths.cuda.yaml
```

本仓库没有附带归档实验的完整训练 checkpoint。当前可以直接运行演示、验收和重新训练；重新生成最终八组正式评测，需先完成对应训练。已完成实验的报告保存在 `final_results_20261021/`。
