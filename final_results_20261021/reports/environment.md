# 实验环境审计

审计日期：2026-10-20。以下区分 Windows 控制端与已连通的 GPU 执行端；Windows 本机缺少 CUDA 不表示远端正式训练资源缺失。

## 代码与工作目录

- 本地工作副本：`D:\projects\CLIPDenoising_20261020`
- 上游仓库：`https://github.com/alwaysuu/CLIPDenoising`
- 上游 commit：`2647bf815df0299a03d16df34b896fa266ceefda`
- 修改后的确切源码以每个 run 的 `effective_config.json`、checkpoint 中的 `source_hash` 为准。该 hash 包含实际 Python 源码；仅凭上游 commit 不能代表本次修改后的实现。
- 评测会重新计算配置内容 hash 和当前源码 hash，并核对 checkpoint 的实验、阶段、seed 与完整 optimizer update 边界。

## Windows 控制端：实际只读查询

| 项目 | 实测结果 |
|---|---|
| 系统 | Windows 11 专业版，10.0.26200 |
| CPU | Intel Core i5-14400，10 核、16 逻辑处理器 |
| 内存 | 总 31.62 GiB；审计时空闲约 7.32 GiB |
| 显卡 | Intel UHD Graphics 730；另有三个远程显示虚拟适配器 |
| NVIDIA/CUDA | 未检测到 NVIDIA 显卡；`nvidia-smi`、`nvcc` 不在 PATH |
| WSL | `wsl --list --verbose` 返回未安装 |
| 系统 Python | `C:\Users\cheng\AppData\Local\Programs\Python\Python314\python.exe`，3.14.3 |
| 本项目隔离环境 | `D:\projects\CLIPDenoising_20261020\.venv\Scripts\python.exe`，复用系统 site-packages 并补充本项目依赖 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| CUDA 查询 | `torch.version.cuda=None`，`is_available=False`，`device_count=0` |

实际导入 torch、torchvision、NumPy、OpenCV、scikit-image、YAML、einops，以及 `torch.from_numpy` 均成功。本地系统 Python 缺少 `psutil`，项目工具应使用已经补齐依赖的 `.venv`，不能把系统 Python 的依赖缺失归因于训练代码。

本地已复用包包括 numpy 2.4.4、Pillow 12.2.0、scipy 1.17.1、opencv-python 4.13.0.92、PyYAML 6.0.3、scikit-image 0.26.0、tqdm 4.67.3、einops 0.8.2。最终依赖以各执行环境的实际快照为准。

`C:\Users\cheng\anaconda3` 是指向 `D:\anaconda3` 的 Junction。另一个既有 `copdph` 环境是 Python 3.11.15 / torch 2.3.1+cpu，但缺少 torchvision 及较多依赖；没有修改这些既有环境。

磁盘首次审计时剩余空间：

| 卷 | 剩余 GiB |
|---|---:|
| C | 2.8 |
| D | 100.8 |
| E | 139.2 |
| F | 68.7 |
| H | 1872.8 |

因此本项目源码、下载、缓存和隔离环境使用 D 盘。正式训练应以执行端实际可用空间及逐次 checkpoint 保存前的磁盘检查为准，不能使用此表代替实时检查。

## GPU 执行端

主控制流程已连接并核验服务器资源，随后向本资源审计子任务提供了以下结果：

- 两张 NVIDIA GeForce RTX 3090，各24GB。用户要求双卡加速后，使用设备0/1运行独立实验；性能修订配置为正式训练每卡4个进程，smoke/pilot每卡1个进程。
- 可复用环境：`.conda/envs/copdph4ch_py310`。
- PyTorch 2.2.1+cu121，torchvision 0.17.1。
- NumPy 1.26.4，SciPy 1.15.3。
- 另建复用 site-packages 的隔离 venv，并在隔离环境补齐 `lmdb`、`scikit-image==0.24` 等必要依赖。

这些是执行端环境信息，不是 Windows 本机硬件信息。具体服务器运行目录、进程标识、GPU 状态和依赖快照由主控制流程保存在运行记录中。未在本报告中填入未经查询的显存空闲量、训练吞吐或预计完成时间。

## 已执行的评测工具检查

- 九个官方 Gaussian / Spatial Gaussian / Poisson 条件生成的最终 float32 输入与上游实现逐元素一致。
- 扩展 Speckle、Salt-and-Pepper 使用稳定哈希种子；重复生成一致，且不改变调用方 NumPy RNG 状态。
- 直接复用官方 RGB 浮点 PSNR / SSIM；相同输入的指标检查通过。
- 检查了逐层等权 OOD 汇总、配对差值，以及噪声 hash 不匹配时拒绝配对。
- 合成小数组仅用于验证 PNG、同色标误差图、统一裁剪和 g/r/g*r 原始数组导出，不作为图像去噪实验结果。
- 使用本地 `.venv` 实际运行了两个评测 CLI：八组正式 checkpoint 不齐时，最终评测按预期返回 blocked；汇总输出只有表头与明确的未完成原因，没有伪造指标。
- 正式训练完成后，同一入口自动生成完整逐图与条件表；本报告中的工具检查不能替代 100 / 5000 / 300000 次更新的真实实验。

运行时使用 checkpoint、配置、源码、数据、噪声和逐图 CSV 的 hash 关联结果。正式评测只在八个 300000-update checkpoint 全部固定后进行；开发评测仅使用 CBSD432 的 32 张开发验证图和 Gaussian σ15。
