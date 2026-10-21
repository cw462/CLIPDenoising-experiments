# 已验证命令与执行入口

项目本地目录：`D:\projects\CLIPDenoising_20261020`。
GPU执行目录：`/home/imss/cw/CLIPDenoising_20261020`。
上游commit：`2647bf815df0299a03d16df34b896fa266ceefda`。

本地隔离环境复用系统CPU PyTorch；服务器隔离环境复用
`/home/imss/.conda/envs/copdph4ch_py310` 的CUDA PyTorch，未修改基础环境。

已执行的数据准备及CPU验收：

```powershell
.\.venv\Scripts\python.exe tools/prepare_data.py
.\.venv\Scripts\python.exe tools/prepare_data.py --urban100-only
.\.venv\Scripts\python.exe tests/test_calibration_contract.py --weights pretrained/RN50.pt --device cpu --report reports/calibration_contract.json
.\.venv\Scripts\python.exe tests/test_resume_contract.py --paths configs/local_paths.yaml --device cpu --report reports/resume_contract.json
.\.venv\Scripts\python.exe -m pytest tests/test_data_contract.py
```

服务器主队列入口（双GPU独立实验，无阶段确认）：

```bash
cd /home/imss/cw/CLIPDenoising_20261020
.venv/bin/python tools/run_pipeline.py --mode main --paths configs/local_paths.yaml
```

长期执行由任务专用tmux会话 `clipdenoise_20261020` 承载
`bash launch_pipeline.sh`，日志在 `reports/pipeline_console.log`，
最终进程退出码在 `reports/pipeline_exit.json`。
重启入口相同，先核对进程/配置/断点，已有匹配运行进程时接续监测。

单组调试/恢复例：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_experiment.py --exp B0 --stage full --seed 0 --paths configs/local_paths.yaml --resume
```

只有主队列未在监测该组、且该组进程已结束时才需要手工执行此命令。
CPU诊断、GPU吞吐诊断与smoke/pilot/full各阶段分别记录，不互相折算步数。

最终评测入口只接受8组完整300000次更新后的checkpoint：

```bash
.venv/bin/python tools/evaluate_experiment.py --paths configs/local_paths.yaml --visuals
.venv/bin/python tools/summarize_experiments.py --paths configs/local_paths.yaml
```

这两个入口由主队列自动调用。

性能修订后的配置开启 `cache_decoded_images: true`，并设置
`jobs_per_gpu: 4`：正式8组分别使用独立进程，在两张卡上各运行4组；
smoke/pilot仍每卡1组。每组batch16和300000次更新保持不变。
原运行保存在服务器 `archive_pre_cache/`；新版全部正式组从配对初始化开始。
缓存等价性验收由主队列自动执行：

```bash
.venv/bin/python tests/test_cache_contract.py --paths configs/local_paths.yaml --report reports/cache_contract.json
```

独立容量测试使用 `tmp/benchmark_capacity.py` 和
`tmp/run_capacity_benchmark.py`，结果在 `reports/capacity_benchmark/`；
这些测试不计入正式实验更新数。
