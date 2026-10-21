# 门控＋C：与历史GitHub结果配对的三个种子

> 本页保留此前严格停止策略的设计与失败记录。后续用户授权的独立复核继续策略见 [新执行说明](G1C_Readback_Recovery_20261021.md)。


> 最新结论：本页所述NumPy隔离候选已失败并退役。GPU0独立验证成功760步、第761步GPU张量回读检查失败；GPU1并发seed2成功29步、第30步CPU raw_gt读取检查失败。正式三seed300k未启动，自动接续器只读检查明确拒绝放行。下面保留设计与尝试经过，不能将条件式执行计划当作仍在后台运行。详见[失败记录](../results/g1c_numpy_isolation_failure_20261021/README.md)。

用户固定本轮随机种子为 **0、1、2**，与已上传GitHub的G1完全对应。每组从原RN50和该seed的G1解码器/门控初始化开始，完成300000次优化器更新。不是从已有300k去噪模型续训，不继承smoke或失败run权重。

| 新任务 | 计划执行 | 历史对照 |
|---|---|---|
| G1C seed 0，300k | GPU 0独立单卡 | G1 seed 0，300k |
| G1C seed 1，300k | GPU 1独立单卡 | G1 seed 1，300k |
| G1C seed 2，300k | GPU 1另一独立进程；未验证双任务吞吐前接续seed 1 | G1 seed 2，300k |

每个进程保持真实batch16、FP32、原Mixup/PFA、AdamW和92000/208000学习率周期。编码器参数冻结，BN训练行为保持原实现。目标函数为L1＋0.05×平滑后图像梯度匹配损失，不增加推理参数。三组共900000次更新，无DDP；不再执行单独C、A或重新训练历史基线。

使用`configs/plan_g1c_multiseed_server.yaml`与`tools/run_g1c_multiseed_pipeline.py`，独立输出`runs/g1c_multiseed`和`reports/g1c_multiseed`。每5000步保存完整恢复点，保留最近两份。相同seed配对绑定初始化种子及固定评测噪声；历史训练逐步样本和随机状态没有完整存档，因此不声称两轮完整随机轨迹逐位一致。

当前恢复条件尚未满足：前轮G1C成功完成1634步后，第1635步CPU raw_gt的两次读取相差一个bit，严格保护停止。独立25秒固定buffer短测通过，代码审计未找到明确的异步或别名误报，但没有确定根因或验证修复。配置的`execution_blocked_reason`明确记录此问题；准备队列不启动GPU，正式执行入口也拒绝越过该问题。该标记是已观察故障的执行条件，不是等待用户阶段批准。

## 同版本NumPy构建隔离候选

共享环境有NumPy 1.26.0遗留包元数据和实际1.26.4二进制并存，但实际NumPy/PyTorch二进制均匹配Conda安装校验值，不能据此称二进制损坏。新专用venv使用经过官方SHA256校验的NumPy 1.26.4 wheel，NumPy构建从MKL换为OpenBLAS；PyTorch 2.2.1/CUDA 12.1及原项目其他依赖继续复用。该变更是受控候选排除措施，尚非已定位的根因修复。环境记录与启动器见`reproducibility/g1c_numpy_isolation_20261021/`。

第一次隔离环境尝试在模型运行前因缺少lmdb依赖退出，记录保留。补齐原固定版本lmdb 1.5.1/scikit-image 0.24.0，并以较低路径优先级继承原项目其余依赖后，r2完整导入检查通过；NumPy仍明确来自新venv。

r2于北京时间23:07:49启动，三seed实际GPU初始化/前向/恢复检查均通过，23:09:31开始seed0独立10000步验证，真实batch16、全输入保护和完整300k调度。验证目录为`runs/g1c_numpy_candidate_r2/full/G1C/seed0/numpy_isolation_validation10000`。该权重不进入正式训练。独立driver还须核验连续10000条训练/proof记录、5000/10000两个完整恢复点及全部保护标志，才生成单独的`reproducibility/runtime_g1c_isolated.yaml`并启动全新三seed300k，原blocked基础配置保持不改。

正式输出将使用`runs/g1c_multiseed_isolated`与`reports/g1c_multiseed_isolated`。GPU1独立双进程测试出现任何输入保护异常也会阻止正式启动；仅在实际总吞吐提高且显存余量足够时启用两任务并发。新环境验证未完成前，不能把候选检查通过写成正式训练已完成。

三种子入口、种子对应与评测逻辑的CPU测试不代表实际GPU训练已完成。故障run、数据和历史模型全部保留，失败步骤不自动重试，不放宽字节一致性检查。GPU 1双任务仅在新的独立短测证明总吞吐有益且显存足够后启用；当前默认每卡一个进程。

恢复后自动进行各seed实际模型检查和200步smoke，再运行三个完整300k，完成后自动评测四数据集15种条件。新G1C每种子3150行，三种子9450行；与历史G1对应9450行合并，报告逐seed配对差、跨seed均值和样本标准差、六项最高OOD条件Table1、全部条件与负例。缺少任何seed或完整保护记录，不生成三种子完成结论。

结果经完整性核对后上传用户指定的[GitHub仓库](https://github.com/cw462/CLIPDenoising-experiments)，包含代码、有效配置、指标、对照表、报告与状态；原始数据、模型权重及私有连接信息不随元数据提交。用户已授权自动继续和上传，无需再次询问阶段确认。

可审查准备命令：

```bash
.venv/bin/python tools/run_g1c_multiseed_pipeline.py --prepare-only
```

该命令只写计划状态。故障解除并冻结新执行配置后，正式入口会自动串接检查、训练和评测：

```bash
.venv/bin/python tools/run_g1c_multiseed_pipeline.py
```

当前不应手工清空阻断字段假装修复；没有证据支持新的可靠完成时间。
