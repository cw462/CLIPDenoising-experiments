# GPU恢复数值验收记录

版本说明：本文的具体早期误差数值及失败记录来自性能修订前的验收，
服务器原件保留在 `archive_pre_cache/reports/`。缓存和并发修订后按相同
数值准则重新验收，当前结果以新版 `resume_contract.json` 为准。

CPU恢复和下一步重放均逐位一致。GPU上，磁盘断点刚恢复后的完整状态也逐位一致；
随机状态、采样器、输入/GT、学习率、调度器和更新计数始终要求完全一致。

原模型使用CUDA双线性上采样。PyTorch 2.2官方文档明确说明此操作可能产生非确定性梯度：
https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.interpolate.html

本机RTX3090实测，继续一次GPU更新后出现浮点舍入差异：
首次逐位断言在约3.3e-9差异处失败；按atol1e-7检查时，累积2个诊断micro-batch
的两个权重张量最大绝对差异约1.85e-7，其他状态和loss保持一致。
这些失败保留在 `resume_contract_gpu_atol1e7_failure.json` 中。

最终GPU计算结果使用FP32绝对容差1e-6、相对容差1e-5，与任务书身份输出比较的
建议量级一致。只对继续更新后的模型/优化器浮点值使用此容差；
恢复瞬间的全状态、计数、RNG、采样器、输入仍为逐位比较。
未改变训练算法、上采样、BN、Mixup或确定性全局设置，不宣称300k轨迹逐位重放。
