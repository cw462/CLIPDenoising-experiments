# CLIPDenoising 本地 Codex 实验执行方案

日期：2026-10-20  
版本：连续执行版，检查通过后自动进入正式训练。  
研究任务：基于 CBSD432、CBSD68 等公开数据集的可泛化图像去噪。  
执行状态：这是交给本地 Codex 的任务书，尚未在你的机器上进行训练，不含实测涨点结果。

**主线：完整 CLIPDenoising + PFA → 最后一级 RGB 分支细节校准 → 消融与跨噪声评测。**  
改进对应课题第（3）项“去噪网络设计”；可选的双视图一致性对应第（5）项。默认连续完成结构主线；双视图作为独立扩展，不影响主线继续执行。

## 1 使用方式和边界

将本文件放入本地项目根目录，在已安装的 Codex 中打开该目录，然后复制第13节的“完整主线执行指令”。一次指令覆盖环境、数据、代码、小规模检查、300k正式训练、消融、评测和报告，不需要在阶段之间另发启动指令。

默认执行顺序为阶段0→1→2→3→4→6。阶段3检查通过后自动启动阶段4，完成正式训练和消融后自动进入阶段6。阶段5双视图默认不加入主队列。

**100步和5000步是正确性与训练稳定性检查，不是需要用户再次确认的关卡，也不要求改进模型在短程内必须涨点。** 只要数据、权重、梯度、数值和保存恢复检查通过，就继续正式实验；不要停在“已生成正式命令，等待确认”。训练时间预估用于汇报和安排，不作为额外审批步骤。

本地 Codex 应执行代码修改、检查与运行，不能只复述方案。沿用现有 AGENTS.md 和仓库约束；保护用户已有文件和实验，不覆盖已有输出目录、不执行强制清理。读取路径、安装隔离环境、修改本地实验代码属于本任务；不需要创建远程仓库或推送代码。

缺少数据、预训练权重或 GPU 时，完成其他不依赖该资源的工作，明确列出缺什么、需要放在哪个路径、哪些步骤尚未运行。不得用随机权重或假数据冒充真实复现，不得绕过下载权限。假数据仅用于尺寸与梯度检查，并明确标记为 synthetic smoke。

出现依赖错误、OOM或数值异常时，先按本方案排查并在同一比较协议下修复。只有无法自行解决的资源/权限/实现阻塞才暂停受影响任务并报告原因；不得因测试分数不理想而删实验、挑seed或偷偷改变协议。

默认采用 Linux/WSL2 的命令形式；若用户使用 Windows 原生环境，应提供对应的 PowerShell 命令。不要硬编码不存在的 CUDA、Python、显卡型号、conda 路径或训练耗时。

## 2 原文设置与本方案新增内容

| 项目 | 设置 | 性质 |
|---|---|---|
| 基线 | 冻结 CLIP RN50 + 原卷积解码器 + 原 PFA | 原文完整方法 |
| 训练数据 | CBSD432，在线生成 Gaussian σ=15 | 原文设置 |
| 正式训练 | 300000 次 optimizer update，有效单视图 batch 16，patch 128×128 | 原文设置 |
| 优化 | AdamW，L1；具体分段学习率按下方官方配置 | 论文与代码核对 |
| PFA | γ=0.025；保持原文的逐层乘性扰动 | 原文设置 |
| G1 模块 | 最后一级原始 RGB 分支的门控残差校准 | 本方案提出，待验证 |
| 开发划分 | CBSD432 固定按原图划分 400/32，划分种子 2026 | 本方案约定 |
| 训练种子 | 正式主对比 0、1、2 | 本方案约定 |
| 调试预算 | 100 update 冒烟；5000 update 短程开发 | 本方案约定，不能当正式结果 |

原文没有要求上述开发划分、种子和调试预算。官方配置已核对：manual_seed=100；AdamW weight_decay=1e-4、betas=[0.9,0.999]；梯度裁剪范数0.01；Mixup开启、beta=1.2、use_identity=true；学习率periods=[92000,208000]、eta_mins=[3e-4,1e-6]，即首段保持3e-4，第二段余弦下降。正式受控比较使用本方案的0/1/2种子，其余默认保留官方配置。

Mixup在加噪后同步混合GT与含噪图，因此混合样本的有效噪声方差会改变。准确表述为“基础噪声采样σ=15，并沿用官方Mixup”，不宣称每个最终训练输入都严格具有σ=15。不要为追求概念上的固定噪声强度而静默关闭Mixup。

## 3 阶段 0 环境与代码审计

### 3.1 获取并锁定官方仓库

官方仓库：
https://github.com/alwaysuu/CLIPDenoising

若当前目录没有仓库，可在用户指定的实验目录执行：

```bash
git clone https://github.com/alwaysuu/CLIPDenoising.git
cd CLIPDenoising
git rev-parse HEAD
git status --short
```

如果已有仓库，直接使用并记录当前 commit，不盲目覆盖或更新。新建隔离分支或工作副本，保持原版基线可运行。不要把用户的未提交修改当成可删除文件。

README 已确认的官方入口：

```bash
python basicsr/train.py -opt Denoising/Options/CLIPDenoising_SyntheticDenoising_GaussianSigma15.yml
python Denoising/test_synthetic_denoising.py
```

这两条命令仍需先配置本地数据、RN50 权重和测试 checkpoint 路径，不能认为克隆后就可直接运行。

### 3.2 已核对的真实修改位置

本方案已读取官方源码，路径如下；本地仍须核对所用commit是否一致：

| 文件 | 作用 |
|---|---|
| basicsr/models/archs/CLIPDenoising_arch.py | CLIPDenoising、PFA与up_path |
| basicsr/models/archs/CLIPEncoder_util.py | 编码器及UNetUpBlock |
| basicsr/models/image_restoration_model.py | ImageCleanModel、Mixup、优化和验证 |
| basicsr/data/paired_image_dataset.py | Gaussian训练数据集 |
| basicsr/utils/options.py | 实验输出路径解析 |
| Denoising/utils.py | 测试噪声及指标辅助函数 |

CLIPDenoising.forward的最后一次循环是i=4，调用up_path[4]并传入out[0]，后者是原始含噪RGB。UNetUpBlock先bilinear上采样（align_corners=True）、再center_crop旁路，最后执行torch.cat([up,crop1],1)。默认是64通道解码特征在前、3通道RGB在后。

**只在up_path[4]的crop1进入torch.cat之前做校准，以up作为指导。** 保留其他UNetUpBlock行为和原有参数键，尤其up_path.4.conv_block及last；不要把所有上采样块一起改成校准版本。

先阅读 README、训练 YAML、模型构建逻辑、数据集类和测试脚本。使用 rg 进行定位，例如：

```bash
rg -n "network_g|type:|pretrain|resume|sigma|total_iter|batch_size" Denoising/Options basicsr
rg -n "torch.cat|concat|interpolate|upsample|clip.load|RN50|requires_grad|no_grad" basicsr Denoising
```

追踪训练配置 → 网络类 → forward，确认图 4 中 level-1 的最后一个 Concat：
- 一路是上采样后的解码特征；
- 一路是原始含噪 RGB；
- 拼接后进入原 Conv-block 和最终重建卷积。

记录真实文件名、类名、张量形状、拼接顺序及图像归一化状态。保留原拼接顺序；下面公式中的 [s,y] 只是示意。不得仅凭相似名称修改其他 skip connection。

### 3.3 建立隔离环境

检查操作系统、GPU、可用显存、驱动、Python、PyTorch/CUDA 和磁盘。仓库当前没有根目录requirements.txt或setup.py，先勘察import，再建立独立环境；不要直接执行不存在的安装入口。有兼容性冲突时使用经过实际验证的组合并记录原因，不修改全局环境。

源码含作者机器的sys.path、RN50/data/output路径、GPU编号和自动恢复目录。移植时统一参数化，尤其防止误用作者GPU编号或误恢复其他实验。RN50加载函数会跳过不匹配键，必须输出实际匹配的编码器权重数量和缺失键，确认确实加载了预训练权重。

输出：
- reports/environment.md：硬件、环境、Git commit、运行目录。
- reports/code_audit.md：关键调用链、模型输入范围、数据流与冻结策略。
- requirements.lock.txt 或 environment.yml：实际可用的依赖快照。
- configs/local_paths.yaml：本机数据、预训练权重、输出目录。
- reports/commands.md：本机实际验证过的命令。

官方源码冻结RN50参数，但ImageCleanModel在训练及验证结束后调用net_g.train()，没有单独保持encoder.eval()，因此训练态BN统计会更新。默认B0/G1共同保留这一行为，实际记录BN状态及running statistics变化。若研究固定BN，设置独立freeze_bn实验，并同时训练对应B0/G1，不能把固定BN的收益归给校准模块。

原实现输入为RGB浮点/255后加噪，不对Gaussian含噪图额外clamp，也未使用标准CLIP mean/std归一化。不要习惯性添加CLIP分类预处理、224中心裁剪或归一化。源码虽不返回最深层特征，仍执行额外encoder层；首轮保留实际计算图，参数与耗时按真实代码统计。

## 4 阶段 1 数据与评测协议

### 4.1 数据角色

| 数据 | 用途 | 是否参与选模型 |
|---|---|---|
| CBSD432 的400张开发训练图 | 实现调试、5000步短程开发 | 是 |
| CBSD432 的32张开发验证图 | 仅 σ=15，固定噪声，检查训练与开发选择 | 是 |
| 完整 CBSD432 | 方案固定后的正式训练 | 训练使用 |
| CBSD68 | 最终主测试集 | 否 |
| Kodak24、McMaster、Urban100 | 条件允许时加入最终泛化测试 | 否 |

先按原图分组，再裁 patch；检查重复文件、内容重复和跨集合重叠。保存 splits/dev_train.txt、splits/dev_val.txt、数据来源及 manifest 哈希。

正式训练若用回全部432张图，原32张就不再是独立验证集。主结果使用预先固定的第300000次 update checkpoint；不得再按这32张图的指标选“最佳泛化模型”。开发阶段的模型与正式模型分开保存。

官方 checkpoint 可按固定协议做一次复现核验；最终测试分数不作为搜索 gate 宽度、损失权重或 checkpoint 的依据。若后续确实利用某种 OOD 噪声开发方法，必须将其标成 OOD 开发分布，并重新划出未参与开发的测试条件。

### 4.2 噪声实现

训练使用同一实现的 Gaussian σ=15；若张量在 [0,1]，标准差换算为 15/255。不要在未经核对时添加 clamp、JPEG压缩、颜色转换或多噪声混合。

官方当前测试脚本只实现Gaussian、Spatial Gaussian和Poisson三类。首先复用这三类；Speckle与椒盐属于需要补齐的评测分支，不能声称原脚本已经覆盖。记录实际函数及全部参数：

| 类型 | 评测参数 |
|---|---|
| Gaussian | σ=15为ID；σ=25、50为OOD |
| Spatial Gaussian | σ=45、50、55 |
| Poisson | α=2.5、3.0、3.5，定义以作者代码为准 |
| Speckle | 方差0.02、0.03、0.04，不得当标准差使用 |
| Salt and Pepper | d=0.012、0.016、0.02 |

Spatial Gaussian在官方代码中是对高斯噪声逐通道做3×3均值卷积，不是任意绘制空间变化的σ图。Poisson复用Denoising/utils.py的add_poisson_noise；其scale用于缩放采样噪声，不能直接替换成另一种λ参数化。官方脚本还包含Spatial Gaussian=40、Poisson=2，可作为附加条件；与论文主表对应的条件单列。

Speckle与椒盐优先查论文所引用的MaskDenoising实现，记录来源。若无法取得对应实现，可以建立明确标注的扩展测试：Speckle令y=x+x*ε、ε~N(0,v)，不额外截断含噪输入；椒盐对每个空间像素以d/2概率把三个RGB通道同时置0、以d/2概率同时置1。该定义是本项目扩展协议，不能拿其结果直接与论文表格声称精确复现。

官方三类噪声的主要复现表沿用原脚本在每张图前np.random.seed(0)的行为；B0/G1共用输入。扩展或额外噪声重复试验可按图/类型/强度使用稳定哈希种子，并单独标明协议，避免Python内置hash的跨进程变化。源域验证噪声也固定，不把噪声抽样差异误判成模型错误。

### 4.3 指标实现

官方独立测试将输入reflect-pad到32的倍数，推理后裁回原尺寸并clamp到[0,1]，用RGB浮点评估。目录标签默认为CBSD68、McM、Kodak、Urban100，路径配置需映射到实际下载名称。核对SSIM版本、PSNR范围及裁边，保留官方指标实现。保存逐图指标后按图平均，不把整个数据集的总MSE转换值与平均逐图PSNR混在一起。

原训练YAML把CBSD68的σ50设为validation；开发研究必须替换为CBSD432的32张σ15验证图，所有对比一致。训练过程与最终验证还可能走float/uint8不同分支，因此主结果统一用独立评测入口，不混用日志中的不同指标。

最终评测使用原图尺寸；若必须分块，记录块大小、重叠和融合策略，所有模型一致。不能用128×128测试裁块结果替代原图benchmark结果。

## 5 阶段 2 实现细节校准模块

### 5.1 插入位置与数学定义

记 y 为最终 RGB 旁路的原始含噪张量，s 为该位置上采样后的解码特征。保留原 CLIP 输入、F1至F4、多尺度解码和PFA。

定义：

```text
h = y - B(y)
z = concat(P(s), y, h)
t = Conv3x3(z, 32) -> ReLU -> Conv3x3(32, 32) -> ReLU
g = sigmoid(Conv1x1(t, 1))
r = Conv3x3(t, 3)
y_cal = y + g * r
output = original_reconstruction_head(original_order_concat(s, y_cal))
```

约定：
- P：1×1卷积，将实际 s 通道数投影到16；通过代码确定输入通道，不能硬写假定的64。
- B：3×3固定均值滤波，边界使用明确记录的padding，保证尺寸不变。
- g：单通道空间门控，广播到RGB。
- r：三通道校准量；最后一个卷积权重和偏置均零初始化。
- y_cal 不在新增模块内进行额外截断；后续沿用原实现。
- r零初始化使初始 y_cal=y；不再同时加一个初始化为零的外部缩放系数。
- h包含纹理与噪声，只用作输入线索，不称其为“真实噪声”。
- g表示模型学到的修正幅度调制，不宣称已校准为概率意义上的置信度。

最终模型仍直接预测干净图像。第一轮只训练解码器和新增模块，保持原L1损失；不同时加入SSIM损失、频域损失、噪声估计器、F5、LoRA或额外基础模型。

### 5.2 代码组织

保持官方B0可运行，新增网络变体及配置开关，例如 adapter_mode: none/gated/ungated/no_decoder_guidance。实际注册方式以仓库为准。

建议新增路径属于待创建文件，不是声称官方已有：
- configs/experiments/：变体和预算配置。
- tools/run_experiment.py：将实验配置转换为官方可执行配置并调用训练入口。
- tools/run_pipeline.py：顺序执行检查、正式训练、消融与评测，保存队列进度。
- tools/evaluate_experiment.py：统一评测与逐图结果导出。
- tools/summarize_experiments.py：汇总、差值和可视化。
- tests/test_calibration_contract.py：只验证本次关键结构和梯度风险。
- runs/<stage>/<experiment>/<seed>/：独立日志、权重和实际生效配置。

尽量复用原数据加载、优化器和评测函数，避免为了新接口重写完整训练框架。

### 5.3 必须通过的实现验收

1. 共享参数加载：增强模型加载基线权重时，只允许新增模块对应的missing keys；其他missing/unexpected keys必须处理，不能用strict=False静默跳过。
2. 身份初始化：相同共享权重、相同输入、eval、PFA关闭时，B0与零初始化G1输出在FP32下 allclose（建议atol=1e-6、rtol=1e-5）。
3. 禁用模块：none模式恢复原路径，尺寸、拼接顺序与输出一致。
4. 梯度：L1反传后解码器及修正头有梯度，冻结CLIP参数无梯度。r零初始化会使其他新增分支首步梯度暂时为零，检查若干更新后是否开始获得梯度，不误报首步失败。
5. 冻结状态：encoder参数不更新；默认沿用官方训练态BN，运行统计允许按原代码更新，并检查B0/G1策略一致。只有独立freeze_bn实验才要求相关buffers不变。PFA按预期仅训练启用。
6. 尺寸：128×128和一张非整齐尺寸的图都能按原padding策略正确返回原尺寸。
7. 断点恢复：保存/恢复模型、优化器、scheduler、AMP scaler（若有）、update计数和随机状态；恢复不能重新从学习率起点开始。
8. 参数与性能：实测总参数、可训练参数、峰值显存和耗时；不要预写估算值当测量值。

## 6 阶段 3 冒烟和短程开发

先B0，再G1，串行运行。

本阶段完成且关键检查通过后，自动固定本轮实现和配置，转入阶段4，不等待用户重复授权。5000步暂时没有性能提升不构成停止正式训练的理由。

- 冒烟：真实开发训练图，100次optimizer update，检查loss有限、训练能前进、权重可保存、验证和可视化可完成。
- 短程：B0和G1各5000次update，同一开发划分和σ15源域验证。默认保留正式学习率日程的前5000步，不压缩成5000步的余弦退火；不能把短程实验标成完整300k的复现。
- 默认从相同CLIP预训练权重和配对的解码器初始化开始。若为了调试使用官方去噪checkpoint，B0/G1都从同一个checkpoint出发，均获得相同额外训练预算，并明确标成warm-start开发实验。
- 不在5000步短程中使用最终OOD测试集筛选结构。短程用于检查loss是否有限、梯度是否正确、训练及保存恢复是否稳定；G1暂时未优于B0不阻止正式训练。实现错误修复并复查后继续，不能根据短程判断最终OOD收益。
- 用实际稳定运行后的秒/update估算正式单组时间：剩余update×秒/update；记录验证与保存额外开销，不预先承诺几小时完成。

OOM时优先降低micro-batch并使用梯度累积。例如单卡micro-batch=4、累积4次，可得到优化器意义的有效batch=16。scheduler、保存间隔和总300k均按optimizer update计数，累积loss正确归一化。**本代码含训练态BN及batch内Mixup，累积与真实batch16并不完全等价**；B0/G1必须使用相同micro-batch、累积及Mixup策略，并标注该协议差异。不要为掩盖这一差异而静默冻结BN或关闭Mixup。双视图损失的配对不能跨样本错配。AMP若启用，对照组一致，并先验证数值稳定。

实现累积时，原训练包装类中的zero_grad、optimizer.step、梯度裁剪与EMA/scheduler更新也须按optimizer update调整，不能只在外层循环除loss。梯度范数0.01的裁剪在全部micro-batch反传结束、AMP unscale之后、optimizer.step之前执行一次。micro-batch统一以noisy views计数：单视图4个clean、双视图2个clean各2视图，均为4 views；累积4次得到16 views，禁止双视图无意变成32 views。

## 7 阶段 4 正式结构实验

开发设计固定后，正式重新训练；不继承400/32开发阶段的pilot权重，各组从规定的预训练CLIP和配对解码器初始化开始，重新计满300k。不要把原官方checkpoint与额外训练了300k的改进模型直接当成同预算比较。

| ID | 方法 | 必需性 | 说明 |
|---|---|---|---|
| B0 | 原CLIPDenoising + PFA | 必需 | 所有新模块关闭 |
| G1 | B0 + 门控细节校准 | 必需 | 主改进 |
| G2 | B0 + 无门控残差校准 | 结构消融 | 使用 y+r，其他局部结构尽量相同，报告参数差异 |
| G3 | B0 + 不使用解码特征的校准 | 结构消融 | 移除P(s)指导，只根据y和h预测，报告参数变化 |

B0与G1分别使用seed 0、1、2，均训练300000次update。共同参数使用配对初始化，数据与噪声随机流尽量独立于新增模块初始化，所有实验固定seed映射。

默认按以下顺序串行完成8组正式训练，每组均使用完整CBSD432和300000次optimizer update：

| 队列顺序 | 实验 | seed |
|---|---|---|
| 1 | B0 | 0 |
| 2 | G1 | 0 |
| 3 | B0 | 1 |
| 4 | G1 | 1 |
| 5 | B0 | 2 |
| 6 | G1 | 2 |
| 7 | G2 | 0 |
| 8 | G3 | 0 |

默认队列共240万次正式optimizer update，另加小规模检查。全部自动串行推进，不能完成seed 0后就停等第二条指令。G2/G3默认仅seed 0，在报告中明确为单种子消融；如扩展到多seed，应作为预先记录的附加实验，不把单次差异表述为稳定机制结论。

8组训练完成、checkpoint和协议固定后，自动进入阶段6统一评测与报告。不要依据前两组的最终OOD分数重新设计后续实验。

在没有更改算法或预算的前提下完成预定实验；不要因为某个测试噪声没涨点就删除该条件或只汇报最佳seed。出现负结果时照实保留。

原代码若被发现需要修复，先单独验证修复后的共同基线，并使用相同修复重新训练各变体，不能让修复仅出现在G1。

## 8 可选阶段 5 双视图一致性

完成结构主线后再开展。本阶段保留Gaussian σ15基础采样和官方Mixup设置。

从同一干净patch分别生成独立噪声y1=x+n1、y2=x+n2。启用Mixup时，对GT和两路输入共用同一混合系数、配对置换及identity开关，保持两路GT一致。将两路含噪图拼成一个batch前向，再拆回成对预测，避免额外改变BN更新次数；PFA按元素独立采样。损失如下：

```text
L_rec = 0.5 * (mean_abs(pred1 - x_mix) + mean_abs(pred2 - x_mix))
L_con = mean_abs(pred1 - pred2)
L = L_rec + lambda_consistency * L_con
```

x_mix是两路共用的混合后GT；identity分支时x_mix=x。完成成对Mixup后，必须绕过ImageCleanModel原来的通用Mixup调用，避免拼成2B后被再次随机混合、破坏两路配对。V0/V1及GV0/GV1共用同一成对增强实现。

lambda第一版预先设为0.05；0为必要对照。训练损失只对可训练网络的输出产生约束，不把两个冻结输入CLIP特征的距离当作能更新decoder的损失。

保持每次optimizer update处理16个noisy views：
- B0：16个单视图patch；
- V0/V1：8个干净patch，各生成2个噪声视图，共16个noisy views；
- 所有组300k update，另外记录每update的unique clean patches、noisy views和实际耗时。

这能控制主要前向计算量，但单视图与双视图的独立内容数不同；因此一致性贡献主要由V1与V0的比较判断。

| ID | 方法 |
|---|---|
| V0 | B0 + 双视图监督，lambda=0 |
| V1 | B0 + 双视图监督，lambda=0.05 |
| GV0 | G1 + 双视图监督，lambda=0 |
| GV1 | G1 + 双视图监督，lambda=0.05 |

若研究组合效果，GV0也是必要对照，避免把双视图采样效应归给一致性。推理只用单视图、单次前向。不要为了对比把双视图clean batch仍设为16却不披露翻倍的noisy views。

## 9 阶段 6 评测与可视化

### 9.1 正式输出

每个模型在相同测试图和噪声实例上保存：
- 逐图PSNR、SSIM；
- 条件均值、相对B0的逐图配对差值；
- 训练种子维度的均值与标准差；
- 总/可训练参数、单张推理耗时、峰值显存；
- checkpoint、源代码commit、配置哈希、数据/噪声清单标识。

CUDA耗时先预热、计时前后同步，固定图像尺寸与batch=1；把读取、噪声生成、模型推理和文件保存时间分开。计算受限可仅对预先指定的标准尺寸测耗时；不要把不同尺寸的平均耗时直接比较。

优先按“数据集×噪声类型×强度”完整报告。若给OOD总分，先声明聚合规则：各图等权→同族强度等权→纳入的噪声族等权→数据集等权；Gaussian σ15不计入OOD平均。官方代码支持的三族主表与补充到五族的扩展表分开，列清楚纳入条件，不混成一个“官方复现总分”。同时保留每条件结果，避免总分掩盖退化。

### 9.2 建议结果字段

```text
experiment,train_seed,stage,checkpoint,update,git_commit,config_hash,
dataset,image_id,noise_type,noise_level,noise_seed,
psnr,ssim,forward_ms,peak_memory_mb
```

运行元信息另存manifest.json，包含训练图数量、unique clean patches/update、noisy views/update、micro-batch、累积步数、AMP、实际依赖和总训练时间。

### 9.3 可视化与机制分析

固定选择示例图和局部坐标，保存干净图、含噪图、B0、G1、绝对误差图、相同区域放大图。误差图使用同一色标；PNG可视化不要先JPEG压缩。

额外保存g、r及g*r。门控图必须结合修正量看，不能仅凭g大就断言该处噪声多。使用GT只做离线平坦区/边缘/纹理区分析，不输入推理网络。记录失败案例与退化条件，不只挑好看的样例。

生成：
- reports/main_results.csv
- reports/per_image_results.csv
- reports/ablation_results.csv
- reports/runtime_results.csv
- reports/figures/
- reports/experiment_report.md

报告中分开“已运行结果”“未运行实验”“机制假设”。如果不能稳定涨点，明确列出差值与波动，不编造预期增益。

## 10 给本地 Codex 的脚本接口要求

以下是要求Codex实现的接口，不是官方已有命令。先实现并实际跑通，再把准确命令写到reports/commands.md：

默认使用一个总入口连续执行主线；具体脚本须由本地Codex实现并验证：

```bash
python tools/run_pipeline.py --mode main --paths configs/local_paths.yaml
```

总入口负责按阶段和第7节队列自动推进，默认包含正式训练，完成后调用统一评测和报告生成。下列单组命令用于调试或恢复，不代表每组结束都需要用户再次发指令：

```bash
python tools/run_experiment.py --exp B0 --stage smoke --seed 0 --paths configs/local_paths.yaml
python tools/run_experiment.py --exp G1 --stage smoke --seed 0 --paths configs/local_paths.yaml
python tools/run_experiment.py --exp B0 --stage pilot --seed 0 --paths configs/local_paths.yaml
python tools/run_experiment.py --exp G1 --stage pilot --seed 0 --paths configs/local_paths.yaml
python tools/run_experiment.py --exp B0 --stage full --seed 0 --paths configs/local_paths.yaml
python tools/run_experiment.py --exp G1 --stage full --seed 0 --paths configs/local_paths.yaml
```

要求：
- --stage smoke/pilot/full 明确对应100/5000/300000次optimizer update，使用不同run目录。
- --exp控制结构，所有变体复用共同baseline配置。
- 生成实际生效配置，保留命令和日志；路径按本机配置解析。
- 增加断点恢复选项，恢复前校验实验ID、配置与checkpoint匹配。
- 已存在同名输出时默认拒绝覆盖；允许显式resume或新的run标签。
- 训练单卡串行排队，避免多个300k任务抢显存。
- 正式评测单独执行并保留测试访问记录，不能每隔几百步偷偷用最终测试集挑checkpoint。

持续执行与恢复要求：
1. 保存runs/pipeline_state.json及reports/progress.md，记录阶段、实验ID、seed、源码/配置哈希、状态、当前optimizer update、checkpoint、日志、进程标识和失败原因。状态至少区分pending/running/completed/blocked。
2. 每个正式run仅在达到300000次optimizer update且最终checkpoint可加载时标记completed。进程仍运行或仅发出了启动命令，不能标成训练完成。
3. 每5000次正式optimizer update保存一次可恢复状态，并在结束时保存最终权重；保存发生在完整update边界，采用原子写入，至少保留最近两份有效恢复点。包含模型及BN buffers、优化器、scheduler、计数、RNG、AMP scaler及EMA（启用时），记录数据采样器恢复方式。
4. 中断后读取队列状态与实际进程状态：跳过已验证完成且配置匹配的组，恢复未完成组；已有匹配进程仍运行时不要重复启动。恢复前核对配置和checkpoint，禁止误用作者硬编码的自动恢复目录。状态不完整时明确标记restart，不伪装连续训练。
5. 优先使用本机已有且可验证的持久进程/任务队列运行长任务，保存日志和退出码；不要默认聊天窗口关闭后进程一定继续。当前会话能继续时保持执行和检查；会话中断后，下次读取状态接续同一队列。
6. OOM、NaN或磁盘不足先定位并处理，无法恢复时记录具体阻塞，不无限重试，也不暗中减少训练步数或改变实验。只要资源与实现正常，各组和阶段之间不增加确认步骤。

如果原仓库已有足够的runner，可复用已有接口而不新建包装器，但必须提供与这些任务等价、实际验证过的命令。

## 11 完成标准

默认主线完成标准：
- [ ] B0与G1可在同一环境、同一协议训练和推理。
- [ ] 零初始化等价性、梯度、冻结状态和恢复检查通过。
- [ ] B0/G1分别完成seed 0、1、2，各300k；G2/G3分别完成seed 0，各300k，共8组正式训练。
- [ ] CBSD68以及可获得的其他公开集完成ID/OOD统一测试。
- [ ] 有逐图结果、完整条件表、局部对比和失败案例。
- [ ] 完成无门控与无解码指导消融；若仍有阻塞，则主任务标记为未完成并准确列出缺项。
- [ ] 给出实际参数/耗时和源代码、配置、checkpoint关联。
- [ ] 自动生成最终报告和队列状态；明确区分已完成、运行中与阻塞，不把后台启动当成实验结束。

“框架跑通”和“稳定涨点”是两个不同结论。5000步开发不替代300k；单个seed的一次微小提升不直接表述为稳定优于基线。

## 12 对课题七个环节的覆盖

| 环节 | 对应实现和输出 |
|---|---|
| 1 数据预处理与噪声模拟 | 数据manifest、源域划分、官方噪声生成器、固定测试噪声 |
| 2 特征提取与表征 | 冻结CLIP RN50的多尺度特征及状态审计 |
| 3 去噪网络设计 | 原解码器与新增末级RGB细节校准模块，核心改进 |
| 4 特征迁移与鲁棒表示 | 预训练特征冻结迁移、PFA与跨噪声验证 |
| 5 损失与训练策略 | 原L1；可选双视图一致性和公平预算对照 |
| 6 重建与可视化 | 原图尺寸恢复、误差图、局部细节、校准图 |
| 7 评估与消融 | ID/OOD指标、G2/G3消融、多seed和效率分析 |

## 13 可直接粘贴的 Codex 指令

### 完整主线执行指令

```text
请读取当前目录的 CLIPDenoising_Local_Codex_Plan.md，实际连续执行完整主线：阶段0→1→2→3→4→6。

先核对官方源码、最终Concat位置和本机资源，配置数据与预训练权重，保留原PFA、Mixup、学习率、BN行为和L1损失，实现B0与最后一级RGB旁路校准模块G1，以及G2/G3消融。保护已有文件和实验输出。

依次完成正确性检查、B0/G1各100次optimizer update运行检查和各5000次短程实验。检查通过后自动启动正式训练，不停在“等待确认”或仅输出命令。短程暂时没有涨点不构成停止理由；发现真实实现问题先修复、复查，再继续。

正式训练使用完整CBSD432，基础Gaussian噪声σ15并沿用官方Mixup。各组从同一预训练CLIP与配对的解码器初始化开始，不能继承pilot权重；同一正式run的有效断点可正常恢复。每组训练300000次optimizer update，串行队列为：
B0 seed0 → G1 seed0 → B0 seed1 → G1 seed1 → B0 seed2 → G1 seed2 → G2 seed0 → G3 seed0。

实现并运行主队列入口，保存pipeline_state.json、progress.md、日志和完整断点。各组之间自动推进，避免重复启动；已有完成结果且配置一致时核验后复用。不要静默减少种子或步数。

全部正式checkpoint固定后，自动统一评测CBSD68及可获得的其他公开集，完成逐条件PSNR/SSIM、配对差值、多seed波动、单seed消融、参数/耗时及统一可视化，生成experiment_report.md。最终测试集不参与结构、超参或checkpoint选择；失败和负结果照实保留。

不要只给建议或代码，实际执行以上流程。资源齐备时不需要我再次确认进入300k训练。依赖、OOM、NaN等问题先自行排查；无法解决的真实阻塞再报告具体原因、已完成工作及所需资源。不得编造指标，也不得把任务启动称为训练完成。

默认主线完成后交付报告；双视图阶段5不属于本次默认队列，不在主线中额外插入。
```

### 中断后继续指令

```text
请读取 CLIPDenoising_Local_Codex_Plan.md、runs/pipeline_state.json 和 reports/progress.md，恢复同一主实验队列。

先核对真实进程、源码/配置哈希和checkpoint状态，不重复启动仍在运行的组，不覆盖已完成结果。跳过验证通过的completed组，从未完成组的完整断点续训；缺少有效状态时明确记录restart或具体阻塞。

自动继续剩余300k训练、消融、统一评测及报告，不要求我逐阶段确认。保持原比较协议，不根据测试成绩改结构或筛选结果。最后准确报告已完成与未完成项目。
```

### 可选双视图扩展指令

```text
在主线之外，依据任务书阶段5开展双视图扩展，先验证V0和V1，再按研究范围验证GV0/GV1。

保持每个optimizer update总noisy views=16：8个clean patch各2个独立σ15噪声视图。lambda=0和0.05分别为对照与一致性组，保留原PFA与成对Mixup，共用混合后GT，避免训练包装类再次Mixup。

先跑正确性检查，通过后自动进行每组300000次optimizer update的正式训练；第一轮固定seed0。记录unique clean patches、noisy views、训练时间与单种子局限。独立保存扩展队列，不覆盖主线结果。

一致性的贡献主要比较V1-V0与GV1-GV0，推理保持单次前向。不要引入额外训练噪声类型，不要把两个冻结输入特征的距离当作可训练损失。评测与报告保留完整条件和未改善结果。
```

## 14 来源

1. Cheng, Liang, Tan. Transfer CLIP for Generalizable Image Denoising. CVPR 2024。依据用户提供的论文正文与补充材料，第3节、第4.1节及架构图。
   https://arxiv.org/abs/2403.15132
2. 官方代码与README训练入口：
   https://github.com/alwaysuu/CLIPDenoising
3. 官方README提供的预训练模型入口：
   https://drive.google.com/drive/folders/1F8md2zen0iPlhGnI5-VdjdBOLY8DBIiF?usp=drive_link
4. 已核对的训练配置：
   https://github.com/alwaysuu/CLIPDenoising/blob/main/Denoising/Options/CLIPDenoising_SyntheticDenoising_GaussianSigma15.yml
5. 已核对的架构与融合实现：
   https://github.com/alwaysuu/CLIPDenoising/blob/main/basicsr/models/archs/CLIPDenoising_arch.py
   https://github.com/alwaysuu/CLIPDenoising/blob/main/basicsr/models/archs/CLIPEncoder_util.py
6. 已核对的训练包装类和独立测试：
   https://github.com/alwaysuu/CLIPDenoising/blob/main/basicsr/models/image_restoration_model.py
   https://github.com/alwaysuu/CLIPDenoising/blob/main/Denoising/test_synthetic_denoising.py

本方案中的校准模块、开发划分、运行阶段、实验编号与一致性权重是为本课题拟定的实验设计，不能写成原文已经验证的改进。

