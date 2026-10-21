# 固定样例图索引

共 48 张 B0/G1 seed0 对比图，来自 4 个数据集的 8 张固定图像，每张图像对应 6 种预先指定的噪声条件。已核对全部 48 个 PNG 文件存在、文件头有效、映射不重复，且与当前目录的 PNG 文件集合完全一致。每个数据集各 12 张。

索引依据 [per_image_results.csv](../per_image_results.csv) 中 `experiment=G1`、`train_seed=0` 且 `visual_directory` 非空的记录生成，并核对相同条件的 B0 记录及噪声、数据清单和源码哈希一致。此目录只建立下载图像的链接，没有重新生成评测或图片。

文件名沿用汇总脚本的原始规则：`dataset + '_' + SHA256('/'.join((train_seed, dataset, image_id, noise_type, noise_level)))[:16] + '.png'`。哈希使用 CSV 原始字符串；强度值不重新格式化。

每张图包含 Clean、Noisy（仅显示时裁剪）、B0、G1、B0 绝对误差、G1 绝对误差；误差显示范围固定为 [0, 0.20]，B0/G1 裁剪坐标一致。这些是固定样例，不是按指标挑选的最佳案例；结论应结合 [完整中文分析](../results_interpretation_zh.md) 和全部逐图指标。

| 序号 | 数据集 | 图像 | 噪声类型 | 强度 | PNG |
|---:|---|---|---|---:|---|
| 1 | CBSD68 | 101085.png | gaussian | 15 | [查看对比图](CBSD68_15b745d5cb68ac87.png) |
| 2 | CBSD68 | 101085.png | gaussian | 50 | [查看对比图](CBSD68_9c41ab783abc484d.png) |
| 3 | CBSD68 | 101085.png | poisson | 3.0 | [查看对比图](CBSD68_72a7552688745e68.png) |
| 4 | CBSD68 | 101085.png | salt_pepper | 0.016 | [查看对比图](CBSD68_e9379ff50a606ae8.png) |
| 5 | CBSD68 | 101085.png | spatial_gaussian | 50 | [查看对比图](CBSD68_23287e31fa11aa8b.png) |
| 6 | CBSD68 | 101085.png | speckle | 0.03 | [查看对比图](CBSD68_da2646a90dfcc340.png) |
| 7 | CBSD68 | 101087.png | gaussian | 15 | [查看对比图](CBSD68_7ee4074a54cc7b73.png) |
| 8 | CBSD68 | 101087.png | gaussian | 50 | [查看对比图](CBSD68_91dc64c4f519d5ef.png) |
| 9 | CBSD68 | 101087.png | poisson | 3.0 | [查看对比图](CBSD68_a75b9e8bd53bd47f.png) |
| 10 | CBSD68 | 101087.png | salt_pepper | 0.016 | [查看对比图](CBSD68_a7e14a692384f6f9.png) |
| 11 | CBSD68 | 101087.png | spatial_gaussian | 50 | [查看对比图](CBSD68_7c3f9c7f6ab53c5b.png) |
| 12 | CBSD68 | 101087.png | speckle | 0.03 | [查看对比图](CBSD68_a8318196f6e06324.png) |
| 13 | Kodak24 | kodim01.png | gaussian | 15 | [查看对比图](Kodak24_43f9253256ced957.png) |
| 14 | Kodak24 | kodim01.png | gaussian | 50 | [查看对比图](Kodak24_a7fc2fd9f0a8ae99.png) |
| 15 | Kodak24 | kodim01.png | poisson | 3.0 | [查看对比图](Kodak24_0a9c456faec3d61b.png) |
| 16 | Kodak24 | kodim01.png | salt_pepper | 0.016 | [查看对比图](Kodak24_f36f904a92bcf141.png) |
| 17 | Kodak24 | kodim01.png | spatial_gaussian | 50 | [查看对比图](Kodak24_260fd2e4476c0feb.png) |
| 18 | Kodak24 | kodim01.png | speckle | 0.03 | [查看对比图](Kodak24_1edcd32dc18cce69.png) |
| 19 | Kodak24 | kodim02.png | gaussian | 15 | [查看对比图](Kodak24_6ef964983f915bd8.png) |
| 20 | Kodak24 | kodim02.png | gaussian | 50 | [查看对比图](Kodak24_51fa996c79100239.png) |
| 21 | Kodak24 | kodim02.png | poisson | 3.0 | [查看对比图](Kodak24_43963d35676e1d03.png) |
| 22 | Kodak24 | kodim02.png | salt_pepper | 0.016 | [查看对比图](Kodak24_ebeb98b4bde2bcd6.png) |
| 23 | Kodak24 | kodim02.png | spatial_gaussian | 50 | [查看对比图](Kodak24_c3ef65d1723b1772.png) |
| 24 | Kodak24 | kodim02.png | speckle | 0.03 | [查看对比图](Kodak24_8d916f77780d8ffb.png) |
| 25 | McMaster | 1.tif | gaussian | 15 | [查看对比图](McMaster_20d5df43ba935594.png) |
| 26 | McMaster | 1.tif | gaussian | 50 | [查看对比图](McMaster_032eb727996bd538.png) |
| 27 | McMaster | 1.tif | poisson | 3.0 | [查看对比图](McMaster_05e7b857acaa9b96.png) |
| 28 | McMaster | 1.tif | salt_pepper | 0.016 | [查看对比图](McMaster_9e5622a0c21d6ed2.png) |
| 29 | McMaster | 1.tif | spatial_gaussian | 50 | [查看对比图](McMaster_78d2607d83198770.png) |
| 30 | McMaster | 1.tif | speckle | 0.03 | [查看对比图](McMaster_188ad45fd001249a.png) |
| 31 | McMaster | 10.tif | gaussian | 15 | [查看对比图](McMaster_948e2ae08d433089.png) |
| 32 | McMaster | 10.tif | gaussian | 50 | [查看对比图](McMaster_6c6fea188448b6ff.png) |
| 33 | McMaster | 10.tif | poisson | 3.0 | [查看对比图](McMaster_bacdfc2217b84a07.png) |
| 34 | McMaster | 10.tif | salt_pepper | 0.016 | [查看对比图](McMaster_447c902f8f7dc574.png) |
| 35 | McMaster | 10.tif | spatial_gaussian | 50 | [查看对比图](McMaster_02fa1db97ad7c25d.png) |
| 36 | McMaster | 10.tif | speckle | 0.03 | [查看对比图](McMaster_81724f938db04a74.png) |
| 37 | Urban100 | img_001_SRF_2_HR.png | gaussian | 15 | [查看对比图](Urban100_a8aa12fc7e47e1c4.png) |
| 38 | Urban100 | img_001_SRF_2_HR.png | gaussian | 50 | [查看对比图](Urban100_5efb12bdbad8dd25.png) |
| 39 | Urban100 | img_001_SRF_2_HR.png | poisson | 3.0 | [查看对比图](Urban100_8e98f0ca9235d7b5.png) |
| 40 | Urban100 | img_001_SRF_2_HR.png | salt_pepper | 0.016 | [查看对比图](Urban100_ab5d141d77e634c9.png) |
| 41 | Urban100 | img_001_SRF_2_HR.png | spatial_gaussian | 50 | [查看对比图](Urban100_21a6b88e03f91dc7.png) |
| 42 | Urban100 | img_001_SRF_2_HR.png | speckle | 0.03 | [查看对比图](Urban100_0ed0e43e8cda06a0.png) |
| 43 | Urban100 | img_002_SRF_2_HR.png | gaussian | 15 | [查看对比图](Urban100_b4e65055b01946c7.png) |
| 44 | Urban100 | img_002_SRF_2_HR.png | gaussian | 50 | [查看对比图](Urban100_26fa6558309a372e.png) |
| 45 | Urban100 | img_002_SRF_2_HR.png | poisson | 3.0 | [查看对比图](Urban100_4fc12ce8104407c3.png) |
| 46 | Urban100 | img_002_SRF_2_HR.png | salt_pepper | 0.016 | [查看对比图](Urban100_e5ac9f1ae48c53af.png) |
| 47 | Urban100 | img_002_SRF_2_HR.png | spatial_gaussian | 50 | [查看对比图](Urban100_b5d78146b8fc9427.png) |
| 48 | Urban100 | img_002_SRF_2_HR.png | speckle | 0.03 | [查看对比图](Urban100_0d58961cf5cf1b77.png) |

