# Swin Transformer 木马家族分类训练

以 Tomwhard/BIT-senior-primarysemester 的 92a2f05
版本 data(2).zip 中的 SwinClassifier 为基础，保留原窗口注意力、移位窗口、
Patch Merging、多阶段结构。models.py 保留原CNN/ViT定义作为源码参照，
本包训练与预测入口只实例化 SwinClassifier，不会切换到树模型或CNN。

## 文件说明

| 文件 | 内容 |
|---|---|
| models.py | 仓库原架构，开放MLP比例并增加通道/头数校验 |
| train.py | 训练、逐轮训练/验证准确率、最后测试、准确耗时统计 |
| data.py | 按清洗主表划分读取图片；.bytes转换与原全文件灰度图规则一致 |
| plots.py | 样本数、学习曲线、运行时间、吞吐量和混淆矩阵 |
| predict.py | 最佳模型对单个灰度PNG或.bytes预测家族 |
| compare.py | 轻量化实验的验证集精度/参数量/运行时间对比 |
| tests/test_pipeline.py | 前向反向、轻量化、统计口径、异常输入测试 |
| prepare_data.py | 从仓库数据压缩包中提取训练所需的主表和图片 |

## 数据与环境

需要 Python 3.10+ 和 requirements.txt 中的依赖。
CPU可执行安装 `python -m pip install -r requirements.txt`；NVIDIA显卡请先根据
PyTorch官方安装页安装适合驱动的CUDA版torch，再装其余依赖。
不需要torchvision，不执行任何恶意软件文件。

解压仓库的 data(2).zip，数据目录应含 dataset.csv 和 images/，对应803张32x32灰度图。
本包不包含原始数据；--data-dir明确指向解压得到的 data/final/data。
图像来自完整字节序列分成1024段后求已知字节均值，不是前1024字节截图。
仅使用图像支路，不使用355维静态特征或2305维融合向量。

## Windows / Anaconda 快速开始

在 Anaconda Prompt 中，从仓库根目录依次运行：

```bat
conda create -n malware-swin python=3.11 -y
conda activate malware-swin
cd training
python -m pip install -r requirements.txt
python prepare_data.py
python train.py --data-dir ../data/final/data --out runs/smoke --epochs 2 --device cpu
```

再次运行请换一个输出目录，例如 `runs/smoke2`；程序不会覆盖已有实验。
图表、指标和 best.pt 均在 `training/runs/smoke/` 中。此入口为命令行训练和预测，尚未接入前后端。

## 运行

终端进入本README所在目录，替换下方数据路径为实际路径。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --data-dir ../data/final/data --out runs/smoke --epochs 2 --device cpu
```

正式基线（原Swin1的32/64/128通道，2/2/2层，MLP比例4）：

```powershell
python train.py --data-dir ../data/final/data --out runs/base --epochs 30 --embed-dim 32 --depths 2 2 2 --heads 2 4 8 --mlp-ratio 4 --skip-test
```

先只减通道，控制变量：

```powershell
python train.py --data-dir ../data/final/data --out runs/slim24 --epochs 30 --embed-dim 24 --depths 2 2 2 --heads 2 4 8 --mlp-ratio 4 --skip-test
python compare.py --runs runs/base runs/slim24 --out runs/comparison
```

进一步实验可将 --mlp-ratio 改为3或2，再尝试 --depths 2 2 1。
总嵌入维度不变时，单纯减少头数通常不会显著减少参数。每阶段通道须能被头数整除。
本版仍是从头训练，不是训练后剪枝，也不含蒸馏或量化。
仅按验证集结果选架构，选定后固定配置再运行一次最终实验（不带--skip-test）。
推荐同时看Macro-F1，并用多个随机种子复核；不能预先保证压缩后准确率不降。
默认自动选择CUDA/CPU，Windows默认--workers 0；显存不足降低--batch-size。
输出目录必须不存在，以防覆盖旧实验。

预测示例：

```powershell
python predict.py --checkpoint runs/smoke/best.pt --input sample.png --kind png
python predict.py --checkpoint runs/smoke/best.pt --input sample.bytes --kind bytes
```

PNG必须来自同一预处理规则；不能对任意图片做有意义分类。.bytes必须是BIG2015文本格式。
类别为Ramnit、Lollipop、Kelihos_ver3、Vundo、Simda、Tracur、Kelihos_ver1、Obfuscator.ACY、Gatak。
该数据只有恶意软件，输出是9类家族标签；没有良性/未知家族检测能力。
Top-5是未经校准的softmax得分，不是“恶意概率”。

## 指标与图表口径

- 每轮优化只访问训练集。随后在eval模式下，用当轮固定权重分别评估训练/验证集。
  这样训练准确率不会混入Dropout噪声或一个epoch内变化的权重。
- 训练优化采用训练集类别权重及label smoothing；图表中的训练/验证loss统一采用
  无类别权重、无平滑的平均交叉熵，便于比较。
- 依据验证集Macro-F1保存best.pt。最后重新加载best.pt计算训练、验证和测试准确率，
  不将最后一轮模型和最佳模型结果混用。
- 测试准确率只出现在最终柱状图和metrics.json，不画成每轮测试曲线，
  避免把测试结果作为调参信号。--skip-test时不推理测试集；仍会检查主表与图片完整性。
- dataset_counts.png：各家族、各划分的唯一实际样本数量。
- learning_curves.png：准确率、loss、每轮耗时以及累计训练处理次数。
  562条训练数据运行30轮产生16860次训练样本呈现，仍只有562个不同训练样本。
- final_metrics.png：最佳检查点的训练/验证/测试准确率、Macro-F1、评估耗时与吞吐量。
- test_confusion_matrix.png：测试集混淆矩阵（只有最终测试启用时生成）。
- 每轮耗时分为优化、训练集评估、验证集评估；均包括数据加载和设备传输。
  CUDA计时前后同步。累计运行时间另含数据核验、保存、画图、最终评估和延迟基准，
  不含解释器导入与模型初始化；各项口径在JSON中明确。
- inference_benchmark：在设备上的batch=1纯前向，5次预热后测30次，报告均值/P50/P95；
  不包含读文件、预处理或传输。预测接口另记录冷启动前向与预处理耗时，不含模型加载。
- compare.py按验证准确率比较，检查数据、种子、训练预算和设备等条件；
  同名不同路径实验宜用不同目录名以便图表区分。计时仍受系统负载影响。

## 实跑与限制

此前审核包中的 example_run 使用803张现有图片、固定Swin1默认结构完成两轮CPU集成验证，
用于确认训练、反向传播、保存、重载、评估、图表均能工作；两轮结果不代表收敛精度。
本目录只提交源码与验证说明（VERIFICATION.md）；运行后会在 --out 目录生成自己的日志和图表。
包中不提供试运行权重：训练后自动生成best.pt。
原始.bytes没有完整提交，因此这里只做合成.bytes与原图像聚合规则的一致性测试，
真实原始样本→预测的全流程还需要你提供数据后验证。
