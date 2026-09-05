# 特征提取与特征融合模块（US-05 + US-06）

> 基于轻量级 Transformer 的木马家族分类系统 · 周丁楠负责部分

本目录是周丁楠负责的特征工程成果，包含两个部分：

- **US-05 基础特征提取**：从木马样本中提取字节序列、信息熵和 N-gram 频率特征；
- **US-06 特征融合**：把多类特征归一化、自适应加权并拼接为统一的 `fused_vector`。

## 目录结构

```text
.
├── feature_extractor.py
├── run.py
├── config.json
├── fusion.py
├── fusion_config.json
├── requirements.txt
├── .gitignore
├── tests
│   ├── test_feature_extractor.py
│   └── test_fusion.py
└── outputs
    ├── byte_seq.npy
    ├── entropy.npy
    ├── ngram_1_features.npy
    ├── ngram_2_features.npy
    ├── samples.csv
    ├── fused_vector.npy
    ├── fusion_weights.json
    └── fusion_summary.json
```

## 环境要求

- Python 3.10+
- NumPy 1.24+

无需其他第三方依赖。

## US-05：特征提取

### 功能

`feature_extractor.py` 从 `.bytes` 文件中提取：

| 特征 | 形状 | 说明 |
| --- | --- | --- |
| `byte_seq` | `[样本数, 1024]` | 截断/填充后的原始字节序列 |
| `entropy` | `[样本数, 1]` | 已知字节的 Shannon 信息熵 |
| `ngram_features` | `[样本数, top_k_n]` | 1-gram、2-gram 频率特征 |

同时输出 `byte_seq_mask`、样本清单 `samples.csv` 和特征摘要。

### 运行

```bash
python run.py
```

本地数据路径写在 `config.json` 中：

```json
{
  "input": {
    "bytes_dir": "../../数据集/kaggle2015-sample/subtrain",
    "labels_csv": "../../数据集/kaggle2015-sample/subtrainLabels.csv",
    "manifest_csv": "../../数据集/data/final/data/dataset.csv"
  },
  "output_dir": "outputs"
}
```

上传到 GitHub 后，如果数据不在相同相对路径下，请通过命令行参数覆盖：

```bash
python run.py --bytes-dir <样本目录> --labels <标签文件> --manifest <清单文件>
```

## US-06：特征融合

### 功能

`fusion.py` 接收 US-05 生成的特征，完成：

- 归一化：支持 `minmax`、`standard`、`scale`、`none`
- 自适应加权：支持 `variance_softmax`、`variance_proportional`、`uniform`
- 特征拼接：输出固定维度 `fused_vector`

### 运行

```bash
python fusion.py --outputs-dir outputs --out-dir outputs
```

也可以直接运行：

```bash
python fusion.py
```

`--outputs-dir` 中需要包含：

- `byte_seq.npy`
- `entropy.npy`
- `ngram_{n}_features.npy`

### 作为库使用

```python
from fusion import fuse_feature_groups

result = fuse_feature_groups(
    byte_seq=byte_seq_array,
    entropy=entropy_array,
    ngram_features={1: ngram_1, 2: ngram_2},
)

print(result.fused_vector.shape)
print(result.weights)
print(result.slices)
```

### 融合配置

`fusion_config.json` 关键配置：

```json
{
  "feature_groups": {
    "byte_seq": { "enabled": true, "normalize": "minmax", "scale": 255.0 },
    "entropy": { "enabled": true, "normalize": "minmax" },
    "ngram": { "enabled": true, "ns": [1, 2], "normalize": "none" }
  },
  "adaptive_weighting": {
    "method": "variance_softmax",
    "temperature": 1.0,
    "eps": 1e-6
  }
}
```

默认采用 `variance_softmax`，方差越大的特征组权重越高，可作为 US-07 特征选择的基础。

## 输出结果

`outputs/` 中包含：

### 特征提取输出

- `byte_seq.npy`
- `byte_seq_mask.npy`
- `entropy.npy`
- `ngram_1_features.npy`
- `ngram_2_features.npy`
- `ngram_1_vocab.csv`
- `ngram_2_vocab.csv`
- `samples.csv`
- `summary.json`

### 特征融合输出

- `fused_vector.npy`
- `fusion_weights.json`
- `fusion_summary.json`

## 测试

运行全部单元测试：

```bash
python -m unittest discover -s tests -v
```

## 集成说明

- `fused_vector.npy` 的行顺序需要与 `samples.csv` 保持一致。
- 与模型和后端联调前，需要冻结融合后的维度。
- 提交 GitHub 时，建议忽略 `__pycache__/` 和临时输出目录。
