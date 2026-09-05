"""US-06 特征融合模块。

在 US-05 基础特征提取的基础上，将字节序列、信息熵和 N-gram 统计特征融合成
固定维度的 ``fused_vector``，供轻量级 Transformer 模型直接读取。

本模块支持两种使用方式：

1. 作为库导入，调用 :func:`fuse_feature_groups` 对任意 NumPy 数组进行融合；
2. 作为命令行工具，读取 ``../特征提取/outputs`` 下已生成的特征文件并落盘。

只依赖 NumPy 与 Python 标准库。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG: dict[str, Any] = {
    "feature_groups": {
        "byte_seq": {
            "enabled": True,
            "normalize": "minmax",
            "scale": 255.0,
        },
        "entropy": {
            "enabled": True,
            "normalize": "minmax",
            "expected_min": 0.0,
            "expected_max": 8.0,
        },
        "ngram": {
            "enabled": True,
            "ns": [1, 2],
            "normalize": "none",
        },
    },
    "adaptive_weighting": {
        "method": "variance_softmax",
        "temperature": 1.0,
        "eps": 1e-6,
    },
}


@dataclass
class FusedFeatures:
    """特征融合结果。"""

    fused_vector: np.ndarray
    feature_groups: dict[str, np.ndarray]
    slices: dict[str, tuple[int, int]]
    weights: dict[str, float]
    config: dict[str, Any]


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    """把输入转换为 (N, D) 形状的二维数组。"""
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return array.astype(np.float32, copy=False)


def normalize_group(array: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    """按配置对单个特征组做归一化。"""
    method = options.get("normalize", "none")
    if method == "none":
        return array

    if method == "minmax":
        lower = options.get("expected_min")
        upper = options.get("expected_max")
        if lower is not None and upper is not None:
            denominator = float(upper) - float(lower)
            if denominator <= 0:
                raise ValueError("expected_max must be greater than expected_min")
            return np.clip((array - float(lower)) / denominator, 0.0, 1.0)

        data_min = array.min(axis=0, keepdims=True)
        data_max = array.max(axis=0, keepdims=True)
        denominator = data_max - data_min
        denominator[denominator == 0] = 1.0
        return (array - data_min) / denominator

    if method == "standard":
        mean = array.mean(axis=0, keepdims=True)
        std = array.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (array - mean) / std

    if method == "scale":
        scale = float(options.get("scale", 1.0))
        if scale == 0:
            raise ValueError("scale must be non-zero")
        return array / scale

    raise ValueError(f"unsupported normalize method: {method}")


def _group_variance_score(array: np.ndarray, eps: float) -> float:
    """计算特征组的方差得分，用于自适应加权。"""
    centered = array - array.mean(axis=0, keepdims=True)
    variance = float(np.mean(np.square(centered), dtype=np.float64))
    return float(np.log1p(max(variance, 0.0)) + eps)


def adaptive_weights(
    arrays: dict[str, np.ndarray],
    options: dict[str, Any],
) -> dict[str, float]:
    """根据特征组方差计算归一化权重。"""
    method = options.get("method", "variance_softmax")
    eps = float(options.get("eps", 1e-6))
    temperature = float(options.get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = {name: _group_variance_score(array, eps) for name, array in arrays.items()}

    if method == "uniform":
        return {name: 1.0 / len(scores) for name in scores}

    if method == "variance_softmax":
        raw = np.asarray([score for score in scores.values()], dtype=np.float64)
        shifted = raw - raw.max()
        exp_values = np.exp(shifted / temperature)
        normalized = exp_values / exp_values.sum()
        return {name: float(weight) for name, weight in zip(scores.keys(), normalized)}

    if method == "variance_proportional":
        total = float(sum(scores.values()))
        if total <= 0:
            return {name: 1.0 / len(scores) for name in scores}
        return {name: score / total for name, score in scores.items()}

    raise ValueError(f"unsupported adaptive_weighting method: {method}")


def fuse_feature_groups(
    byte_seq: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    ngram_features: dict[int, np.ndarray] | None = None,
    config: dict[str, Any] | None = None,
) -> FusedFeatures:
    """融合字节序列、信息熵与 N-gram 特征。

    参数：
        byte_seq: (N, max_length) 字节序列；
        entropy: (N, 1) 信息熵；
        ngram_features: {n: (N, top_k_n)} N-gram 频率特征；
        config: 融合配置，默认使用 :data:`DEFAULT_CONFIG`。
    """
    config = config or DEFAULT_CONFIG
    groups_config = config.get("feature_groups", DEFAULT_CONFIG["feature_groups"])
    weighting_config = config.get(
        "adaptive_weighting", DEFAULT_CONFIG["adaptive_weighting"]
    )

    arrays: dict[str, np.ndarray] = {}
    sample_count: int | None = None

    def register(name: str, array: np.ndarray) -> None:
        nonlocal sample_count
        processed = _as_2d(array, name)
        if sample_count is None:
            sample_count = processed.shape[0]
        elif processed.shape[0] != sample_count:
            raise ValueError(
                f"feature group {name} has {processed.shape[0]} samples, "
                f"expected {sample_count}"
            )
        arrays[name] = processed

    byte_config = groups_config.get("byte_seq", {})
    if byte_config.get("enabled", True):
        if byte_seq is None:
            raise ValueError("byte_seq is required when byte_seq.enabled is true")
        register("byte_seq", normalize_group(byte_seq, byte_config))

    entropy_config = groups_config.get("entropy", {})
    if entropy_config.get("enabled", True):
        if entropy is None:
            raise ValueError("entropy is required when entropy.enabled is true")
        register("entropy", normalize_group(entropy, entropy_config))

    ngram_config = groups_config.get("ngram", {})
    if ngram_config.get("enabled", True):
        if not ngram_features:
            raise ValueError("ngram_features is required when ngram.enabled is true")
        requested_ns = [int(n) for n in ngram_config.get("ns", sorted(ngram_features))]
        missing = [n for n in requested_ns if n not in ngram_features]
        if missing:
            raise ValueError(f"missing ngram_features for n={missing}")
        concatenated = np.concatenate(
            [np.asarray(ngram_features[n], dtype=np.float32) for n in requested_ns],
            axis=1,
        )
        register("ngram", normalize_group(concatenated, ngram_config))

    if not arrays:
        raise ValueError("at least one feature group must be enabled")

    weights = adaptive_weights(arrays, weighting_config)
    fused_vector = np.concatenate(
        [arrays[name] * float(weights[name]) for name in arrays],
        axis=1,
    ).astype(np.float32)

    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, array in arrays.items():
        width = int(array.shape[1])
        slices[name] = (offset, offset + width)
        offset += width

    return FusedFeatures(
        fused_vector=fused_vector,
        feature_groups={name: arrays[name].copy() for name in arrays},
        slices=slices,
        weights=weights,
        config=config,
    )


def load_feature_outputs(
    outputs_dir: str | Path,
    ngram_ns: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """从 US-05 输出目录读取字节序列、熵与 N-gram 特征。"""
    outputs_dir = Path(outputs_dir)
    byte_seq = np.load(outputs_dir / "byte_seq.npy")
    entropy = np.load(outputs_dir / "entropy.npy")
    summary_path = outputs_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if ngram_ns is None:
        ngram_ns = sorted(int(key) for key in summary.get("ngram", {}).keys())
    if not ngram_ns:
        ngram_ns = [1, 2]

    ngram_features: dict[int, np.ndarray] = {}
    for n in ngram_ns:
        path = outputs_dir / f"ngram_{n}_features.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        ngram_features[n] = np.load(path)

    return byte_seq, entropy, ngram_features, summary


def save_fusion_outputs(
    fused: FusedFeatures,
    output_dir: str | Path,
) -> dict[str, Any]:
    """保存融合向量、权重和摘要。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "fused_vector.npy", fused.fused_vector)
    with open(output_dir / "fusion_weights.json", "w", encoding="utf-8") as handle:
        json.dump(fused.weights, handle, ensure_ascii=False, indent=2)

    summary: dict[str, Any] = {
        "sample_count": int(fused.fused_vector.shape[0]),
        "fused_dim": int(fused.fused_vector.shape[1]),
        "feature_groups": {
            name: {
                "shape": list(array.shape),
                "slice": list(fused.slices[name]),
            }
            for name, array in fused.feature_groups.items()
        },
        "weights": fused.weights,
        "config": fused.config,
    }
    with open(output_dir / "fusion_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US-06 木马样本特征融合")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("outputs"),
        help="US-05 特征输出目录",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).with_name("outputs"),
        help="融合结果输出目录",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("fusion_config.json"),
        help="融合配置文件路径",
    )
    parser.add_argument("--ngram-ns", default=None, help="逗号分隔的 N-gram 阶数")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ngram_ns = None
    if args.ngram_ns:
        ngram_ns = [int(item.strip()) for item in args.ngram_ns.split(",") if item.strip()]
        config.setdefault("feature_groups", {}).setdefault("ngram", {})["ns"] = ngram_ns

    byte_seq, entropy, ngram_features, summary = load_feature_outputs(
        args.outputs_dir, ngram_ns=ngram_ns
    )
    fused = fuse_feature_groups(
        byte_seq=byte_seq,
        entropy=entropy,
        ngram_features=ngram_features,
        config=config,
    )
    save_fusion_outputs(fused, args.out_dir)
    print(f"Loaded {summary.get('sample_count', fused.fused_vector.shape[0])} samples.")
    print(f"Fused vector shape: {fused.fused_vector.shape}")
    print(f"Fusion weights: {json.dumps(fused.weights, ensure_ascii=False)}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
