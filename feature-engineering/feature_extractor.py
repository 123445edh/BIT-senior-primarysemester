"""US-05 基础特征提取模块。

从木马样本的 ``.bytes`` 文本表示中提取三类特征：

1. 字节序列 ``byte_seq``：原始字节按顺序组成，未知字节 ``??`` 映射为配置值，
   并按 ``max_length`` 截断/填充；
2. 信息熵 ``entropy``：基于已知字节分布计算 Shannon 熵；
3. N-gram 频率 ``ngram_features``：统计连续 N 个字节片段的出现频率。

本模块只依赖 NumPy 与 Python 标准库，便于在本地环境直接运行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MAX_LENGTH = 1024
DEFAULT_PAD_VALUE = 0
DEFAULT_UNKNOWN_VALUE = 0
SUPPORTED_NGRAMS = (1, 2, 3)


def _is_hex_token(token: bytes) -> bool:
    """判断 token 是否为两位十六进制字节（如 ``4D``）。"""
    if len(token) != 2:
        return False
    try:
        int(token, 16)
    except ValueError:
        return False
    return True


def _is_hex_address(token: bytes) -> bool:
    """判断 token 是否为 8 位十六进制地址（如 ``00401000``）。"""
    if len(token) != 8:
        return False
    try:
        int(token, 16)
    except ValueError:
        return False
    return True


def parse_bytes_file(
    path: str | Path,
    unknown_value: int = DEFAULT_UNKNOWN_VALUE,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """读取一个 ``.bytes`` 文件并解析为字节序列。

    返回：
        values: 完整字节序列（uint8），未知字节用 ``unknown_value`` 填充；
        known_mask: 与 ``values`` 等长的布尔数组，True 表示该位置是已知字节；
        total_tokens: 总 token 数；
        known_tokens: 已知字节数；
        unknown_tokens: 未知字节数；
        invalid_tokens: 无法解析的 token 数（按未知字节处理）。
    """
    values = bytearray()
    known = bytearray()
    total_tokens = 0
    known_tokens = 0
    unknown_tokens = 0
    invalid_tokens = 0

    with open(path, "rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue

            # Microsoft BIG 2015 的 .bytes 文件每行以 8 位十六进制地址开头。
            if _is_hex_address(parts[0]):
                tokens = parts[1:]
            else:
                tokens = parts

            for token in tokens:
                if token == b"??":
                    values.append(unknown_value)
                    known.append(0)
                    total_tokens += 1
                    unknown_tokens += 1
                elif _is_hex_token(token):
                    values.append(int(token, 16))
                    known.append(1)
                    total_tokens += 1
                    known_tokens += 1
                else:
                    values.append(unknown_value)
                    known.append(0)
                    total_tokens += 1
                    invalid_tokens += 1

    values_array = np.asarray(values, dtype=np.uint8)
    known_array = np.asarray(known, dtype=bool)
    return (
        values_array,
        known_array,
        total_tokens,
        known_tokens,
        unknown_tokens,
        invalid_tokens,
    )


def make_byte_sequence(
    values: np.ndarray,
    max_length: int = DEFAULT_MAX_LENGTH,
    pad_value: int = DEFAULT_PAD_VALUE,
    truncate: str = "keep_head",
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """将字节序列截断/填充为固定长度，并生成 attention mask。

    返回：
        sequence: 固定长度 uint8 数组；
        mask: 固定长度 uint8 数组，1 表示真实字节，0 表示填充；
        original_length: 原始字节长度；
        truncated: 被截断的字节数；
        padded: 填充的字节数。
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if truncate not in ("keep_head", "keep_tail"):
        raise ValueError("truncate must be 'keep_head' or 'keep_tail'")
    if not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be in [0, 255]")

    values = np.asarray(values, dtype=np.uint8)
    original_length = int(values.size)
    mask = np.zeros(max_length, dtype=np.uint8)

    if original_length >= max_length:
        if truncate == "keep_head":
            sequence = values[:max_length].copy()
        else:
            sequence = values[-max_length:].copy()
        mask[:] = 1
        truncated = original_length - max_length
        padded = 0
    else:
        padded = max_length - original_length
        if truncate == "keep_head":
            sequence = np.pad(values, (0, padded), constant_values=pad_value)
            mask[:original_length] = 1
        else:
            sequence = np.pad(values, (padded, 0), constant_values=pad_value)
            mask[padded:] = 1
        truncated = 0

    return sequence.astype(np.uint8), mask.astype(np.uint8), original_length, truncated, padded


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """根据字节计数计算 Shannon 熵（单位 bit，范围 0~8）。"""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def count_ngrams(values: np.ndarray, known_mask: np.ndarray, n: int) -> np.ndarray:
    """统计全样本 N-gram 计数，返回长度为 ``256**n`` 的 int64 数组。

    只统计窗口内全部为已知字节的 N-gram，未知字节处不跨窗口统计。
    """
    if n not in SUPPORTED_NGRAMS:
        raise ValueError(f"unsupported n-gram size: {n}")
    values = np.asarray(values, dtype=np.uint8)
    known_mask = np.asarray(known_mask, dtype=bool)

    if n == 1:
        return np.bincount(values[known_mask], minlength=256).astype(np.int64)

    if n == 2:
        valid = known_mask[:-1] & known_mask[1:]
        if not valid.any():
            return np.zeros(65536, dtype=np.int64)
        indices = (values[:-1].astype(np.int64) << 8) | values[1:].astype(np.int64)
        return np.bincount(indices[valid], minlength=65536).astype(np.int64)

    valid = known_mask[:-2] & known_mask[1:-1] & known_mask[2:]
    if not valid.any():
        return np.zeros(16777216, dtype=np.int64)
    indices = (
        (values[:-2].astype(np.int64) << 16)
        | (values[1:-1].astype(np.int64) << 8)
        | values[2:].astype(np.int64)
    )
    return np.bincount(indices[valid], minlength=16777216).astype(np.int64)


def count_selected_ngrams(
    values: np.ndarray,
    known_mask: np.ndarray,
    n: int,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """统计指定 N-gram 的每样本计数，并返回有效 N-gram 总数。"""
    if n not in SUPPORTED_NGRAMS:
        raise ValueError(f"unsupported n-gram size: {n}")
    values = np.asarray(values, dtype=np.uint8)
    known_mask = np.asarray(known_mask, dtype=bool)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selected_indices = np.sort(selected_indices)
    size = int(selected_indices.size)

    if n == 1:
        full_counts = np.bincount(values[known_mask], minlength=256)
        total = int(known_mask.sum())
        return full_counts[selected_indices].astype(np.int64), total

    if n == 2:
        valid = known_mask[:-1] & known_mask[1:]
        total = int(valid.sum())
        if total == 0 or size == 0:
            return np.zeros(size, dtype=np.int64), total
        indices = (values[:-1].astype(np.int64) << 8) | values[1:].astype(np.int64)
        valid_indices = indices[valid]
    else:
        valid = known_mask[:-2] & known_mask[1:-1] & known_mask[2:]
        total = int(valid.sum())
        if total == 0 or size == 0:
            return np.zeros(size, dtype=np.int64), total
        indices = (
            (values[:-2].astype(np.int64) << 16)
            | (values[1:-1].astype(np.int64) << 8)
            | values[2:].astype(np.int64)
        )
        valid_indices = indices[valid]

    positions = np.searchsorted(selected_indices, valid_indices)
    positions = np.minimum(positions, size - 1)
    hits = selected_indices[positions] == valid_indices
    counts = np.bincount(positions[hits], minlength=size)
    return counts.astype(np.int64), total


def select_top_k(global_counts: np.ndarray, k: int) -> np.ndarray:
    """按全局计数降序选择前 k 个 N-gram 索引，计数相同时按索引升序打破平局。"""
    global_counts = np.asarray(global_counts)
    if k <= 0 or k >= global_counts.size:
        k = global_counts.size
    if k == global_counts.size:
        return np.arange(global_counts.size, dtype=np.int64)

    partitioned = np.argpartition(global_counts, -k)[-k:]
    order = np.lexsort((partitioned, -global_counts[partitioned]))
    return partitioned[order].astype(np.int64)


def _gram_hex(index: int, n: int) -> str:
    """将整数 N-gram 索引转换为大写十六进制表示。"""
    raw = index.to_bytes(n, "big")
    return raw.hex().upper()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV 文件（支持 utf-8-sig），返回按原始列名映射的行。"""
    import csv

    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return rows


def load_sample_records(
    bytes_dir: str | Path,
    labels_csv: str | Path,
    manifest_csv: str | Path | None,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    """加载待提取样本，优先对齐预处理清单中的 Id/Class/split。"""
    bytes_dir = Path(bytes_dir)
    labels_csv = Path(labels_csv)
    manifest_csv = Path(manifest_csv) if manifest_csv else None

    label_lookup: dict[str, int] = {}
    for row in read_csv_rows(labels_csv):
        label_lookup[row["Id"]] = int(row["Class"])

    records: list[dict[str, Any]] = []
    if manifest_csv is not None and manifest_csv.is_file():
        for row in read_csv_rows(manifest_csv):
            sample_id = row["Id"]
            if sample_id in label_lookup:
                records.append(
                    {
                        "Id": sample_id,
                        "Class": int(row["Class"]),
                        "split": row.get("split", "unknown"),
                    }
                )

    if not records:
        for sample_id in sorted(label_lookup):
            records.append(
                {
                    "Id": sample_id,
                    "Class": label_lookup[sample_id],
                    "split": "unknown",
                }
            )

    records = [r for r in records if (bytes_dir / f"{r['Id']}.bytes").is_file()]
    records.sort(key=lambda r: r["Id"])
    if max_samples and max_samples > 0:
        records = records[:max_samples]
    return records


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    """将配置中的相对路径解析为绝对路径。"""
    candidate = Path(str(path).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _resolve_config_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    resolved["input"]["bytes_dir"] = str(resolve_path(resolved["input"]["bytes_dir"], base_dir))
    resolved["input"]["labels_csv"] = str(resolve_path(resolved["input"]["labels_csv"], base_dir))
    if resolved["input"].get("manifest_csv"):
        resolved["input"]["manifest_csv"] = str(
            resolve_path(resolved["input"]["manifest_csv"], base_dir)
        )
    resolved["output_dir"] = str(resolve_path(resolved["output_dir"], base_dir))
    return resolved


def resolve_top_k(top_k: Any, n: int, max_possible: int) -> int:
    """将配置中的 top_k 解析为不超过最大可能维度的整数。"""
    if isinstance(top_k, dict):
        value = top_k.get(str(n), top_k.get(n, 1024))
    else:
        value = top_k
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 1024
    if value <= 0 or value >= max_possible:
        return max_possible
    return min(value, max_possible)


@dataclass
class FeatureBatch:
    """一批样本的特征提取结果。"""

    ids: list[str]
    classes: np.ndarray
    splits: list[str]
    byte_seq: np.ndarray
    byte_seq_mask: np.ndarray
    byte_lengths: np.ndarray
    truncated: np.ndarray
    padded: np.ndarray
    known_counts: np.ndarray
    unknown_counts: np.ndarray
    unknown_ratios: np.ndarray
    entropy: np.ndarray
    ngram_features: dict[int, np.ndarray]
    ngram_vocab: dict[int, list[dict[str, Any]]]
    config: dict[str, Any]
    elapsed_seconds: float = 0.0


def build_features(config: dict[str, Any], base_dir: Path | None = None) -> FeatureBatch:
    """按配置批量提取特征，返回 :class:`FeatureBatch`。"""
    import time

    start = time.perf_counter()
    config = _resolve_config_paths(config, base_dir or Path.cwd())
    bytes_dir = Path(config["input"]["bytes_dir"])
    labels_csv = Path(config["input"]["labels_csv"])
    manifest_csv = (
        Path(config["input"]["manifest_csv"])
        if config["input"].get("manifest_csv")
        else None
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(config.get("max_samples", 0))
    records = load_sample_records(bytes_dir, labels_csv, manifest_csv, max_samples)
    if not records:
        raise ValueError(f"No matching .bytes files found under {bytes_dir}")

    byte_config = config["byte_sequence"]
    max_length = int(byte_config.get("max_length", DEFAULT_MAX_LENGTH))
    pad_value = int(byte_config.get("pad_value", DEFAULT_PAD_VALUE))
    unknown_value = int(byte_config.get("unknown_value", DEFAULT_UNKNOWN_VALUE))
    truncate = byte_config.get("truncate", "keep_head")
    entropy_enabled = bool(config.get("entropy", {}).get("enabled", True))

    ngram_config = config["ngram"]
    ngram_ns = [int(n) for n in ngram_config.get("ns", [1, 2, 3])]
    for n in ngram_ns:
        if n not in SUPPORTED_NGRAMS:
            raise ValueError(f"unsupported n-gram size: {n}")
    top_k_map = ngram_config.get("top_k", 1024)
    normalize = ngram_config.get("normalize", "frequency")
    if normalize not in ("frequency", "count"):
        raise ValueError("normalize must be 'frequency' or 'count'")

    sample_count = len(records)
    batch = FeatureBatch(
        ids=[record["Id"] for record in records],
        classes=np.asarray([record["Class"] for record in records], dtype=np.int64),
        splits=[record["split"] for record in records],
        byte_seq=np.zeros((sample_count, max_length), dtype=np.uint8),
        byte_seq_mask=np.zeros((sample_count, max_length), dtype=np.uint8),
        byte_lengths=np.zeros(sample_count, dtype=np.int64),
        truncated=np.zeros(sample_count, dtype=np.int64),
        padded=np.zeros(sample_count, dtype=np.int64),
        known_counts=np.zeros(sample_count, dtype=np.int64),
        unknown_counts=np.zeros(sample_count, dtype=np.int64),
        unknown_ratios=np.zeros(sample_count, dtype=np.float64),
        entropy=np.zeros((sample_count, 1), dtype=np.float64),
        ngram_features={},
        ngram_vocab={},
        config=config,
    )

    global_counts: dict[int, np.ndarray] = {}
    for n in ngram_ns:
        global_counts[n] = np.zeros(256**n, dtype=np.int64)

    print(f"Pass 1/2: parsing {sample_count} samples ...", flush=True)
    for index, record in enumerate(records):
        path = bytes_dir / f"{record['Id']}.bytes"
        values, known_mask, _, known_tokens, unknown_tokens, _ = parse_bytes_file(
            path, unknown_value=unknown_value
        )
        sequence, mask, original_length, truncated, padded = make_byte_sequence(
            values, max_length=max_length, pad_value=pad_value, truncate=truncate
        )
        batch.byte_seq[index] = sequence
        batch.byte_seq_mask[index] = mask
        batch.byte_lengths[index] = original_length
        batch.truncated[index] = truncated
        batch.padded[index] = padded
        batch.known_counts[index] = known_tokens
        batch.unknown_counts[index] = unknown_tokens
        batch.unknown_ratios[index] = (
            unknown_tokens / original_length if original_length else 0.0
        )
        if entropy_enabled:
            known_counts = np.bincount(values[known_mask], minlength=256)
            batch.entropy[index, 0] = shannon_entropy_from_counts(known_counts)

        for n in ngram_ns:
            global_counts[n] += count_ngrams(values, known_mask, n)

        if (index + 1) % 25 == 0 or index + 1 == sample_count:
            print(f"  {index + 1}/{sample_count}", flush=True)

    print("Pass 2/2: building N-gram feature vectors ...", flush=True)
    for n in ngram_ns:
        k = resolve_top_k(top_k_map, n, 256**n)
        selected = select_top_k(global_counts[n], k)
        vocab: list[dict[str, Any]] = []
        for rank, gram_index in enumerate(selected, start=1):
            vocab.append(
                {
                    "rank": rank,
                    "gram": _gram_hex(int(gram_index), n),
                    "byte_value": int(gram_index) if n == 1 else None,
                    "global_count": int(global_counts[n][gram_index]),
                }
            )
        batch.ngram_vocab[n] = vocab
        batch.ngram_features[n] = np.zeros((sample_count, len(selected)), dtype=np.float32)

    for index, record in enumerate(records):
        path = bytes_dir / f"{record['Id']}.bytes"
        values, known_mask, _, _, _, _ = parse_bytes_file(path, unknown_value=unknown_value)
        for n in ngram_ns:
            selected = np.asarray(
                [item["byte_value"] if n == 1 else int(item["gram"], 16) for item in batch.ngram_vocab[n]],
                dtype=np.int64,
            )
            counts, total = count_selected_ngrams(values, known_mask, n, selected)
            if normalize == "frequency":
                if total > 0:
                    batch.ngram_features[n][index] = (counts / total).astype(np.float32)
                else:
                    batch.ngram_features[n][index] = 0.0
            else:
                batch.ngram_features[n][index] = counts.astype(np.float32)

        if (index + 1) % 25 == 0 or index + 1 == sample_count:
            print(f"  {index + 1}/{sample_count}", flush=True)

    batch.elapsed_seconds = time.perf_counter() - start
    return batch


def save_outputs(batch: FeatureBatch, output_dir: str | Path) -> dict[str, Any]:
    """将特征提取结果写入磁盘，返回摘要信息。"""
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "byte_seq.npy", batch.byte_seq)
    np.save(output_dir / "byte_seq_mask.npy", batch.byte_seq_mask)
    np.save(output_dir / "byte_lengths.npy", batch.byte_lengths)
    np.save(output_dir / "entropy.npy", batch.entropy)
    np.save(output_dir / "known_byte_count.npy", batch.known_counts)
    np.save(output_dir / "unknown_byte_count.npy", batch.unknown_counts)
    np.save(output_dir / "unknown_byte_ratio.npy", batch.unknown_ratios)

    sample_path = output_dir / "samples.csv"
    with open(sample_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Id",
                "Class",
                "split",
                "byte_length",
                "truncated",
                "padded",
                "known_byte_count",
                "unknown_byte_count",
                "unknown_byte_ratio",
                "entropy",
            ]
        )
        for index, sample_id in enumerate(batch.ids):
            writer.writerow(
                [
                    sample_id,
                    int(batch.classes[index]),
                    batch.splits[index],
                    int(batch.byte_lengths[index]),
                    int(batch.truncated[index]),
                    int(batch.padded[index]),
                    int(batch.known_counts[index]),
                    int(batch.unknown_counts[index]),
                    float(batch.unknown_ratios[index]),
                    float(batch.entropy[index, 0]),
                ]
            )

    for n, features in batch.ngram_features.items():
        np.save(output_dir / f"ngram_{n}_features.npy", features)
        vocab_path = output_dir / f"ngram_{n}_vocab.csv"
        with open(vocab_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            if n == 1:
                writer.writerow(["rank", "gram", "byte_value", "global_count"])
            else:
                writer.writerow(["rank", "gram", "global_count"])
            for item in batch.ngram_vocab[n]:
                if n == 1:
                    writer.writerow(
                        [item["rank"], item["gram"], item["byte_value"], item["global_count"]]
                    )
                else:
                    writer.writerow([item["rank"], item["gram"], item["global_count"]])

    byte_config = batch.config["byte_sequence"]
    summary: dict[str, Any] = {
        "sample_count": len(batch.ids),
        "ids": batch.ids,
        "byte_sequence": {
            "max_length": int(byte_config.get("max_length", DEFAULT_MAX_LENGTH)),
            "pad_value": int(byte_config.get("pad_value", DEFAULT_PAD_VALUE)),
            "truncate": byte_config.get("truncate", "keep_head"),
            "shape": list(batch.byte_seq.shape),
        },
        "entropy": {
            "shape": list(batch.entropy.shape),
        },
        "ngram": {
            str(n): {
                "top_k": int(batch.ngram_features[n].shape[1]),
                "shape": list(batch.ngram_features[n].shape),
                "vocab_file": f"ngram_{n}_vocab.csv",
                "feature_file": f"ngram_{n}_features.npy",
            }
            for n in batch.ngram_features
        },
        "outputs": {
            "samples_csv": "samples.csv",
            "byte_seq_npy": "byte_seq.npy",
            "byte_seq_mask_npy": "byte_seq_mask.npy",
            "entropy_npy": "entropy.npy",
        },
        "elapsed_seconds": batch.elapsed_seconds,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
