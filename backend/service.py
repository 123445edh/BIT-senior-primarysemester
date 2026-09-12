import io
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 把仓库根目录加入路径，让 Python 能找到 training 包
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.data import FAMILIES, PREPROCESS
from training.models import SwinClassifier

# 模型配置（与 training/README.md 中的基线一致）
MODEL_CONFIG = {
    "num_classes": 9,
    "embed_dim": 32,
    "depths": (2, 2, 2),
    "num_heads": (2, 4, 8),
    "dropout": 0.0,
    "drop_path": 0.0,
    "mlp_ratio": 4.0,
}

_CHECKPOINT_CACHE = None
_MODEL_CACHE = None


def _load_checkpoint(checkpoint_path: str | Path):
    """加载模型权重（全局缓存，只加载一次）"""
    global _CHECKPOINT_CACHE
    if _CHECKPOINT_CACHE is None:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        # 校验类别与预处理协议
        if checkpoint["families"] != FAMILIES:
            raise ValueError("模型类别与当前代码不一致")
        if checkpoint["preprocess"] != PREPROCESS:
            raise ValueError("模型预处理协议与当前代码不一致")
        _CHECKPOINT_CACHE = checkpoint
    return _CHECKPOINT_CACHE


def _get_model(checkpoint_path: str | Path):
    """获取模型实例（全局缓存）"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        checkpoint = _load_checkpoint(checkpoint_path)
        model = SwinClassifier(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE


def bytes_to_tensor(raw_bytes: bytes) -> torch.Tensor:
    """
    将 .bytes 文件原始字节流转为 32x32 灰度图张量。
    规则与 training/data.py 中的 bytes_tensor 完全一致：
      全序列 1024 等分，已知字节取均值（四舍五入），未知字节不参与均值。
    """
    values = bytearray()
    known = bytearray()

    for line in io.BytesIO(raw_bytes):
        tokens = line.split()
        # 跳过 8 位十六进制地址（如 00401000）
        if tokens and re.fullmatch(rb"[0-9A-Fa-f]{8}", tokens[0]):
            tokens = tokens[1:]
        for token in tokens:
            if token == b"??":
                values.append(0)
                known.append(0)
            elif re.fullmatch(rb"[0-9A-Fa-f]{2}", token):
                values.append(int(token, 16))
                known.append(1)
            else:
                raise ValueError(f"无法解析的字节 token: {token!r}")

    if not known:
        raise ValueError("样本中没有有效字节")

    values_arr = np.frombuffer(bytes(values), dtype=np.uint8)
    known_arr = np.frombuffer(bytes(known), dtype=np.uint8)

    # 1024 等分聚合
    n_bins = 1024
    total = len(values_arr)
    edges = np.floor_divide(np.arange(n_bins + 1, dtype=np.int64) * total, n_bins)
    lengths = np.diff(edges)
    known_counts = np.add.reduceat(known_arr, edges[:-1], dtype=np.int64)
    known_sums = np.add.reduceat(values_arr * known_arr, edges[:-1], dtype=np.int64)
    # np.add.reduceat 对空 bin 处理需修正：长度为零的 bin 求和无意义，用 0 覆盖
    empty_bins = lengths == 0
    known_counts[empty_bins] = 0
    known_sums[empty_bins] = 0

    pixels = np.zeros(n_bins, dtype=np.uint8)
    usable = known_counts > 0
    pixels[usable] = np.floor_divide(
        known_sums[usable] + np.floor_divide(known_counts[usable], 2),
        known_counts[usable],
    ).astype(np.uint8)

    image = pixels.reshape(32, 32)
    tensor = torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0
    return tensor


def predict_from_bytes(raw_bytes: bytes, checkpoint_path: str | Path) -> dict:
    """
    完整推理流水线：
      原始 .bytes 字节流 → 32x32 灰度图 → Swin 模型 → 统一 JSON 结果
    """
    model = _get_model(checkpoint_path)

    # 1. 字节流 → 图像张量 (1, 32, 32)
    x = bytes_to_tensor(raw_bytes).unsqueeze(0)  # (1, 1, 32, 32)

    # 2. 模型推理
    with torch.inference_mode():
        logits = model(x)
        scores = logits.softmax(dim=1)[0].cpu()

    # 3. 组装 top5
    values, indices = scores.topk(5)
    top5 = [
        {"family": FAMILIES[int(i)], "score": float(v)}
        for v, i in zip(values, indices)
    ]

    return {
        "predicted_family": top5[0]["family"],
        "confidence": top5[0]["score"],
        "top5": top5,
        "attention_data": [],  # Swin 不输出 attention map，前端热力图可后续再议
    }