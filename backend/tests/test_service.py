# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# 让 pytest 找到 backend/ 和 training/
BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR.parent / "training"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from service import bytes_to_tensor, predict_from_bytes


def test_bytes_to_tensor_shape():
    sample = b"00401000 4D 5A 90 00 ?? 03\n00401006 00 00 00 00\n"
    tensor = bytes_to_tensor(sample)
    assert tensor.shape == (1, 32, 32), f"期望(1,32,32)，实际{tensor.shape}"


def test_bytes_to_tensor_rejects_exe():
    sample = b"\x4d\x5a\x90\x00\x03\x00\x00\x00"
    try:
        bytes_to_tensor(sample)
        assert False, "应该抛出异常"
    except ValueError:
        pass


def test_predict_from_bytes_returns_top5():
    checkpoint = BACKEND_DIR.parent / "training" / "runs" / "base" / "best.pt"
    if not checkpoint.is_file():
        return  # 模型未训练完成，跳过

    sample = b"00401000 4D 5A 90 00\n00401004 03 00 00 00\n"
    result = predict_from_bytes(sample, checkpoint)
    assert "predicted_family" in result
    assert 0 <= result["confidence"] <= 1
    assert len(result["top5"]) == 5