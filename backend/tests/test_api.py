# -*- coding: utf-8 -*-
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import app as app_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.db, "DB_PATH", str(tmp_path / "test.db"))
    app_module.db.init_db()
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_missing_file(client):
    r = client.post("/api/predict")
    assert r.status_code == 400
    assert r.get_json()["status"] == "error"


def test_predict_and_history(client, monkeypatch):
    # 用假推理结果替换真实模型调用，避免测试依赖 torch 和权重文件
    monkeypatch.setattr(
        app_module, "predict_from_bytes",
        lambda data, path: {
            "predicted_family": "Mirai",
            "confidence": 0.92,
            "top5": [{"family": "Mirai", "score": 0.92}],
            "attention_data": [],
        },
    )

    data = {"file": (io.BytesIO(b"00401000 4D 5A 90 00"), "sample.bytes")}
    r = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "success"
    assert body["predicted_family"] == "Mirai"
    assert "confidence" in body
    assert "top5" in body

    r2 = client.get("/api/history?limit=20")
    assert r2.status_code == 200
    history = r2.get_json()["history"]
    assert len(history) >= 1
    assert history[0]["filename"] == "sample.bytes"
    assert history[0]["result"] == "Mirai"
