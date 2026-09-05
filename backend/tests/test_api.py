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


def test_predict_and_history(client):
    data = {"file": (io.BytesIO(b"malware-sample-bytes"), "sample.bin")}
    r = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "success"
    assert body["predicted_family"]
    assert "confidence" in body
    assert "top5" in body

    r2 = client.get("/api/history?limit=20")
    assert r2.status_code == 200
    history = r2.get_json()["history"]
    assert len(history) >= 1
    assert history[0]["filename"] == "sample.bin"
    assert history[0]["result"] == body["predicted_family"]
