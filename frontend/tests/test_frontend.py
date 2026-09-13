# -*- coding: utf-8 -*-
"""前端服务最小测试：首页渲染 + API 代理透传 + 后端不可用时的错误处理。"""
import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND_DIR))

import pytest
import app as frontend_app


class FakeResp:
    def __init__(self, content=b"", status_code=200):
        self.content = content
        self.status_code = status_code
        self.raw = type("Raw", (), {"headers": {"Content-Type": "application/json"}})()


@pytest.fixture()
def client():
    frontend_app.app.config["TESTING"] = True
    with frontend_app.app.test_client() as c:
        yield c


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200


def test_proxy_health(client, monkeypatch):
    monkeypatch.setattr(
        frontend_app.requests, "get",
        lambda *a, **k: FakeResp(b'{"status": "ok"}', 200),
    )
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_proxy_backend_down(client, monkeypatch):
    def raise_conn(*a, **k):
        raise frontend_app.requests.exceptions.ConnectionError()

    monkeypatch.setattr(frontend_app.requests, "get", raise_conn)
    r = client.get("/api/health")
    assert r.status_code == 502
    assert r.get_json()["status"] == "error"
