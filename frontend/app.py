# -*- coding: utf-8 -*-
"""
木马家族分类系统 —— 前端服务

职责：
  1. 渲染前端页面（index.html）
  2. 代理 /api/* 请求到后端 Flask 服务，避免浏览器跨域

后端地址可通过环境变量 BACKEND_URL 配置，默认 http://127.0.0.1:5000
启动：python app.py  （默认监听 http://127.0.0.1:8000）
"""
import os

import requests
from flask import Flask, Response, render_template, request

app = Flask(__name__, static_folder="static", template_folder="templates")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
REQUEST_TIMEOUT = 120  # 推理可能较慢，放宽超时


@app.get("/")
def index():
    return render_template("index.html")


@app.route("/api/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def api_proxy(subpath):
    """将 /api/* 请求透传到后端服务。"""
    url = f"{BACKEND_URL}/api/{subpath}"

    try:
        if request.method == "GET":
            resp = requests.get(
                url,
                params=request.args.to_dict(flat=False),
                timeout=REQUEST_TIMEOUT,
            )
        elif request.method == "POST":
            # 处理文件上传（multipart/form-data）
            if request.files:
                files = {}
                for key, f in request.files.items():
                    files[key] = (f.filename, f.stream.read(), f.mimetype)
                resp = requests.post(
                    url,
                    files=files,
                    data=request.form.to_dict(flat=False),
                    timeout=REQUEST_TIMEOUT,
                )
            else:
                resp = requests.post(
                    url,
                    json=request.get_json(silent=True),
                    data=request.form.to_dict(flat=False),
                    timeout=REQUEST_TIMEOUT,
                )
        elif request.method == "PUT":
            resp = requests.put(
                url,
                json=request.get_json(silent=True),
                data=request.form.to_dict(flat=False),
                timeout=REQUEST_TIMEOUT,
            )
        elif request.method == "DELETE":
            resp = requests.delete(
                url,
                json=request.get_json(silent=True),
                timeout=REQUEST_TIMEOUT,
            )
        else:
            return Response(status=405)
    except requests.exceptions.ConnectionError:
        return (
            {"status": "error", "message": "无法连接到后端服务，请确认后端已启动"},
            502,
        )
    except requests.exceptions.Timeout:
        return (
            {"status": "error", "message": "后端请求超时，请稍后重试"},
            504,
        )

    excluded_headers = {
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    }
    headers = [
        (k, v)
        for k, v in resp.raw.headers.items()
        if k.lower() not in excluded_headers
    ]
    return Response(resp.content, status=resp.status_code, headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get("FRONTEND_PORT", 8000))
    app.run(debug=True, host="127.0.0.1", port=port)
