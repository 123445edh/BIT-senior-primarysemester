# -*- coding: utf-8 -*-
"""
木马家族分类系统 —— 后端服务

接口：
  GET  /api/health                 健康检查
  POST /api/predict                上传样本并返回分类结果（multipart/form-data，字段名 file）
  GET  /api/history                查询历史分类记录（?limit=20）
  GET  /api/history/<record_id>    查询单条历史记录明细（含 Top-5 候选概率）
"""
from flask import Flask, request, jsonify
from service import predict_from_bytes
from pathlib import Path
import json
import db
import os



MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "training" / "runs" / "base32_ep100" / "best.pt"),
)

app = Flask(__name__)
# 启动时初始化数据库（自动建表 + 增量补列）
db.init_db()


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify({"status": "error", "message": "缺少 file 字段"}), 400

    filename = file.filename or "unknown"
    data = file.read()
    file_size = len(data)

    try:
        result = predict_from_bytes(data, MODEL_PATH)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    record_id = db.add_record(
        filename,
        result["predicted_family"],
        result["confidence"],
        file_size,
        top5=result.get("top5"),
        attention=result.get("attention_data"),
    )

    return jsonify({
        "status": "success",
        "record_id": record_id,
        "predicted_family": result["predicted_family"],
        "confidence": result["confidence"],
        "top5": result["top5"],
        "attention_data": result["attention_data"],
    })



@app.get("/api/history")
def history():
    limit = request.args.get("limit", 20, type=int)
    if limit is None or limit < 1:
        limit = 20
    rows = db.get_history(limit)
    items = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "result": r["predicted_family"],
            "confidence": r["confidence"],
            "file_size": r["file_size"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]
    return jsonify({"history": items})


@app.get("/api/history/<int:record_id>")
def history_detail(record_id):
    """单条历史记录明细：基础字段 + 已保存的 Top-5 / Attention 数据"""
    row = db.get_record(record_id)
    if row is None:
        return jsonify({"status": "error", "message": "记录不存在"}), 404

    def _load(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    return jsonify({
        "status": "success",
        "record": {
            "id": row["id"],
            "filename": row["filename"],
            "predicted_family": row["predicted_family"],
            "confidence": row["confidence"],
            "file_size": row["file_size"],
            "timestamp": row["timestamp"],
            "top5": _load(row["top5_json"]),
            "attention_data": _load(row["attention_json"]),
        },
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", 5000))
    app.run(debug=True, host=host, port=port)
