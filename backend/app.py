# -*- coding: utf-8 -*-
"""
木马家族分类系统 —— 后端服务

接口：
  GET  /api/health         健康检查
  POST /api/predict        上传样本并返回分类结果（multipart/form-data，字段名 file）
  GET  /api/history        查询历史分类记录（?limit=20）
"""
from flask import Flask, request, jsonify
import db

app = Flask(__name__)

# 启动时初始化数据库（自动建表）
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

    # ===== TODO：接入模型推理（等程思涵的模型好了以后替换这里）=====
    # 现在先用 Mock 假结果，保证前后端能先联调
    predicted_family = "Mirai"
    confidence = 0.92
    top5 = [{"family": "Mirai", "score": 0.92}]
    attention_data = []
    # =========================================================

    db.add_record(filename, predicted_family, confidence, file_size)

    return jsonify({
        "status": "success",
        "predicted_family": predicted_family,
        "confidence": confidence,
        "top5": top5,
        "attention_data": attention_data,
    })


@app.get("/api/history")
def history():
    limit = request.args.get("limit", 20, type=int)
    if limit is None or limit < 1:
        limit = 20
    rows = db.get_history(limit)
    items = [
        {"id": r[0], "filename": r[1], "result": r[2], "timestamp": r[4]}
        for r in rows
    ]
    return jsonify({"history": items})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
