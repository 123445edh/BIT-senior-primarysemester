# 木马家族分类系统 - 后端

基于 Flask + SQLite 的后端服务，负责接收样本、调用模型、保存历史、返回结果。

## 接口

- `GET /api/health` 健康检查
- `POST /api/predict` 上传样本并返回分类结果（multipart/form-data，字段名 `file`）
- `GET /api/history?limit=20` 查询历史记录

## 安装与运行

```bash
pip install -r requirements.txt
python app.py
```

启动后访问 http://127.0.0.1:5000/api/health

## 测试

```bash
pytest tests/
```

## 目录结构

- `app.py` Flask 路由
- `db.py` SQLite 数据库操作
- `tests/test_api.py` 接口测试
- `history.db` 运行后自动生成的数据库文件（不提交）
