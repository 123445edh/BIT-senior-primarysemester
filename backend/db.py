# -*- coding: utf-8 -*-
"""SQLite 历史记录表操作"""
import json
import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "history.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS classification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_family TEXT,
            confidence REAL,
            file_size INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # 增量迁移：老库补齐明细列，供前端"点击历史行查看详情"使用
    existing = {row[1] for row in c.execute("PRAGMA table_info(classification_history)")}
    for column in ("top5_json", "attention_json"):
        if column not in existing:
            c.execute("ALTER TABLE classification_history ADD COLUMN %s TEXT" % column)
    conn.commit()
    conn.close()


def add_record(filename, predicted_family, confidence, file_size, top5=None, attention=None):
    """写入一条分类记录，返回新记录的 id。

    top5 / attention 为可序列化的明细数据，老调用方不传时为 NULL。
    """
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO classification_history "
        "(filename, predicted_family, confidence, file_size, top5_json, attention_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            filename,
            predicted_family,
            confidence,
            file_size,
            json.dumps(top5, ensure_ascii=False) if top5 else None,
            json.dumps(attention, ensure_ascii=False) if attention else None,
        ),
    )
    conn.commit()
    record_id = c.lastrowid
    conn.close()
    return record_id


def get_history(limit=20):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT id, filename, predicted_family, confidence, file_size, timestamp "
        "FROM classification_history ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_record(record_id):
    """按 id 查询单条记录明细，不存在时返回 None"""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT id, filename, predicted_family, confidence, file_size, timestamp, "
        "top5_json, attention_json FROM classification_history WHERE id = ?",
        (record_id,),
    )
    row = c.fetchone()
    conn.close()
    return row
