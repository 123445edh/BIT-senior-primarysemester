# -*- coding: utf-8 -*-
"""SQLite 历史记录表操作"""
import sqlite3

DB_PATH = "history.db"


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
    conn.commit()
    conn.close()


def add_record(filename, predicted_family, confidence, file_size):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO classification_history "
        "(filename, predicted_family, confidence, file_size) "
        "VALUES (?, ?, ?, ?)",
        (filename, predicted_family, confidence, file_size),
    )
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT id, filename, predicted_family, confidence, timestamp "
        "FROM classification_history ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows
