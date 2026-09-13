# -*- coding: utf-8 -*-
"""
清理历史分类记录库 backend/history.db 的命令行工具。

=====================================================================
用法
=====================================================================
先激活环境

在**仓库根目录**下执行（脚本会自动定位 backend/history.db，不依赖 cwd）：

  # 1) 查看当前记录（只读，不做任何改动）
  python scripts/clean_history.py --list

  # 2) 预览将要删除的内容（dry-run，只统计不删）
  python scripts/clean_history.py --all --dry-run
  python scripts/clean_history.py --keep 10 --dry-run

  # 3) 清空全部历史记录（删除前会自动备份 history.db）
  python scripts/clean_history.py --all

  # 4) 只保留最近 10 条，其余删除
  python scripts/clean_history.py --keep 10

  # 5) 删除 7 天前的记录
  python scripts/clean_history.py --days 7

  # 6) 删除某个日期之前的记录（YYYY-MM-DD，UTC 自然日）
  python scripts/clean_history.py --before 2026-09-06

  # 7) 跳过交互确认（便于写进自动化 / 定时任务）
  python scripts/clean_history.py --all --yes

  # 8) 删除后压缩数据库文件，回收磁盘空间
  python scripts/clean_history.py --all --yes --vacuum

  # 9) 操作另一个库文件（比如备份文件）
  python scripts/clean_history.py --db backend/history.db.bak-20260913-120000 --list

  # 10) 不备份直接删（危险，不建议）
  python scripts/clean_history.py --all --yes --no-backup

=====================================================================
参数说明
=====================================================================
  --all             删除全部记录
  --keep N          保留最近 N 条（按 id 倒序），删除更早的
  --days N          删除 N 天前的记录
  --before DATE     删除 DATE（YYYY-MM-DD）之前的记录
  --list            只列出记录，不删除
  --dry-run         只显示将删除多少条，不实际删除
  -y, --yes         跳过交互确认
  --vacuum          清理后执行 VACUUM 压缩库文件
  --db PATH         指定数据库路径（默认 <仓库根>/backend/history.db）
  --no-backup       删除前不备份（默认备份为 history.db.bak-<时间戳>）

=====================================================================
注意事项
=====================================================================
  * 时间戳字段是 SQLite 的 CURRENT_TIMESTAMP，存的是 **UTC 时间**，
    比北京时间早 8 小时。所以 --days / --before 的比较基准也是 UTC 自然日。
  * 默认行为有保护：写操作先打印统计，需在终端确认（输入 y 回车）才执行；
    非交互环境（管道、计划任务）下必须显式加 -y，否则脚本会直接退出。
  * 删除前会把整个库文件复制一份为 history.db.bak-<时间戳>，
    放在原库同目录下，出问题可以直接改回来。
  * 当记录被清空时，脚本会顺手重置自增 id（sqlite_sequence），
    让下一条新记录从 1 开始编号；只删一部分时不重置。

示例输出：
    $ python scripts/clean_history.py --keep 10
    [信息] 数据库：D:\\...\\backend\\history.db
    [信息] 当前共 37 条记录
    [计划] 保留最近 10 条，将删除 27 条：
           id 1  ~ id 27
    [确认] 确定执行删除吗？(y/N) y
    [备份] D:\\...\\history.db.bak-20260913-120500
    [完成] 已删除 27 条，剩余 10 条
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

TABLE = "classification_history"

# 脚本位于 <仓库根>/scripts/ 下，因此库文件在上一级的 backend/ 里
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "backend" / "history.db"


def info(msg):
    print("[信息] " + msg)


def plan(msg):
    print("[计划] " + msg)


def ok(msg):
    print("[完成] " + msg)


def fail(msg):
    print("[错误] " + msg)


def connect(db_path: Path):
    """连接数据库并把行转成可按列名访问的对象"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def check_table(conn) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
    ).fetchone()
    return row is not None


def fetch_rows(conn):
    return conn.execute(
        "SELECT id, filename, predicted_family, confidence, file_size, timestamp "
        "FROM %s ORDER BY id DESC" % TABLE
    ).fetchall()


def list_records(rows):
    if not rows:
        print("（暂无记录）")
        return
    print("%-6s %-34s %-16s %-10s %-22s" % ("ID", "文件名", "预测家族", "置信度", "分类时间(UTC)"))
    print("-" * 92)
    for r in rows:
        conf = r["confidence"]
        conf_str = ("%.2f%%" % (conf * 100)) if isinstance(conf, (int, float)) else "-"
        name = r["filename"] or ""
        if len(name) > 32:
            name = name[:29] + "..."
        print(
            "%-6s %-34s %-16s %-10s %-22s"
            % (r["id"], name, r["predicted_family"] or "-", conf_str, r["timestamp"] or "-")
        )
    print("-" * 92)
    print("共 %d 条" % len(rows))


def backup_db(db_path: Path) -> Path:
    """备份整个库文件，返回备份路径"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = db_path.with_name(db_path.name + ".bak-" + stamp)
    shutil.copy2(str(db_path), str(target))
    print("[备份] " + str(target))
    return target


def confirm(prompt: str, assume_yes: bool) -> bool:
    """交互确认；非交互环境（无 TTY）时必须显式 -y"""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        fail("当前不是交互终端，请显式加 -y / --yes 再执行删除。")
        return False
    try:
        answer = input("[确认] %s (y/N) " % prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def build_plan(conn, args):
    """
    根据参数算出「要删哪些」。返回 (delete_sql, params, 描述文字, 将删除条数)
    返回 None 表示没有可删的（视为无需操作）
    """
    rows = fetch_rows(conn)
    total = len(rows)
    if total == 0:
        return None

    if args.all:
        count = total
        desc = "清空全部记录"
        return ("DELETE FROM %s" % TABLE, (), desc, count)

    if args.keep is not None:
        keep = args.keep
        if keep >= total:
            return None
        # 保留 id 最大的 keep 条
        keep_ids = [r["id"] for r in rows[:keep]]
        placeholders = ",".join("?" * len(keep_ids))
        count = total - keep
        desc = "保留最近 %d 条（id %d~%d），删除其余" % (
            keep,
            min(keep_ids),
            max(keep_ids),
        )
        return (
            "DELETE FROM %s WHERE id NOT IN (%s)" % (TABLE, placeholders),
            tuple(keep_ids),
            desc,
            count,
        )

    # --days / --before：按 UTC 自然日比较
    if args.days is not None:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
        desc = "删除 %d 天前（早于 %s UTC）的记录" % (args.days, cutoff_date)
    else:
        cutoff_date = args.before
        desc = "删除早于 %s（UTC）的记录" % cutoff_date

    cutoff = cutoff_date + " 00:00:00"
    count = conn.execute(
        "SELECT COUNT(*) FROM %s WHERE timestamp < ?" % TABLE, (cutoff,)
    ).fetchone()[0]
    if count == 0:
        return None
    return ("DELETE FROM %s WHERE timestamp < ?" % TABLE, (cutoff,), desc, count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="清理历史分类记录库 backend/history.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/clean_history.py --list\n"
            "  python scripts/clean_history.py --all --dry-run\n"
            "  python scripts/clean_history.py --keep 10\n"
            "  python scripts/clean_history.py --days 7 --yes\n"
            "  python scripts/clean_history.py --before 2026-09-06 --vacuum\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="删除全部记录")
    mode.add_argument("--keep", type=int, metavar="N", help="保留最近 N 条，删除其余")
    mode.add_argument("--days", type=int, metavar="N", help="删除 N 天前的记录")
    mode.add_argument("--before", metavar="DATE", help="删除 DATE(YYYY-MM-DD) 之前的记录")
    mode.add_argument("--list", action="store_true", help="只列出记录，不删除")

    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    parser.add_argument("-y", "--yes", action="store_true", help="跳过交互确认")
    parser.add_argument("--vacuum", action="store_true", help="清理后 VACUUM 压缩库文件")
    parser.add_argument("--db", metavar="PATH", help="数据库路径（默认 backend/history.db）")
    parser.add_argument("--no-backup", action="store_true", help="删除前不备份")

    args = parser.parse_args()

    db_path = Path(args.db).resolve() if args.db else DEFAULT_DB
    info("数据库：" + str(db_path))

    if not db_path.exists():
        fail("文件不存在，无需清理。")
        return 1

    # --before 格式校验
    if args.before:
        try:
            datetime.strptime(args.before, "%Y-%m-%d")
        except ValueError:
            fail("--before 需要 YYYY-MM-DD 格式，例如 2026-09-06")
            return 2

    conn = connect(db_path)
    try:
        if not check_table(conn):
            info("库中没有 %s 表，无需清理。" % TABLE)
            return 0

        rows = fetch_rows(conn)
        info("当前共 %d 条记录" % len(rows))

        # 只读模式
        if args.list or not any([args.all, args.keep is not None, args.days is not None, args.before]):
            list_records(rows)
            if not args.list:
                print()
                info("未指定清理方式，仅列出记录。可用 --all / --keep N / --days N / --before DATE。")
            return 0

        plan_result = build_plan(conn, args)
        if plan_result is None:
            info("没有符合条件的记录需要清理。")
            return 0

        sql, params, desc, count = plan_result
        plan("%s，将删除 %d 条，剩余 %d 条" % (desc, count, len(rows) - count))

        if args.dry_run:
            info("dry-run 模式，未做任何改动。")
            return 0

        if not confirm("确定执行删除吗？", args.yes):
            info("已取消，未做任何改动。")
            return 0

        if not args.no_backup:
            backup_db(db_path)

        cur = conn.execute(sql, params)
        deleted = cur.rowcount
        conn.commit()

        # 整表清空后重置自增 id，让下一条记录从 1 开始
        if fetch_rows(conn) == []:
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (TABLE,))
                conn.commit()
                info("表已清空，自增 id 已重置（下一条记录将从 1 开始）")
            except sqlite3.OperationalError:
                pass  # 没有 sqlite_sequence 表说明该库未用 AUTOINCREMENT，忽略

        ok("已删除 %d 条，剩余 %d 条" % (deleted, len(fetch_rows(conn))))

        if args.vacuum:
            conn.execute("VACUUM")
            conn.commit()
            ok("已执行 VACUUM，库文件大小：%d 字节" % db_path.stat().st_size)

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
