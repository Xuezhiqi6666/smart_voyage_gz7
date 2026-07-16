"""
执行 SQL 文件工具脚本

"""

import os
import sys

# 将项目根目录加入 Python 搜索路径，以便导入根目录下的 config.py
# （当前文件在二级子目录中，需向上两级才能到达项目根目录）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import mysql.connector
from mysql.connector import Error
from config import Config


def execute_sql_file(sql_file_path: str):
    """
    读取并执行指定的 SQL 文件。

    :param sql_file_path: SQL 文件的路径
    """
    cfg = Config()

    # 解析 SQL 文件路径（支持相对路径）
    if not os.path.isabs(sql_file_path):
        sql_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), sql_file_path)

    if not os.path.exists(sql_file_path):
        print(f"❌ SQL 文件不存在: {sql_file_path}")
        return

    # 读取 SQL 文件内容
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    print(f"📄 读取 SQL 文件: {sql_file_path}")
    print(f"🔗 连接数据库: {cfg.host}:{cfg.port}  用户: {cfg.user}")

    conn = None
    cursor = None
    try:
        # 建立数据库连接（不指定 database，因为 SQL 文件可能包含 CREATE DATABASE / USE）
        conn = mysql.connector.connect(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user, # 如果用root，可以改为 “root”
            password=cfg.password
        )
        cursor = conn.cursor()

        # 按分号拆分 SQL 语句，过滤空语句和纯注释
        statements = []
        current = []
        for line in sql_content.splitlines():
            stripped = line.strip()
            # 跳过纯注释行和空行
            if stripped.startswith('--') or stripped == '':
                continue
            current.append(line)
            if stripped.endswith(';'):
                stmt = '\n'.join(current).strip()
                if stmt and stmt != ';':
                    statements.append(stmt)
                current = []

        # 处理末尾没有分号的最后一条语句
        if current:
            stmt = '\n'.join(current).strip()
            if stmt and stmt != ';':
                statements.append(stmt)

        print(f"📋 共解析出 {len(statements)} 条 SQL 语句\n")

        # 逐条执行
        success_count = 0
        fail_count = 0
        for i, stmt in enumerate(statements, 1):
            try:
                cursor.execute(stmt)
                # 消耗可能的多结果集（如 CREATE TABLE 不返回结果，但 SELECT 会）
                try:
                    cursor.fetchall()
                except Error:
                    pass
                conn.commit()
                # 截取语句前 60 个字符作为摘要显示
                preview = stmt[:60].replace('\n', ' ')
                print(f"  ✅ [{i}/{len(statements)}] {preview}...")
                success_count += 1
            except Error as e:
                preview = stmt[:60].replace('\n', ' ')
                print(f"  ❌ [{i}/{len(statements)}] {preview}...")
                print(f"     错误: {e}")
                fail_count += 1

        print(f"\n{'='*50}")
        print(f"🏁 执行完毕: 成功 {success_count} 条, 失败 {fail_count} 条")

    except Error as e:
        print(f"❌ 数据库连接或执行错误: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("🔒 数据库连接已关闭")


if __name__ == '__main__':
    # sql_path = "create_all_tables.sql"
    sql_path = "insert_data.sql"
    execute_sql_file(sql_path)
