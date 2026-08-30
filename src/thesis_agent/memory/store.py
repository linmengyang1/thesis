"""长期记忆存储:SQLite 命名空间 KV(跨会话持久)。

langgraph Store 在当前版本接口变动频繁,这里用轻量自研存储,
语义等价(namespace / key / value),且同步接口可在 langgraph 的
线程池条件边中直接使用。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..config import get_config


class MemoryStore:
	def __init__(self, db_path: Path | str | None = None) -> None:
		if db_path is None:
			db_path = get_config().db_file.parent / 'memory.sqlite'
		self.db_path = Path(db_path)
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
		# 并发安全:WAL 允许读写并发,busy_timeout 让并行节点等待而非直接报 SQLITE_BUSY
		self._conn.execute('PRAGMA journal_mode=WAL')
		self._conn.execute('PRAGMA busy_timeout=5000')
		self._conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS memory (
				namespace TEXT NOT NULL,
				key TEXT NOT NULL,
				value TEXT NOT NULL,
				updated_at INTEGER NOT NULL DEFAULT 0,
				PRIMARY KEY (namespace, key)
			)
			"""
		)
		self._conn.commit()

	def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
		import time

		self._conn.execute(
			'INSERT OR REPLACE INTO memory (namespace, key, value, updated_at) VALUES (?, ?, ?, ?)',
			(namespace, key, json.dumps(value, ensure_ascii=False), int(time.time())),
		)
		self._conn.commit()

	def get(self, namespace: str, key: str) -> dict[str, Any] | None:
		row = self._conn.execute(
			'SELECT value FROM memory WHERE namespace = ? AND key = ?', (namespace, key)
		).fetchone()
		if row is None:
			return None
		try:
			return json.loads(row[0])
		except json.JSONDecodeError:
			return None

	def list(self, namespace: str) -> list[dict[str, Any]]:
		rows = self._conn.execute(
			'SELECT key, value FROM memory WHERE namespace = ? ORDER BY updated_at DESC', (namespace,)
		).fetchall()
		return [json.loads(r[1]) for r in rows]

	def recent(self, namespace: str, limit: int = 10) -> list[dict[str, Any]]:
		rows = self._conn.execute(
			'SELECT value FROM memory WHERE namespace = ? ORDER BY updated_at DESC LIMIT ?',
			(namespace, limit),
		).fetchall()
		return [json.loads(r[0]) for r in rows]

	def delete(self, namespace: str, key: str) -> None:
		self._conn.execute('DELETE FROM memory WHERE namespace = ? AND key = ?', (namespace, key))
		self._conn.commit()

	def close(self) -> None:
		self._conn.close()
