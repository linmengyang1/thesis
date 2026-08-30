"""任务板:TaskItem 数据模型 + SQLite 持久化。

任务板是多 agent 编排的核心状态:Orchestrator 依据任务的
依赖(deps)与状态(status)决定何时向哪个子 agent 分发什么任务。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

TaskKind = Literal['plan', 'research', 'draft', 'review', 'revise', 'format']
TaskStatus = Literal['queued', 'in_progress', 'needs_review', 'in_revision', 'approved', 'merged', 'blocked']


class TaskItem(BaseModel):
	id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	run_id: str = ''  # 代际标识:区分同一任务板上的多轮运行,查找必须带 run_id
	title: str
	kind: TaskKind
	status: TaskStatus = 'queued'
	deps: list[str] = Field(default_factory=list)  # 依赖的任务 id
	assigned_agent: str = ''
	model_tier: str = 'cheap'  # strong / medium / cheap
	artifact_path: str = ''  # 产物文件路径(相对 output_dir)
	acceptance: list[str] = Field(default_factory=list)  # 验收标准,供 Critic 判据
	revision_count: int = 0
	chapter_id: str = ''  # 关联章节 id(草稿/评审任务用)
	note: str = ''


class TaskBoard:
	"""基于 SQLite 的任务板。"""

	def __init__(self, db_path: Path | str) -> None:
		self.db_path = Path(db_path)
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
		self._conn.row_factory = sqlite3.Row
		self._init_schema()

	def _init_schema(self) -> None:
		self._conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS tasks (
				id TEXT PRIMARY KEY,
				run_id TEXT NOT NULL DEFAULT '',
				title TEXT NOT NULL,
				kind TEXT NOT NULL,
				status TEXT NOT NULL,
				deps TEXT NOT NULL DEFAULT '[]',
				assigned_agent TEXT NOT NULL DEFAULT '',
				model_tier TEXT NOT NULL DEFAULT 'cheap',
				artifact_path TEXT NOT NULL DEFAULT '',
				acceptance TEXT NOT NULL DEFAULT '[]',
				revision_count INTEGER NOT NULL DEFAULT 0,
				chapter_id TEXT NOT NULL DEFAULT '',
				note TEXT NOT NULL DEFAULT ''
			)
			"""
		)
		self._conn.commit()

	def add(self, task: TaskItem) -> None:
		self._conn.execute(
			"""INSERT OR REPLACE INTO tasks
			(id, run_id, title, kind, status, deps, assigned_agent, model_tier,
			 artifact_path, acceptance, revision_count, chapter_id, note)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				task.id,
				task.run_id,
				task.title,
				task.kind,
				task.status,
				json.dumps(task.deps),
				task.assigned_agent,
				task.model_tier,
				task.artifact_path,
				json.dumps(task.acceptance),
				task.revision_count,
				task.chapter_id,
				task.note,
			),
		)
		self._conn.commit()

	def update(self, task_id: str, **fields) -> None:
		cols = ', '.join(f'{k} = ?' for k in fields)
		vals = list(fields.values())
		self._conn.execute(f'UPDATE tasks SET {cols} WHERE id = ?', [*vals, task_id])
		self._conn.commit()

	def get(self, task_id: str) -> TaskItem | None:
		row = self._conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
		return self._row_to_task(row) if row else None

	def all(self, run_id: str | None = None) -> list[TaskItem]:
		"""列出任务;传入 run_id 时只返回该代次的任务。"""
		if run_id:
			rows = self._conn.execute('SELECT * FROM tasks WHERE run_id = ?', (run_id,)).fetchall()
		else:
			rows = self._conn.execute('SELECT * FROM tasks').fetchall()
		return [self._row_to_task(r) for r in rows]

	def by_status(self, status: TaskStatus, run_id: str | None = None) -> list[TaskItem]:
		if run_id:
			rows = self._conn.execute(
				'SELECT * FROM tasks WHERE status = ? AND run_id = ?', (status, run_id)
			).fetchall()
		else:
			rows = self._conn.execute('SELECT * FROM tasks WHERE status = ?', (status,)).fetchall()
		return [self._row_to_task(r) for r in rows]

	def ready_tasks(self, run_id: str | None = None) -> list[TaskItem]:
		"""依赖全部满足且 queued 的任务,按依赖深度排序。"""
		all_tasks = {t.id: t for t in self.all(run_id=run_id)}
		ready = []
		for task in all_tasks.values():
			if task.status != 'queued':
				continue
			if all(all_tasks.get(dep) and all_tasks[dep].status in ('approved', 'merged') for dep in task.deps):
				ready.append(task)
		return ready

	def _row_to_task(self, row: sqlite3.Row) -> TaskItem:
		return TaskItem(
			id=row['id'],
			run_id=row['run_id'],
			title=row['title'],
			kind=row['kind'],
			status=row['status'],
			deps=json.loads(row['deps']),
			assigned_agent=row['assigned_agent'],
			model_tier=row['model_tier'],
			artifact_path=row['artifact_path'],
			acceptance=json.loads(row['acceptance']),
			revision_count=row['revision_count'],
			chapter_id=row['chapter_id'],
			note=row['note'],
		)

	def close(self) -> None:
		self._conn.close()
