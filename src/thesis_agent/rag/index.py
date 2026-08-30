"""faiss 分区索引 + SQLite 元数据管理。

设计要点(对应 500+ 篇、增量入库):
- 全库按子领域/哈希分成 N 个 shard,每个 shard 一个 faiss IndexFlatIP。
- 向量同时落盘到 SQLite(重建 shard 的数据源),faiss 文件仅作检索加速。
- 新增论文只重建其所在 shard,不触碰其他 shard。
- 检索:每个 shard 独立 top-k,合并后交给 rerank 精排。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_config


class MetadataDB:
	"""论文与分块的元数据 SQLite 存储。"""

	def __init__(self, db_path: Path | str) -> None:
		self.db_path = Path(db_path)
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
		self._conn.row_factory = sqlite3.Row
		self._init_schema()

	def _init_schema(self) -> None:
		self._conn.executescript(
			"""
			CREATE TABLE IF NOT EXISTS papers (
				paper_id TEXT PRIMARY KEY,
				title TEXT NOT NULL,
				authors TEXT NOT NULL DEFAULT '[]',
				year INTEGER,
				venue TEXT NOT NULL DEFAULT '',
				doi TEXT NOT NULL DEFAULT '',
				abstract TEXT NOT NULL DEFAULT '',
				path TEXT NOT NULL DEFAULT '',
				subfield TEXT NOT NULL DEFAULT 'general',
				shard INTEGER NOT NULL DEFAULT 0
			);
			CREATE TABLE IF NOT EXISTS chunks (
				chunk_id TEXT PRIMARY KEY,
				paper_id TEXT NOT NULL,
				seq INTEGER NOT NULL,
				shard INTEGER NOT NULL,
				text TEXT NOT NULL,
				vector BLOB
			);
			CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
			CREATE INDEX IF NOT EXISTS idx_chunks_shard ON chunks(shard);
			CREATE INDEX IF NOT EXISTS idx_papers_subfield ON papers(subfield);
			"""
		)
		self._conn.commit()

	def upsert_paper(self, paper: dict[str, Any]) -> None:
		self._conn.execute(
			"""INSERT OR REPLACE INTO papers
			(paper_id, title, authors, year, venue, doi, abstract, path, subfield, shard)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				paper['paper_id'],
				paper['title'],
				json.dumps(paper.get('authors', [])),
				paper.get('year'),
				paper.get('venue', ''),
				paper.get('doi', ''),
				paper.get('abstract', ''),
				paper.get('path', ''),
				paper.get('subfield', 'general'),
				paper.get('shard', 0),
			),
		)
		self._conn.commit()

	def paper_exists(self, paper_id: str) -> bool:
		return self._conn.execute('SELECT 1 FROM papers WHERE paper_id = ?', (paper_id,)).fetchone() is not None

	def get_paper(self, paper_id: str) -> dict[str, Any] | None:
		row = self._conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
		return self._paper_from_row(row) if row else None

	def all_papers(self) -> list[dict[str, Any]]:
		return [self._paper_from_row(r) for r in self._conn.execute('SELECT * FROM papers').fetchall()]

	def _paper_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
		"""把 papers 行转为 dict,并将 JSON 字段(authors)解码为列表。"""
		paper = dict(row)
		try:
			paper['authors'] = json.loads(paper.get('authors', '[]'))
		except (json.JSONDecodeError, TypeError):
			paper['authors'] = []
		return paper

	def clear_paper_chunks(self, paper_id: str) -> None:
		self._conn.execute('DELETE FROM chunks WHERE paper_id = ?', (paper_id,))
		self._conn.commit()

	def insert_chunk(self, chunk_id: str, paper_id: str, seq: int, shard: int, text: str, vector: bytes) -> None:
		self._conn.execute(
			'INSERT OR REPLACE INTO chunks (chunk_id, paper_id, seq, shard, text, vector) VALUES (?, ?, ?, ?, ?, ?)',
			(chunk_id, paper_id, seq, shard, text, vector),
		)
		self._conn.commit()

	def chunks_of_paper(self, paper_id: str) -> list[dict[str, Any]]:
		rows = self._conn.execute(
			'SELECT * FROM chunks WHERE paper_id = ? ORDER BY seq', (paper_id,)
		).fetchall()
		return [dict(r) for r in rows]

	def chunks_of_shard(self, shard: int) -> list[dict[str, Any]]:
		rows = self._conn.execute('SELECT * FROM chunks WHERE shard = ?', (shard,)).fetchall()
		return [dict(r) for r in rows]

	def chunk_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
		out: dict[str, dict[str, Any]] = {}
		if not chunk_ids:
			return out
		placeholders = ','.join('?' for _ in chunk_ids)
		for r in self._conn.execute(f'SELECT * FROM chunks WHERE chunk_id IN ({placeholders})', chunk_ids):
			out[r['chunk_id']] = dict(r)
		return out

	def any_chunk_vector(self) -> bytes | None:
		"""返回任意一条分块向量(用于推断向量维度),无向量时返回 None。"""
		row = self._conn.execute('SELECT vector FROM chunks WHERE vector IS NOT NULL LIMIT 1').fetchone()
		return row[0] if row is not None else None

	def paper_ids_by_ids(self, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
		out: dict[str, dict[str, Any]] = {}
		if not paper_ids:
			return out
		placeholders = ','.join('?' for _ in paper_ids)
		for r in self._conn.execute(f'SELECT * FROM papers WHERE paper_id IN ({placeholders})', paper_ids):
			out[r['paper_id']] = self._paper_from_row(r)
		return out

	def close(self) -> None:
		self._conn.close()


class FaissShards:
	"""N 个 faiss shard 的管理器,支持增量追加与按 shard 重建。"""

	def __init__(self, index_dir: Path | str, n_shards: int | None = None) -> None:
		self.index_dir = Path(index_dir)
		self.index_dir.mkdir(parents=True, exist_ok=True)
		self.n_shards = n_shards or get_config().rag.shards
		self.meta = MetadataDB(self.index_dir / 'metadata.sqlite')

	def _index_path(self, shard: int) -> Path:
		return self.index_dir / f'shard_{shard}.faiss'

	def _load_shard(self, shard: int):
		import os
		import shutil
		import tempfile

		import faiss

		path = self._index_path(shard)
		if not path.exists():
			return None
		# faiss C++ fopen 无法处理中文路径:先拷到 ASCII 临时文件再读
		fd, tmp = tempfile.mkstemp(suffix='.faiss')
		os.close(fd)
		try:
			shutil.copy(path, tmp)
			return faiss.read_index(tmp)
		finally:
			try:
				os.remove(tmp)
			except OSError:
				pass

	def _save_shard(self, shard: int, index) -> None:
		import os
		import shutil
		import tempfile

		import faiss

		# faiss 写 ASCII 临时文件,再用 Python 搬到目标路径(兼容中文路径)
		fd, tmp = tempfile.mkstemp(suffix='.faiss')
		os.close(fd)
		try:
			faiss.write_index(index, tmp)
			shutil.move(tmp, self._index_path(shard))
		finally:
			try:
				os.remove(tmp)
			except OSError:
				pass

	def _infer_dim(self) -> int:
		"""推断向量维度:优先取元数据中任意分块向量,其次读已有分片;全库无向量时返回 0。"""
		vector = self.meta.any_chunk_vector()
		if vector:
			return len(np.frombuffer(vector, dtype=np.float32))
		for shard in range(self.n_shards):
			index = self._load_shard(shard)
			if index is not None and index.ntotal > 0:
				return index.d
		return 0

	def rebuild_shard(self, shard: int) -> None:
		"""从 SQLite 中的向量重建某个 shard(增量入库的核心操作)。"""
		import faiss

		chunks = self.meta.chunks_of_shard(shard)
		if not chunks:
			# 空 shard:用推断维度写一个空索引;全库无向量时跳过(检索会忽略缺失分片)
			dim = self._infer_dim()
			if dim <= 0:
				return
			self.dim = dim
			index = faiss.IndexFlatIP(dim)
			self._save_shard(shard, index)
			return
		vectors = np.array([np.frombuffer(c['vector'], dtype=np.float32) for c in chunks], dtype=np.float32)
		self.dim = vectors.shape[1]
		index = faiss.IndexFlatIP(self.dim)
		index.add(vectors)
		self._save_shard(shard, index)

	def rebuild_all(self) -> None:
		for shard in range(self.n_shards):
			self.rebuild_shard(shard)

	def add_paper(self, paper: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
		"""新增/更新一篇论文:写入元数据、追加向量、重建所属 shard。"""
		shard = paper['shard']
		self.meta.upsert_paper(paper)
		self.meta.clear_paper_chunks(paper['paper_id'])
		vectors = np.array([np.frombuffer(c['vector'], dtype=np.float32) for c in chunks], dtype=np.float32)
		self.dim = vectors.shape[1]
		for c in chunks:
			self.meta.insert_chunk(c['chunk_id'], c['paper_id'], c['seq'], c['shard'], c['text'], c['vector'])
		self.rebuild_shard(shard)

	def search(self, query_vector: list[float], top_k: int, exclude_paper_ids: set[str] | None = None):
		"""粗排:遍历所有 shard,合并 top_k_candidate 个候选分块。

		返回 [(score, chunk_id)] 列表。faiss 索引与 chunk 的对应关系:
		faiss 内第 i 个向量 = shard 内按 seq 排序的第 i 个 chunk。
		"""
		import faiss

		q = np.array([query_vector], dtype=np.float32)
		exclude_paper_ids = exclude_paper_ids or set()
		candidates: list[tuple[float, str]] = []
		for shard in range(self.n_shards):
			index = self._load_shard(shard)
			if index is None:
				continue
			scores, idxs = index.search(q, top_k)
			chunks = self.meta.chunks_of_shard(shard)
			# chunks 按 seq 排序(SQL 里 ORDER BY seq),与 faiss add 顺序一致
			for score, i in zip(scores[0], idxs[0]):
				if i < 0 or i >= len(chunks):
					continue
				chunk = chunks[i]
				if chunk['paper_id'] in exclude_paper_ids:
					continue
				candidates.append((float(score), chunk['chunk_id']))
		candidates.sort(key=lambda x: x[0], reverse=True)
		return candidates[:top_k]

	def close(self) -> None:
		self.meta.close()
