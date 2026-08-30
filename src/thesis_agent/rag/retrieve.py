"""分层检索:faiss 粗排 → 元数据过滤 → bge-reranker 精排。

返回结果携带 paper_id / chunk_id / 文献元数据,供写作者做
citation key 溯源(防幻觉的核心)。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import get_config
from .embedder import Embedder, Reranker
from .index import FaissShards


@dataclass
class RetrievedChunk:
	chunk_id: str
	paper_id: str
	text: str
	title: str
	authors: list[str]
	year: int | None
	venue: str
	doi: str
	score: float  # rerank 分数


class Retriever:
	def __init__(self, index: FaissShards) -> None:
		self.index = index
		self.cfg = get_config()

	def retrieve(
		self,
		query: str,
		top_k: int | None = None,
		exclude_paper_ids: set[str] | None = None,
		min_year: int | None = None,
		subfield: str | None = None,
	) -> list[RetrievedChunk]:
		"""对单个 query 做分层检索。"""
		top_k = top_k or self.cfg.rag.top_k_final
		q_vec = Embedder.embed_one(query)

		# 1) 粗排:faiss 全 shard 合并
		candidates = self.index.search(q_vec, self.cfg.rag.top_k_candidate, exclude_paper_ids=exclude_paper_ids)
		if not candidates:
			return []
		chunk_map = self.index.meta.chunk_by_ids([cid for _, cid in candidates])
		paper_ids = {c['paper_id'] for c in chunk_map.values()}
		paper_map = self.index.meta.paper_ids_by_ids(list(paper_ids))

		# 2) 元数据过滤
		docs: list[tuple[str, str]] = []  # (chunk_id, text)
		for score, chunk_id in candidates:
			chunk = chunk_map.get(chunk_id)
			if chunk is None:
				continue
			paper = paper_map.get(chunk['paper_id']) or {}
			if min_year is not None and (paper.get('year') or 9999) < min_year:
				continue
			if subfield and paper.get('subfield') not in ('general', subfield):
				continue
			docs.append((chunk_id, chunk['text']))
			if len(docs) >= self.cfg.rag.top_k_candidate:
				break

		# 3) rerank 精排
		rerank_scores = Reranker.score(query, [text for _, text in docs])
		ranked = sorted(zip(rerank_scores, docs), key=lambda x: x[0], reverse=True)

		results: list[RetrievedChunk] = []
		for score, (chunk_id, text) in ranked[:top_k]:
			chunk = chunk_map[chunk_id]
			paper = paper_map.get(chunk['paper_id']) or {}
			results.append(
				RetrievedChunk(
					chunk_id=chunk_id,
					paper_id=chunk['paper_id'],
					text=text,
					title=paper.get('title', ''),
					authors=paper.get('authors', []),
					year=paper.get('year'),
					venue=paper.get('venue', ''),
					doi=paper.get('doi', ''),
					score=float(score),
				)
			)
		return results
