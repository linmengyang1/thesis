"""分层检索:双路召回(faiss 向量 + BM25 关键词,RRF 融合) → 元数据过滤 → bge-reranker 精排。

返回结果携带 paper_id / chunk_id / 文献元数据,供写作者做
citation key 溯源(防幻觉的核心)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import get_config
from .embedder import Embedder, Reranker
from .index import FaissShards

# RRF 融合常数:排名倒数求和的平滑项,业界常用值
_RRF_K = 60

# 分词:英文按词、中文按单字(学术查询以英文为主,避免引入重分词依赖)
_TOKEN_RE = re.compile(r'[a-z0-9]+|[\u4e00-\u9fff]')


def _tokenize(text: str) -> list[str]:
	return _TOKEN_RE.findall(text.lower())


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
	def __init__(self, index: FaissShards, hybrid: bool = True) -> None:
		self.index = index
		self.cfg = get_config()
		# hybrid=False 时仅向量单路召回(用于 A/B 评测对比)
		self.hybrid = hybrid
		# BM25 进程级缓存:(构建时的全库 chunk 数, bm25 索引, chunk_id 顺序表, paper_id 映射)
		# chunk 总数变化(增量入库/删除)即整体重建,库不变则复用
		self._bm25_cache: tuple[int, object, list[str], dict[str, str]] | None = None

	def _bm25_search(
		self, query: str, top_k: int, exclude_paper_ids: set[str] | None
	) -> list[tuple[float, str]]:
		"""BM25 关键词召回路:全库 chunk 文本建索引,返回 [(bm25分数, chunk_id)]。"""
		from rank_bm25 import BM25Okapi

		total = self.index.meta.chunk_count()
		cache = self._bm25_cache
		if cache is None or cache[0] != total:
			rows = self.index.meta.all_chunks_light()
			chunk_ids = [r['chunk_id'] for r in rows]
			paper_by_chunk = {r['chunk_id']: r['paper_id'] for r in rows}
			corpus = [_tokenize(r['text']) for r in rows]
			bm25 = BM25Okapi(corpus) if corpus else None
			self._bm25_cache = (total, bm25, chunk_ids, paper_by_chunk)
			cache = self._bm25_cache
		_, bm25, chunk_ids, paper_by_chunk = cache
		if bm25 is None:
			return []
		q_tokens = _tokenize(query)
		if not q_tokens:
			return []
		exclude = exclude_paper_ids or set()
		# 排除论文的 chunk 不返回,但保留在语料中保持 IDF 稳定
		scores = bm25.get_scores(q_tokens)
		pairs = [
			(float(s), cid)
			for s, cid in zip(scores, chunk_ids)
			if s > 0 and paper_by_chunk.get(cid) not in exclude
		]
		pairs.sort(key=lambda x: x[0], reverse=True)
		return pairs[:top_k]

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

		# 1) 粗排:双路召回(faiss 向量 + BM25 关键词)RRF 融合,保持 (分数, chunk_id) 结构
		vec_candidates = self.index.search(q_vec, self.cfg.rag.top_k_candidate, exclude_paper_ids=exclude_paper_ids)
		bm25_candidates = self._bm25_search(query, self.cfg.rag.top_k_candidate, exclude_paper_ids) if self.hybrid else []
		fused: dict[str, float] = {}
		for ranking in (vec_candidates, bm25_candidates):
			for rank, (_, chunk_id) in enumerate(ranking, start=1):
				fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
		candidates = sorted(((s, cid) for cid, s in fused.items()), key=lambda x: x[1], reverse=True)
		candidates = candidates[: self.cfg.rag.top_k_candidate]
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
