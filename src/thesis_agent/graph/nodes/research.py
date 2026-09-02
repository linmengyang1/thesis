"""Researcher 节点:对每个章节做分层检索(RAG + Web 搜索),产出带溯源的素材。"""
from __future__ import annotations

import re
from typing import Any

from ...memory.conversation import ConversationMemory
from ...rag.retrieve import RetrievedChunk
from ..runtime import ThesisRuntime


def _chunk_to_dict(c: RetrievedChunk) -> dict[str, Any]:
	return {
		'chunk_id': c.chunk_id,
		'paper_id': c.paper_id,
		'text': c.text,
		'title': c.title,
		'authors': c.authors,
		'year': c.year,
		'venue': c.venue,
		'doi': c.doi,
		'score': c.score,
	}


def _web_to_chunk(c: Any) -> dict[str, Any]:
	"""把 web 搜索结果(Citation)转成与本地 chunk 同构的字典,供 Drafter/Editor 消费。

	web 结果没有 paper_id,用 'web:' 前缀 + DOI/标题 slug 构造稳定 id,
	保证 Editor 构建引用库(BibTeX)时能去重并收录;摘要作为正文片段。
	"""
	title = c.title or 'untitled'
	paper_id = 'web:' + (c.doi or re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:40])
	return {
		'chunk_id': paper_id,
		'paper_id': paper_id,
		'text': (c.abstract or '')[:800],
		'title': c.title,
		'authors': c.authors or [],
		'year': c.year,
		'venue': c.venue or 'arXiv',
		'doi': c.doi or '',
		'score': 0.0,
	}


_EXPAND_SYSTEM = """You expand a retrieval query for academic paper search.
Given the paper topic, chapter title and keywords, produce exactly 3 diverse English search queries:
1) a synonym/paraphrase variant, 2) an abbreviation-expansion variant, 3) a related-terms variant.
Return JSON ONLY: {"queries": ["...", "...", "..."]}"""


async def _expand_queries(rt: ThesisRuntime, topic: str, chapter: Any, fallback: str) -> list[str]:
	"""query-expansion 技能:LLM 生成 3 个查询变体,失败回退原始查询。"""
	from ..utils import extract_json

	try:
		llm = rt.registry.cheap()
		user = (
			f'Topic: {topic}\nChapter: {chapter.title}\n'
			f'Keywords: {", ".join(chapter.keywords)}\nBase query: {fallback}'
		)
		resp = await llm.acomplete(
			[{'role': 'system', 'content': _EXPAND_SYSTEM}, {'role': 'user', 'content': user}]
		)
		data = extract_json(resp) or {}
		variants = [str(q).strip() for q in (data.get('queries') or []) if str(q).strip()]
	except Exception:  # noqa: BLE001 - 扩展失败不影响主流程
		variants = []
	return list(dict.fromkeys([fallback] + variants[:3]))


async def research_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	outline = state.get('outline')
	material: dict[str, list[dict[str, Any]]] = {}
	conversations = dict(state.get('conversations', {}))
	budget = rt.config.orchestrator.context_window_tokens
	topic = state.get('topic', '')

	for chapter in outline.chapters:
		query = ' '.join(chapter.keywords[:5]) or chapter.title

		# query-expansion 技能:多查询变体各自检索,结果按 chunk 去重合并(取最高 rerank 分)
		queries = await _expand_queries(rt, topic, chapter, query)
		seen: dict[str, dict[str, Any]] = {}
		for q in queries:
			for r in rt.retriever.retrieve(q, top_k=6):
				prev = seen.get(r.chunk_id)
				if prev is None or r.score > prev.get('score', 0.0):
					seen[r.chunk_id] = _chunk_to_dict(r)
		chunks = sorted(seen.values(), key=lambda d: d.get('score', 0.0), reverse=True)[:8]

		# Web 搜索(arxiv + Semantic Scholar)补足素材:本地库论文不足时避免评审死锁;
		# 外部 API 失败由 search_papers 内部容错跳过,不影响本地检索结果。
		# 只跑原始 query,避免多路变体放大外部 API 调用量。
		from ...search.paper_search import search_papers

		web = await search_papers(query, max_results=rt.config.search.max_results)
		chunks += [_web_to_chunk(c) for c in web]

		material[chapter.chapter_id] = chunks

		# 聊天式调查日志:查询 → 命中素材,超窗压缩
		key = f'research:{chapter.chapter_id}'
		conv = ConversationMemory.from_state(conversations.get(key), budget_tokens=budget)
		conv.add('user', f'Retrieve relevant sources for chapter "{chapter.title}" with queries: {queries}')
		summary = (
			'\n'.join(f"- {c.get('title', '')} ({c.get('year') or '?'})" for c in material[chapter.chapter_id][:10])
			or '(no results)'
		)
		conv.add('assistant', f'Retrieved {len(material[chapter.chapter_id])} sources:\n{summary}')
		if conv.needs_compaction():
			from ...memory.conversation import compact_with_llm

			compacted = await compact_with_llm(rt.registry.cheap(), conv, f'researching {chapter.chapter_id}')
			conv.apply_compaction(compacted)
		conversations[key] = conv.to_state()

	# 标记本代次 research 任务完成
	run_id = state.get('run_id', '')
	for task in rt.task_board.by_status('queued', run_id=run_id):
		if task.kind == 'research':
			rt.task_board.update(task.id, status='approved')
	return {'research_material': material, 'conversations': conversations}
