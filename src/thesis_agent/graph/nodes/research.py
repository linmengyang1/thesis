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


async def research_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	outline = state.get('outline')
	material: dict[str, list[dict[str, Any]]] = {}
	conversations = dict(state.get('conversations', {}))
	budget = rt.config.orchestrator.context_window_tokens

	for chapter in outline.chapters:
		query = ' '.join(chapter.keywords[:5]) or chapter.title
		results = rt.retriever.retrieve(query, top_k=8)
		chunks = [_chunk_to_dict(r) for r in results]

		# Web 搜索(arxiv + Semantic Scholar)补足素材:本地库论文不足时避免评审死锁;
		# 外部 API 失败由 search_papers 内部容错跳过,不影响本地检索结果。
		from ...search.paper_search import search_papers

		web = await search_papers(query, max_results=rt.config.search.max_results)
		chunks += [_web_to_chunk(c) for c in web]

		material[chapter.chapter_id] = chunks

		# 聊天式调查日志:查询 → 命中素材,超窗压缩
		key = f'research:{chapter.chapter_id}'
		conv = ConversationMemory.from_state(conversations.get(key), budget_tokens=budget)
		conv.add('user', f'Retrieve relevant sources for chapter "{chapter.title}" with query: {query}')
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
