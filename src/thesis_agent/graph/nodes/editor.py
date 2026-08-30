"""Editor 节点:整合所有已通过章节 → 构建引用库 → LLM 润色统一 → 生成 BibTeX。"""
from __future__ import annotations

import re
from typing import Any

from ...citations.bibtex import citations_to_bibtex, make_citation_key
from ...models import Citation
from ..runtime import ThesisRuntime

_EDITOR_SYSTEM = """You are a paper editor for an engineering thesis, writing in English.
Combine the given chapter texts into ONE coherent full paper.
Rules:
1. Keep every section and its content; improve language, flow, and consistency.
2. Do NOT remove or add citations that were in the drafts.
3. Ensure citation markers [Author et al., YYYY] appear wherever a claim needs a source.
4. Add an "Abstract" section at the top of the paper.
5. Do NOT add a "References" section — the final bibliography will be appended automatically.
Respond with the full paper text only."""


# 原始 arXiv 头部串(如 "arXiv:2407.11699v1 [cs.CV] 16 Jul 2024")被误当标题,无法形成可引用条目
_ARXIV_STRING_TITLE_RE = re.compile(r'^arxiv:\s?\d{4}\.\d{5}', re.IGNORECASE)


def build_citations(research_material: dict[str, list[dict[str, Any]]]) -> list[Citation]:
	"""从检索素材去重构建 Citation 列表(附 citation key 与 paper_id 溯源)。

	跳过无标题或原始 arXiv 头串的 chunk,避免把占位符/损坏元数据写进引用库。
	"""
	seen: dict[str, Citation] = {}
	for chunks in research_material.values():
		for c in chunks:
			paper_id = c.get('paper_id', '')
			title = (c.get('title') or '').strip()
			if not paper_id or paper_id in seen:
				continue
			if not title or _ARXIV_STRING_TITLE_RE.match(title):
				continue
			seen[paper_id] = Citation(
				key=make_citation_key(
					Citation(
						title=c.get('title', ''),
						authors=c.get('authors', []),
						year=c.get('year'),
						venue=c.get('venue', ''),
						doi=c.get('doi', ''),
					)
				),
				title=c.get('title', ''),
				authors=c.get('authors', []),
				year=c.get('year'),
				venue=c.get('venue', ''),
				doi=c.get('doi', ''),
				paper_id=paper_id,
			)
	return list(seen.values())


async def _filter_topic_relevant(rt: ThesisRuntime, topic: str, citations: list[Citation]) -> list[Citation]:
	"""用 LLM 过滤与论文主题明显无关的候选引用,防止离题论文污染引用库。

	只删"明显无关"的条目;LLM 输出异常时保留全部(宁多勿删,避免误删相关文献)。
	"""
	if not topic or not citations:
		return citations
	listing = '\n'.join(
		f'{i}. [{c.key}] {c.title} | {", ".join(c.authors[:2])} | {c.year or "?"}'
		for i, c in enumerate(citations)
	)
	user_prompt = (
		f'Paper topic: {topic}\n\n'
		f'Candidate references:\n{listing}\n\n'
		f'Return JSON ONLY: {{"keep": [0-based indices of references to KEEP]}}.\n'
		f'Keep a reference only if it is clearly relevant to the paper topic. '
		f'Discard references that are clearly off-topic or irrelevant. When in doubt, keep it.'
	)
	llm = rt.registry.cheap()
	resp = await llm.acomplete(
		[
			{'role': 'system', 'content': 'You are a research assistant filtering a reference list by topic relevance.'},
			{'role': 'user', 'content': user_prompt},
		]
	)
	from ..utils import extract_json

	data = extract_json(resp) or {}
	keep = sorted({i for i in data.get('keep', []) if isinstance(i, int) and 0 <= i < len(citations)})
	if not keep:
		return citations
	return [citations[i] for i in keep]


async def editor_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	drafts = state.get('chapter_drafts', {})
	outline = state.get('outline')
	# 按大纲顺序组织章节
	ordered = [drafts[c.chapter_id] for c in outline.chapters if c.chapter_id in drafts]
	combined = '\n\n'.join(ordered)

	citations = build_citations(state.get('research_material', {}))
	# 主题相关性过滤:剔除与论文主题明显无关的候选,避免离题论文进引用库
	citations = await _filter_topic_relevant(rt, state.get('topic', ''), citations)
	# 让编辑引用库内容给 LLM,保证 citation key 一致性
	bib_text = citations_to_bibtex(citations)

	# 预算截断:引用只保留关键字段(够了),正文按剩余预算截断
	budget = rt.config.orchestrator.context_window_tokens
	from ...memory.conversation import truncate_parts

	bib_budget = min(2000, int(budget * 0.15))
	bib_trimmed = truncate_parts([bib_text], bib_budget)
	body_budget = max(1, int(budget * 0.7))
	body_trimmed = truncate_parts([combined], body_budget)

	llm = rt.registry.medium()
	user_prompt = (
		f'Paper topic: {state.get("topic", "")}\n'
		f'Available references (BibTeX):\n{bib_trimmed}\n\n'
		f'Chapter texts to combine:\n{body_trimmed}\n\n'
		f'Produce the full paper.'
	)
	full_text = await llm.acomplete([{'role': 'system', 'content': _EDITOR_SYSTEM}, {'role': 'user', 'content': user_prompt}])

	# 引用溯源一致化:把作者-年份标记改写为 citation key,确保与 BibTeX 一一对应
	from ...citations.bibtex import rewrite_author_year_citations

	rewritten, warnings = rewrite_author_year_citations(full_text, citations)
	full_text = rewritten

	# 标记本代次 editor 任务完成
	run_id = state.get('run_id', '')
	for task in rt.task_board.by_status('queued', run_id=run_id):
		if task.kind == 'format':
			rt.task_board.update(task.id, status='approved')

	# 写产物到输出目录
	out_dir = rt.config.output_path
	out_dir.mkdir(parents=True, exist_ok=True)
	(out_dir / 'paper.md').write_text(full_text, encoding='utf-8')
	(out_dir / 'references.bib').write_text(bib_text, encoding='utf-8')

	updates: dict[str, Any] = {'citations': citations, 'final_paper': full_text, 'bibtex_text': bib_text}
	# 未匹配的作者-年份引用(草稿里引用了素材库外/被过滤掉的论文)如实上报,不再静默丢弃
	if warnings:
		updates['consistency_problems'] = {f'editor-{i}': w for i, w in enumerate(warnings)}
	return updates
