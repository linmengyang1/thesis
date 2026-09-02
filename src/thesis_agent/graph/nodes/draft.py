"""Drafter 节点:为单个章节撰写初稿,维护聊天式对话记忆。

对话式撰写:同一章节的初稿与修订轮次共用一条对话(conversation_key=draft:{chapter_id}),
修订意见作为新的 user 消息接续;历史超窗时压缩为 compacted_memory。
防幻觉规则:每个论断必须挂检索素材中的文献引用,用 [Author et al., YYYY] 标注。
"""
from __future__ import annotations

from typing import Any

from ...memory.conversation import (
	ConversationMemory,
	compact_with_llm,
	estimate_tokens,
	truncate_text,
)
from ...citations.bibtex import _first_author_surname
from ..runtime import ThesisRuntime

_DRAFTER_SYSTEM = """You are a research paper drafter for an engineering thesis, writing in English.

Rules:
1. Write one academic chapter in Markdown.
2. Every factual claim MUST be backed by a source from the provided research material. Cite as [First author surname et al., YYYY] at the end of the sentence.
3. ONLY cite sources listed in the "Citation whitelist" in the prompt. Inventing a citation that is not in the whitelist is forbidden — if a claim has no matching whitelisted source, drop the claim or leave it uncited.
4. Do NOT invent citations, numbers, or results that are not in the provided material.
5. Follow the chapter description precisely. Use section headings (## ...).
6. Output the chapter text only, no preamble."""

# 单块素材注入上限与总预算比例
_MAX_RESEARCH_CHARS_PER_CHUNK = 600
_RESEARCH_BUDGET_RATIO = 0.6


def _format_whitelist(chunks: list[dict[str, Any]]) -> str:
	"""生成引用白名单:素材库中真实存在的论文(首作者姓氏+年份),Drafter 只能引用这些。

	格式与正文引用标记 [Author et al., YYYY] 对齐,去重并截断标题保持短小。
	"""
	lines = []
	seen: set[tuple[str, str]] = set()
	for c in chunks:
		authors = c.get('authors') or []
		if not authors:
			continue
		surname = _first_author_surname(authors[0])
		year = str(c.get('year') or '?')
		if (surname, year) in seen:
			continue
		seen.add((surname, year))
		title = (c.get('title') or '')[:70]
		lines.append(f'- [{surname} et al., {year}] | {title}')
	return '\n'.join(lines)


def _format_research(chunks: list[dict[str, Any]], budget_chars: int) -> str:
	"""按预算组装检索素材,超限优先保留高分 chunk。"""
	lines = []
	used = 0
	for c in chunks:
		authors = ', '.join(c.get('authors', [])[:2])
		year = c.get('year') or '?'
		text = truncate_text(c.get('text', ''), min(_MAX_RESEARCH_CHARS_PER_CHUNK, max(0, budget_chars - used)))
		line = f"- [{authors}, {year}] ({c.get('title', '')}) {text}"
		used += estimate_tokens(line)
		lines.append(line)
		if used >= budget_chars:
			break
	return '\n'.join(lines) if lines else '(no research material available)'


async def draft_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	chapter = state.get('chapter') or {}
	research = state.get('research', [])
	revision = state.get('revision') or ''
	chapter_id = state.get('chapter_id', '')
	topic = state.get('topic', '')
	venue = state.get('venue', '')
	key = f'draft:{chapter_id}'

	budget = rt.config.orchestrator.context_window_tokens
	conv = ConversationMemory.from_state(state.get('conversations', {}).get(key), budget_tokens=budget)

	# 首轮:组装撰写上下文;修订轮:把评审意见作为新的 user 消息接续对话
	if not conv.messages:
		# 自进化记忆:注入同类章节的历史写作经验与高频问题
		from ...memory.experience import get_experiences, get_recurring_issues

		kind = chapter.get('kind', 'other')
		experiences = get_experiences(rt.store, kind)
		recurring = get_recurring_issues(rt.store, kind)
		memory_block = ''
		if experiences:
			memory_block += 'Lessons from previous similar chapters (avoid repeating):\n' + '\n'.join(
				f'- {e}' for e in experiences
			) + '\n'
		if recurring:
			memory_block += 'Recurring problems to watch out for:\n' + '\n'.join(f'- {r}' for r in recurring) + '\n'

		research_budget_chars = int(budget * _RESEARCH_BUDGET_RATIO * 4)
		whitelist = _format_whitelist(research)
		# venue-style 技能:按目标会议/期刊注入写作规范
		from ...skills.venue import load_venue_guidelines

		venue_block = load_venue_guidelines(venue)
		user_prompt = (
			f'Paper topic: {topic}\n'
			f'Target venue: {venue or "not specified"}\n\n'
			f'Venue style guidelines (follow strictly):\n{venue_block}\n\n'
			f'Chapter: {chapter.get("title", "")}\n'
			f'Chapter description: {chapter.get("description", "")}\n\n'
			f'Research material:\n{_format_research(research, research_budget_chars)}\n\n'
			f'Citation whitelist (you may ONLY cite these sources):\n{whitelist or "(none)"}\n\n'
		)
		if memory_block:
			user_prompt += f'{memory_block}\n'
		user_prompt += '\nWrite the chapter now.'
		conv.add('user', user_prompt)
	else:
		conv.add('user', f'Reviewer feedback to address:\n{revision}\n\nRevise the chapter accordingly.')

	# 历史超窗:先压缩旧轮次
	if conv.needs_compaction():
		summary = await compact_with_llm(rt.registry.cheap(), conv, f'drafting chapter {chapter_id}')
		conv.apply_compaction(summary)

	llm = rt.registry.cheap()
	messages = conv.build(_DRAFTER_SYSTEM)
	text = await llm.acomplete(messages)
	conv.add('assistant', text)

	# 更新任务状态:草稿完成,等待评审
	rt.task_board.update(state.get('task_id', chapter_id), status='needs_review')
	return {'chapter_drafts': {chapter_id: text}, 'conversations': {key: conv.to_state()}}
