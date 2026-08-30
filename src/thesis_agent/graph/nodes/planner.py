"""Planner 节点:根据主题生成论文大纲,并把大纲展开为任务板。"""
from __future__ import annotations

from typing import Any

from ...models import ChapterOutline, Outline
from ...rag.retrieve import RetrievedChunk
from ..runtime import ThesisRuntime
from ..task_board import TaskItem
from ..utils import extract_list

_PLANNER_SYSTEM = """You are an academic paper planning expert for engineering research theses.
Given a research topic and target venue, design the chapter structure of the paper.
The thesis is an engineering/experimental paper, so include chapters such as Introduction,
Related Work, Method, Experiments, and Conclusion.
Respond with JSON ONLY:
{
  "chapters": [
    {
      "chapter_id": "intro",
      "title": "Introduction",
      "description": "one-paragraph description of what this chapter must contain",
      "kind": "intro|related_work|method|experiments|conclusion|other",
      "deps": ["method"],  // chapter_ids that must be finalized before this one (e.g. experiments depends on method)
      "keywords": ["retrieval", "keywords", "for", "rag"]
    }
  ]
}
Use concise English."""


def build_task_board(outline: Outline, run_id: str) -> list[TaskItem]:
	"""把大纲展开为任务板:每章 research→draft→review,外加 editor/finalize。"""
	tasks: list[TaskItem] = []
	draft_by_chapter: dict[str, str] = {}

	for chapter in outline.chapters:
		research_task = TaskItem(
			run_id=run_id,
			title=f'Research {chapter.title}',
			kind='research',
			assigned_agent='researcher',
			model_tier='medium',
			chapter_id=chapter.chapter_id,
			acceptance=['retrieve >= 5 relevant papers with citation keys'],
		)
		# 章节草稿依赖:本章检索任务 + 大纲里声明的跨章节依赖
		draft_deps = [research_task.id] + [draft_by_chapter[d] for d in chapter.deps if d in draft_by_chapter]
		draft_task = TaskItem(
			run_id=run_id,
			title=f'Draft {chapter.title}',
			kind='draft',
			deps=draft_deps,
			assigned_agent='drafter',
			model_tier='cheap',
			chapter_id=chapter.chapter_id,
			acceptance=['every claim carries a citation key', 'structure matches description'],
		)
		review_task = TaskItem(
			run_id=run_id,
			title=f'Review {chapter.title}',
			kind='review',
			deps=[draft_task.id],
			assigned_agent='critic',
			model_tier='strong',
			chapter_id=chapter.chapter_id,
			acceptance=['verdict issued with actionable issues'],
		)
		tasks += [research_task, draft_task, review_task]
		draft_by_chapter[chapter.chapter_id] = draft_task.id

	editor_task = TaskItem(
		run_id=run_id,
		title='Editor: polish + citations + format',
		kind='format',
		deps=[draft_by_chapter[c.chapter_id] for c in outline.chapters],
		assigned_agent='editor',
		model_tier='medium',
	)
	finalize_task = TaskItem(
		run_id=run_id,
		title='Final consistency check + output',
		kind='format',
		deps=[editor_task.id],
		assigned_agent='editor',
		model_tier='medium',
	)
	tasks += [editor_task, finalize_task]
	return tasks


async def planner_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	topic = state.get('topic', '')
	venue = state.get('venue', '')
	llm = rt.registry.strong()
	user_prompt = f'Research topic: {topic}\nTarget venue: {venue or "not specified"}\n\nDesign the chapter structure.'

	resp = await llm.acomplete([{'role': 'system', 'content': _PLANNER_SYSTEM}, {'role': 'user', 'content': user_prompt}])
	chapters = extract_list(resp, 'chapters')
	parsed: list[ChapterOutline] = []
	for c in chapters[:8]:
		parsed.append(
			ChapterOutline(
				chapter_id=str(c.get('chapter_id', '')),
				title=str(c.get('title', '')),
				description=str(c.get('description', '')),
				kind=str(c.get('kind', 'other')),
				deps=[str(d) for d in (c.get('deps') or [])],
				keywords=[str(k) for k in (c.get('keywords') or [])],
			)
		)
	if not parsed:
		# 没有章节时按工科论文默认结构兜底,保证流程可继续
		parsed = [
			ChapterOutline(chapter_id='intro', title='Introduction', description='Problem statement, motivation, contributions', kind='intro', keywords=[topic]),
			ChapterOutline(chapter_id='method', title='Method', description='Proposed method and design', kind='method', keywords=[topic]),
			ChapterOutline(chapter_id='experiments', title='Experiments', description='Setup, results, analysis', kind='experiments', deps=['method'], keywords=[topic]),
			ChapterOutline(chapter_id='conclusion', title='Conclusion', description='Summary and future work', kind='conclusion', keywords=[topic]),
		]
	outline = Outline(topic=topic, venue=venue, chapters=parsed)
	# 生成本轮代际标识,任务板按 run_id 隔离,互不干扰
	import uuid

	run_id = str(uuid.uuid4())[:8]
	tasks = build_task_board(outline, run_id)
	for t in tasks:
		rt.task_board.add(t)
	return {'outline': outline, 'run_id': run_id}
