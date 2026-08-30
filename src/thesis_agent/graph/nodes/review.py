"""评审相关节点:review_dispatch(扇出评审)→ review(单章评审)→ review_router(路由)。

评审是质量闭环的核心:verdict 不通过 → 章节回炉修订;通过 → 进入下一环节。
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from ...memory.conversation import ConversationMemory
from ...models import ReviewVerdict
from ..runtime import ThesisRuntime
from .draft import _format_whitelist

_CRITIC_SYSTEM = """You are a strict academic reviewer for an engineering thesis, writing in English.
Evaluate the chapter against its description and the acceptance criteria.
Specifically check: (1) every citation marker [Author et al., YYYY] in the draft appears in the provided citation whitelist — flag any invented citation; (2) claims are grounded in the provided material; (3) structure matches the chapter description.
Respond with JSON ONLY:
{
  "approved": true or false,
  "score": 0-10,
  "issues": ["specific problem 1", "specific problem 2"],
  "revision_suggestion": "actionable guidance for the drafter (empty if approved)"
}
Be specific and harsh about ungrounded claims, missing citations, and structural gaps."""


def _find_chapter(outline, chapter_id):
	for c in outline.chapters:
		if c.chapter_id == chapter_id:
			return c
	return None


async def review_gate_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	"""汇聚节点:等所有草稿完成后再统一派发评审(避免并行分支重复派发)。"""
	return {}


def review_dispatch_edge(state: dict[str, Any], rt: ThesisRuntime) -> Any:
	"""条件边:为所有 needs_review 的章节草稿扇出评审。"""
	drafts = state.get('chapter_drafts', {})
	run_id = state.get('run_id', '')
	sends: list[Send] = []
	for draft_task in rt.task_board.all(run_id=run_id):
		if draft_task.kind != 'draft' or draft_task.status != 'needs_review':
			continue
		if draft_task.chapter_id not in drafts:
			continue
		review_task = next(
			(t for t in rt.task_board.all(run_id=run_id) if t.kind == 'review' and t.chapter_id == draft_task.chapter_id),
			None,
		)
		if review_task is None:
			continue
		rt.task_board.update(review_task.id, status='in_progress')
		chapter = _find_chapter(state.get('outline'), draft_task.chapter_id)
		material = state.get('research_material', {}).get(draft_task.chapter_id, [])
		sends.append(
			Send(
				'review_node',
				{
					'chapter_id': draft_task.chapter_id,
					'task_id': review_task.id,
					'draft_text': drafts[draft_task.chapter_id],
					'chapter': chapter.model_dump() if chapter else None,
					'material': material,  # 该章节检索素材,供评审核对引用白名单
					'conversations': state.get('conversations', {}),  # 聊天式记忆传给评审
				},
			)
		)
	return sends if sends else 'review_router'


async def review_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	"""对单个章节草稿做评审,产出结构化结论;评审轮次用聊天式记忆接续。"""
	chapter = state.get('chapter') or {}
	draft_text = state.get('draft_text', '')
	chapter_id = state['chapter_id']
	key = f'review:{chapter_id}'

	budget = rt.config.orchestrator.context_window_tokens
	conv = ConversationMemory.from_state(state.get('conversations', {}).get(key), budget_tokens=budget)
	acceptance = '\n'.join(f'- {a}' for a in chapter.get('acceptance', []) or [])
	whitelist = _format_whitelist(state.get('material', []))
	conv.add(
		'user',
		f'Chapter: {chapter.get("title", "")}\n'
		f'Chapter description: {chapter.get("description", "")}\n'
		f'Acceptance criteria:\n{acceptance or "- coherent structure and grounded claims"}\n\n'
		f'Citation whitelist (every citation marker in the draft MUST appear here):\n{whitelist or "(no material provided)"}\n\n'
		f'Chapter draft:\n{draft_text}\n\n'
		f'Verdict:',
	)
	if conv.needs_compaction():
		from ...memory.conversation import compact_with_llm

		summary = await compact_with_llm(rt.registry.cheap(), conv, f'reviewing chapter {chapter_id}')
		conv.apply_compaction(summary)

	llm = rt.registry.strong()
	resp = await llm.acomplete(conv.build(_CRITIC_SYSTEM))
	conv.add('assistant', resp)

	from ..utils import extract_json

	data = extract_json(resp) or {}
	verdict = ReviewVerdict(
		approved=bool(data.get('approved')),
		score=int(data.get('score', 0)),
		issues=[str(i) for i in (data.get('issues') or [])],
		revision_suggestion=str(data.get('revision_suggestion', '')),
	)
	return {'review_results': {chapter_id: verdict.model_dump()}, 'conversations': {key: conv.to_state()}}


async def review_router_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	"""根据评审结果更新任务板:approved 收官,否则回炉(in_revision),超限阻塞。"""
	verdicts = state.get('review_results', {})
	run_id = state.get('run_id', '')
	max_rounds = rt.config.orchestrator.max_review_rounds
	revision_notes: dict[str, str] = {}
	max_iter = state.get('iteration', 0)
	blocked_titles: list[str] = []

	from ...memory.experience import record_feedback_adoption, record_review_outcome

	for chapter_id, verdict_dict in verdicts.items():
		verdict = ReviewVerdict.model_validate(verdict_dict)
		draft_task = next(
			(t for t in rt.task_board.all(run_id=run_id) if t.kind == 'draft' and t.chapter_id == chapter_id), None
		)
		# 自进化记忆:沉淀本次评审教训(按章节类型)
		chapter = _find_chapter(state.get('outline'), chapter_id)
		record_review_outcome(
			rt.store,
			chapter.kind if chapter else 'other',
			verdict.approved,
			verdict.issues,
			draft_task.revision_count + 1 if draft_task else 0,
		)
		# 反馈采纳追踪:评审通过视为本轮问题已采纳,需修订则记为未采纳
		record_feedback_adoption(rt.store, chapter_id, verdict.issues, adopted=verdict.approved)
		if verdict.approved:
			if draft_task:
				rt.task_board.update(draft_task.id, status='approved')
			revision_notes[chapter_id] = ''
			continue
		revision_notes[chapter_id] = verdict.revision_suggestion or '\n'.join(verdict.issues)
		if draft_task is None:
			continue
		rounds = draft_task.revision_count + 1
		max_iter = max(max_iter, rounds)
		if rounds > max_rounds:
			rt.task_board.update(
				draft_task.id,
				status='blocked',
				revision_count=rounds,
				note=f'评审-修订超过 {max_rounds} 轮,需人工介入',
			)
			blocked_titles.append(draft_task.title)
		else:
			rt.task_board.update(draft_task.id, status='in_revision', revision_count=rounds)
	# 通过章节的评审任务收官
	for review_task in rt.task_board.by_status('in_progress', run_id=run_id):
		if review_task.kind == 'review':
			rt.task_board.update(review_task.id, status='approved')

	updates: dict[str, Any] = {'revision_notes': revision_notes, 'iteration': max_iter}
	if blocked_titles:
		# 阻塞章节不中止整篇产出,但把问题挂到 consistency_problems 供最终报告标注
		updates['consistency_problems'] = {
			'escalated': f'评审-修订超限阻塞: {"、".join(blocked_titles)}(需人工介入)'
		}
	return updates


def route_after_review(state: dict[str, Any], rt: ThesisRuntime) -> str:
	"""条件路由:阻塞→editor(产出部分产物并标注问题)/ 全部通过→editor / 否则→dispatch(修订或推进依赖)。"""
	run_id = state.get('run_id', '')
	draft_tasks = [t for t in rt.task_board.all(run_id=run_id) if t.kind == 'draft']
	if not draft_tasks:
		return 'editor'
	statuses = {t.status for t in draft_tasks}
	if 'blocked' in statuses:
		return 'editor'
	if statuses <= {'approved', 'merged'}:
		return 'editor'
	return 'dispatch'
