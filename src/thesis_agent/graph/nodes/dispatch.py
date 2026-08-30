"""dispatch:草稿任务的扇出。

langgraph 1.x 中 Send 扇出必须由条件边返回 Send 列表,节点体不能返回。
- dispatch_node:节点体,仅返回空更新(入口占位)
- dispatch_edge:条件边,返回 [Send('draft_node', ...)] 或 'review_gate'
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from ..runtime import ThesisRuntime


def _find_chapter(outline, chapter_id):
	for c in outline.chapters:
		if c.chapter_id == chapter_id:
			return c
	return None


async def dispatch_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	"""入口占位节点,实际分发在 dispatch_edge 条件边中完成。"""
	return {}


def dispatch_edge(state: dict[str, Any], rt: ThesisRuntime) -> Any:
	"""条件边:把就绪的草稿任务(含修订)扇出为 Send,没有则推进到 review_gate。"""
	outline = state.get('outline')
	if outline is None:
		return 'review_gate'
	run_id = state.get('run_id', '')
	research = state.get('research_material', {})
	revision_notes = state.get('revision_notes', {})
	sends: list[Send] = []
	base = {
		'topic': state.get('topic', ''),
		'venue': state.get('venue', ''),
		'conversations': state.get('conversations', {}),  # 聊天式记忆传给子节点
	}

	# 初次草稿:queued 且依赖满足
	for task in rt.task_board.ready_tasks(run_id=run_id):
		if task.kind != 'draft':
			continue
		rt.task_board.update(task.id, status='in_progress')
		chapter = _find_chapter(outline, task.chapter_id)
		sends.append(
			Send(
				'draft_node',
				{
					**base,
					'chapter': chapter.model_dump() if chapter else None,
					'research': research.get(task.chapter_id, []),
					'revision': None,
					'task_id': task.id,
					'chapter_id': task.chapter_id,
				},
			)
		)

	# 修订轮次:in_revision 的草稿任务,附带评审意见
	for task in rt.task_board.by_status('in_revision', run_id=run_id):
		if task.kind != 'draft':
			continue
		chapter = _find_chapter(outline, task.chapter_id)
		sends.append(
			Send(
				'draft_node',
				{
					**base,
					'chapter': chapter.model_dump() if chapter else None,
					'research': research.get(task.chapter_id, []),
					'revision': revision_notes.get(task.chapter_id, ''),
					'task_id': task.id,
					'chapter_id': task.chapter_id,
				},
			)
		)

	return sends if sends else 'review_gate'
