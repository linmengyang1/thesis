"""图状态定义(langgraph StateGraph 的 shared state)。

短期记忆的持久化底座是 langgraph Checkpointer;本 TypedDict 定义
每一轮共享、可被任何节点读写的状态结构。注意:长期跨会话信息放
memory/store.py 的命名空间,不放这里。

并行分支(Send fan-out)写入 dict/list 时使用 Annotated reducer
合并,避免 last-write-wins 互相覆盖。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..models import Citation, FeedbackEntry, Outline, ReviewVerdict


def _merge_dict(a: dict, b: dict) -> dict:
	"""合并两个 dict(用于并行分支写同一状态键)。"""
	merged = dict(a)
	merged.update(b)
	return merged


class ThesisState(TypedDict, total=False):
	topic: str
	venue: str
	run_id: str  # 本轮运行的代际标识(任务板按它隔离)
	outline: Outline
	# 任务板不放在共享状态:以 SQLite 为真源(rt.task_board),串行节点直接读写
	# 每章检索素材 {chapter_id: list[RetrievedChunk 字典]}
	research_material: Annotated[dict, _merge_dict]
	# 每章草稿 {chapter_id: 文本},并行分支各写各的 key
	chapter_drafts: Annotated[dict, _merge_dict]
	# 每章评审结论 {chapter_id: ReviewVerdict}
	review_results: Annotated[dict, _merge_dict]
	# 修订意见 {chapter_id: 评审给出的修改要求}
	revision_notes: Annotated[dict, _merge_dict]
	citations: Annotated[list[Citation], operator.add]
	feedback_log: Annotated[list[FeedbackEntry], operator.add]
	# 子 agent 聊天式记忆 {conversation_key: ConversationMemory.to_state()},并行分支各写各的 key
	conversations: Annotated[dict, _merge_dict]
	iteration: int  # 评审-修订轮次
	# 最终产物
	final_paper: str
	bibtex_text: str
	consistency_problems: dict[str, str]
