"""主编排:langgraph StateGraph 装配。

流程:
  planner → research → dispatch
  dispatch --(条件边扇出)--> draft_node(并行) --> review_gate
  review_gate --(条件边扇出)--> review_node(并行) --> review_router
  review_router --(条件)--> dispatch(修订/推进依赖) | editor(含超限阻塞:照常产出并标注问题)
  editor → finalize → END
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from ..config import get_config
from .nodes.dispatch import dispatch_edge, dispatch_node
from .nodes.draft import draft_node
from .nodes.editor import editor_node
from .nodes.finalize import finalize_node
from .nodes.planner import planner_node
from .nodes.research import research_node
from .nodes.review import (
	review_dispatch_edge,
	review_gate_node,
	review_node,
	review_router_node,
	route_after_review,
)
from .runtime import ThesisRuntime
from .state import ThesisState


def _bind(rt: ThesisRuntime, fn: Callable) -> Callable:
	"""把 runtime 绑定到异步节点函数。"""

	async def wrapped(state: dict[str, Any]):
		return await fn(state, rt)

	return wrapped


def _bind_edge(rt: ThesisRuntime, fn: Callable) -> Callable:
	"""把 runtime 绑定到同步条件边函数。"""

	def wrapped(state: dict[str, Any]):
		return fn(state, rt)

	return wrapped


def build_graph(rt: ThesisRuntime, checkpointer=None):
	graph = StateGraph(ThesisState)

	graph.add_node('planner', _bind(rt, planner_node))
	graph.add_node('research', _bind(rt, research_node))
	graph.add_node('dispatch', _bind(rt, dispatch_node))
	graph.add_node('draft_node', _bind(rt, draft_node))
	graph.add_node('review_gate', _bind(rt, review_gate_node))
	graph.add_node('review_node', _bind(rt, review_node))
	graph.add_node('review_router', _bind(rt, review_router_node))
	graph.add_node('editor', _bind(rt, editor_node))
	graph.add_node('finalize', _bind(rt, finalize_node))

	graph.add_edge(START, 'planner')
	graph.add_edge('planner', 'research')
	graph.add_edge('research', 'dispatch')
	# 条件边返回 Send 列表时扇出草稿;否则路由到 review_gate
	graph.add_conditional_edges('dispatch', _bind_edge(rt, dispatch_edge))
	# 所有草稿完成后汇聚到 review_gate,避免并行分支重复派发评审
	graph.add_edge('draft_node', 'review_gate')
	graph.add_conditional_edges('review_gate', _bind_edge(rt, review_dispatch_edge))
	graph.add_edge('review_node', 'review_router')
	graph.add_conditional_edges(
		'review_router',
		_bind_edge(rt, route_after_review),
		{'dispatch': 'dispatch', 'editor': 'editor'},
	)
	graph.add_edge('editor', 'finalize')
	graph.add_edge('finalize', END)

	return graph.compile(checkpointer=checkpointer)


async def default_checkpointer():
	"""异步 SQLite checkpointer(短期记忆的持久化底座,配合 ainvoke 使用)。"""
	import aiosqlite
	from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

	cfg = get_config()
	db_file = cfg.db_file.parent / 'checkpoints.sqlite'
	db_file.parent.mkdir(parents=True, exist_ok=True)
	conn = await aiosqlite.connect(str(db_file))
	return AsyncSqliteSaver(conn)
