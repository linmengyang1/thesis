"""高层运行入口:编排整个论文撰写流程(供 CLI 与脚本复用)。"""
from __future__ import annotations

from ..config import get_config
from .orchestrator import build_graph, default_checkpointer
from .runtime import ThesisRuntime


async def run_writer(topic: str, venue: str, thread_id: str = 'default') -> dict:
	"""运行主编排,返回最终状态。"""
	rt = ThesisRuntime()
	checkpointer = None
	try:
		checkpointer = await default_checkpointer()
		app = build_graph(rt, checkpointer=checkpointer)
		return await app.ainvoke(
			{'topic': topic, 'venue': venue},
			config={'configurable': {'thread_id': thread_id}},
		)
	finally:
		rt.close()
		# 关闭 checkpointer 的异步连接,避免事件循环关闭后残留任务报错
		if checkpointer is not None:
			conn = getattr(checkpointer, 'conn', None)
			if conn is not None:
				try:
					await conn.close()
				except Exception:
					pass


def summarize(result: dict) -> str:
	"""把运行结果整理成人类可读摘要。"""
	cfg = get_config()
	lines = ['==== 撰写完成 ====']
	lines.append(f'最终论文: {cfg.output_path / "final_paper.md"}')
	lines.append(f'参考文献: {cfg.output_path / "references.bib"}')
	lines.append(f'一致性检查: {cfg.output_path / "consistency_check.txt"}')
	problems = result.get('consistency_problems') or {}
	if problems:
		lines.append('发现问题:')
		for k, v in problems.items():
			lines.append(f'  - {k}: {v}')
	else:
		lines.append('一致性检查:通过')
	return '\n'.join(lines)
