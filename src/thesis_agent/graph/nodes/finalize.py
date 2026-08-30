"""Finalize 节点:全文一致性校验(引用↔BibTeX)→ 组装最终产物 → 写入输出目录。"""
from __future__ import annotations

from typing import Any

from ...citations.bibtex import validate_citations_in_text
from ..runtime import ThesisRuntime

_FINAL_CHECK_SYSTEM = """You are a structural consistency checker for an engineering thesis.
Verify the paper: (1) an Abstract section exists; (2) the body is organized into coherent chapters with no obvious truncation or duplication.
Do NOT verify individual citations or the References section — rule-based checkers already handle those.
Reply with JSON ONLY: {"ok": true or false, "problems": ["..."]}"""


async def finalize_node(state: dict[str, Any], rt: ThesisRuntime) -> dict[str, Any]:
	paper = state.get('final_paper', '')
	bib_text = state.get('bibtex_text', '')
	citations = state.get('citations', [])

	# 1) 规则校验:内嵌引用 ↔ BibTeX;先保留上游问题(如评审阻塞升级),再叠加本轮结果
	problems = dict(state.get('consistency_problems') or {})
	problems.update(validate_citations_in_text(paper, citations))

	# 2) LLM 结构一致性检查:只查 Abstract/章节完整性;
	#    引用↔BibTeX 已由上面的规则校验用全文核对,不再让 LLM 判引用(避免引用被截断导致的系统性误报)
	budget = rt.config.orchestrator.context_window_tokens
	from ...memory.conversation import truncate_parts

	paper_for_llm = truncate_parts([paper], min(3000, int(budget * 0.5)))
	llm = rt.registry.strong()
	resp = await llm.acomplete(
		[
			{'role': 'system', 'content': _FINAL_CHECK_SYSTEM},
			{'role': 'user', 'content': f'Paper:\n{paper_for_llm}\n\nVerdict:'},
		]
	)
	from ..utils import extract_json

	data = extract_json(resp) or {}
	if not data.get('ok'):
		problems.update({f'llm-{i}': str(p) for i, p in enumerate(data.get('problems') or [])})

	# 3) 组装最终产物(含标题、摘要占位、参考文献)
	title = state.get('topic', 'Untitled Thesis')
	references_section = f'\n## References\n\n{bib_text}\n' if bib_text else ''
	final = f'# {title}\n\n{paper}\n{references_section}'

	out_dir = rt.config.output_path
	out_dir.mkdir(parents=True, exist_ok=True)
	(out_dir / 'final_paper.md').write_text(final, encoding='utf-8')
	if bib_text:
		(out_dir / 'references.bib').write_text(bib_text, encoding='utf-8')
	(out_dir / 'consistency_check.txt').write_text(
		'OK\n' if not problems else 'PROBLEMS:\n' + '\n'.join(f'- {k}: {v}' for k, v in problems.items()),
		encoding='utf-8',
	)

	# 标记本代次 finalize 任务完成
	run_id = state.get('run_id', '')
	for task in rt.task_board.by_status('queued', run_id=run_id):
		if task.kind == 'format':
			rt.task_board.update(task.id, status='approved')

	return {'final_paper': final, 'consistency_problems': problems}
