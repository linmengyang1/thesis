"""交互式聊天界面:论文研究助手,支持随时调用 Web 搜索 / 本地检索工具。

用法:
    thesis-agent chat
    uv run python scripts/chat.py

命令:
    search <关键词>     立即调用 Web 搜索工具(arxiv + Semantic Scholar)
    rag <关键词>        立即调用本地 faiss 检索
    save <n>            把上一条搜索结果中的第 n 条收藏到候选库
    list                查看本次会话已收藏的候选论文
    help / quit         帮助 / 退出
    其他输入            与助手对话;助手在需要时自动调用工具
"""
from __future__ import annotations

import asyncio

from .graph.runtime import ThesisRuntime
from .llm.factory import ModelRegistry
from .llm.tools import TOOL_REGISTRY, run_retrieve_local
from .memory.conversation import ConversationMemory

_CHAT_SYSTEM = """You are a research assistant for engineering thesis writing, helping the user search, investigate and draft.
You have two tools:
- web_search: find academic papers (arxiv / Semantic Scholar).
- retrieve_local: search the user's local paper library (faiss RAG).
Use them whenever the user asks to find literature, review related work, or needs citations.
When citing a paper, report title, authors, year and source. Keep answers concise and useful."""

_HELP = """命令:
  search <关键词>   立即搜索学术论文(arxiv + Semantic Scholar)
  rag <关键词>      立即检索本地论文库(faiss)
  save <n>         收藏上一条搜索结果的第 n 条
  list             查看已收藏候选
  help             显示帮助
  quit             退出"""


def _fmt_candidate(c, n: int) -> str:
	authors = '; '.join(c.authors[:3]) or '未知'
	return f'{n}. [{c.year or "?"}] {c.title}\n   作者: {authors} | 来源: {c.venue or "?"} | DOI: {c.doi or "无"}'


async def run_chat() -> None:
	rt = ThesisRuntime()
	try:
		llm = rt.registry.strong()
		budget = rt.config.orchestrator.context_window_tokens
		conv = ConversationMemory(budget_tokens=budget)
		# 动态工具注册:本地检索需要 runtime
		tool_registry = dict(TOOL_REGISTRY)
		tool_registry['retrieve_local'] = lambda args: run_retrieve_local(args, rt.retriever)

		last_results: list = []
		print('论文研究助手已启动。输入 help 查看命令。')
		while True:
			try:
				line = (await asyncio.to_thread(input, '\n你> ')).strip()
			except (EOFError, KeyboardInterrupt):
				print('\n再见。')
				break
			if not line:
				continue
			low = line.lower()

			if low in ('quit', 'exit', 'q'):
				print('再见。')
				break
			if low in ('help', 'h'):
				print(_HELP)
				continue
			if low.startswith(('search ', '/search ')):
				query = line.split(' ', 1)[1].strip()
				if not query:
					print('用法: search <关键词>')
					continue
				print(f'正在搜索: {query} ...')
				from .search.paper_search import search_papers

				last_results = await search_papers(query, max_results=5)
				if not last_results:
					print('未找到相关论文。')
				else:
					for i, c in enumerate(last_results, 1):
						print(_fmt_candidate(c, i))
					print('提示: save <编号> 收藏候选。')
				continue
			if low.startswith(('rag ', '/rag ')):
				query = line.split(' ', 1)[1].strip()
				if not query:
					print('用法: rag <关键词>')
					continue
				results = rt.retriever.retrieve(query, top_k=5)
				if not results:
					print('本地论文库无相关结果。')
				else:
					for i, r in enumerate(results, 1):
						authors = '; '.join(r.authors[:3]) or '未知'
						print(f'{i}. [{r.year or "?"}] {r.title} | {authors}')
						print(f'   片段: {r.text[:180]}')
				continue
			if low.startswith(('save ', '/save ')):
				try:
					n = int(line.split(' ', 1)[1]) - 1
				except (ValueError, IndexError):
					print('用法: save <编号>')
					continue
				if 0 <= n < len(last_results):
					c = last_results[n]
					rt.store.put('candidates', c.doi or c.title.lower(), c.model_dump())
					print(f'已收藏到候选库: {c.title}')
				else:
					print('编号无效(请先 search,再 save <编号>)。')
				continue
			if low in ('list', 'ls'):
				cands = rt.store.list('candidates')
				if not cands:
					print('候选库为空。')
				else:
					for i, d in enumerate(cands, 1):
						print(f'{i}. {d.get("title", "")} ({d.get("year") or "?"})')
				continue

			# 普通对话:助手可自动调用工具
			conv.add('user', line)
			if conv.needs_compaction():
				from .memory.conversation import compact_with_llm

				summary = await compact_with_llm(llm, conv, 'research assistant chat')
				conv.apply_compaction(summary)
			reply = await llm.acomplete_agent(conv.build(_CHAT_SYSTEM), tool_registry)
			conv.add('assistant', reply)
			print(f'助手> {reply}')
	finally:
		rt.close()


if __name__ == '__main__':
	asyncio.run(run_chat())
