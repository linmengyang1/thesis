"""Web 聊天后端:SSE 流式对话 + 候选库 API。

与 dashboard.py 同属 FastAPI 服务;前端(Vue)通过 POST /api/chat/stream 发起对话,
工具调用与最终回复以 SSE 事件流式推送。会话暂存内存(一期),候选收藏写入 MemoryStore。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .graph.runtime import ThesisRuntime
from .memory.conversation import ConversationMemory

router = APIRouter(prefix='/api')

_CHAT_SYSTEM = """You are a research assistant for engineering thesis writing, helping the user search, investigate and draft.
You have two tools:
- web_search: find academic papers (arxiv / Semantic Scholar).
- retrieve_local: search the user's local paper library (faiss RAG).
Use them whenever the user asks to find literature, review related work, or needs citations.
When citing a paper, report title, authors, year and source. Keep answers concise and useful."""

# 会话内存:{session_id: ConversationMemory}(一期内存态;二期可持久化)
_SESSIONS: dict[str, ConversationMemory] = {}
# 运行时单例:聊天与撰写共用同一套服务(LLM/检索/记忆)
_rt: ThesisRuntime | None = None


def _get_runtime() -> ThesisRuntime:
	global _rt
	if _rt is None:
		_rt = ThesisRuntime()
	return _rt


def _fmt_web_text(citations: list) -> str:
	"""把 web 搜索结果格式化成回填给 LLM 的文本。"""
	lines = [f'找到 {len(citations)} 条结果:']
	for i, c in enumerate(citations, 1):
		authors = '; '.join(c.authors[:3]) or '未知作者'
		lines.append(
			f'{i}. [{c.year or "?"}] {c.title}\n'
			f'   作者: {authors} | 来源: {c.venue or "?"} | DOI: {c.doi or "无"}'
		)
	return '\n'.join(lines)


def _fmt_local_text(results: list) -> str:
	"""把本地检索结果格式化成回填给 LLM 的文本。"""
	lines = [f'本地检索到 {len(results)} 条相关内容:']
	for i, r in enumerate(results, 1):
		authors = '; '.join(r.authors[:3]) or '未知'
		lines.append(
			f'{i}. [{r.year or "?"}] {r.title}\n'
			f'   作者: {authors} | 来源: {r.venue or "?"}\n'
			f'   片段: {r.text[:200]}'
		)
	return '\n'.join(lines)


def _local_to_citation(r) -> dict[str, Any]:
	"""本地检索 chunk → 结构化引用(供前端收藏)。"""
	return {
		'title': r.title,
		'authors': r.authors,
		'year': r.year,
		'venue': r.venue,
		'doi': r.doi,
		'paper_id': r.paper_id,
	}


async def _build_tool_registry(rt: ThesisRuntime, on_event):
	"""构造带事件通知的工具注册表:工具执行时同时推送结构化 citations 事件。"""
	from .search.paper_search import search_papers

	async def web_search(args: dict) -> str:
		query = str(args.get('query', '')).strip()
		top_k = max(1, int(args.get('top_k', 5) or 5))
		if not query:
			return '缺少检索关键词 query。'
		citations = await search_papers(query, max_results=top_k)
		text = _fmt_web_text(citations) if citations else f'未找到与 "{query}" 相关的论文。'
		await on_event(
			{
				'type': 'tool_result',
				'name': 'web_search',
				'content': text,
				'citations': [c.model_dump() for c in citations],
			}
		)
		return text

	async def retrieve_local(args: dict) -> str:
		query = str(args.get('query', '')).strip()
		top_k = max(1, int(args.get('top_k', 5) or 5))
		if not query:
			return '缺少检索关键词 query。'
		results = rt.retriever.retrieve(query, top_k=top_k)
		text = _fmt_local_text(results) if results else f'本地论文库中未找到与 "{query}" 相关的内容。'
		await on_event(
			{
				'type': 'tool_result',
				'name': 'retrieve_local',
				'content': text,
				'citations': [_local_to_citation(r) for r in results],
			}
		)
		return text

	return {'web_search': web_search, 'retrieve_local': retrieve_local}


# 命令直调前缀:/search <关键词>、/rag <关键词>(与 CLI chat 的 search/rag 命令对齐)
_COMMAND_RE = re.compile(r'^/?(search|rag)\s+(.+)$', re.IGNORECASE)


async def _run_command(rt: ThesisRuntime, message: str, on_event) -> bool:
	"""处理 /search、/rag 命令:不经过 LLM,直接调用对应工具并推送事件。已处理返回 True。"""
	m = _COMMAND_RE.match(message.strip())
	if not m:
		return False
	kind, query = m.group(1).lower(), m.group(2).strip()
	if not query:
		return False
	tool_name = 'web_search' if kind == 'search' else 'retrieve_local'
	args = {'query': query, 'top_k': 5}
	await on_event({'type': 'tool_call', 'name': tool_name, 'arguments': args})
	registry = await _build_tool_registry(rt, on_event)
	await registry[tool_name](args)
	return True


@router.post('/chat/stream')
async def chat_stream(payload: dict) -> StreamingResponse:
	"""SSE 流式对话:一条 user 消息 → 工具调用事件 + token 流 → done。"""
	message = str(payload.get('message', '')).strip()
	session_id = str(payload.get('session_id') or 'default')
	if not message:
		return StreamingResponse(iter([f'data: {json.dumps({"type": "error", "message": "empty message"})}\n\n']), media_type='text/event-stream')

	rt = _get_runtime()
	conv = _SESSIONS.get(session_id)
	if conv is None:
		conv = ConversationMemory(budget_tokens=rt.config.orchestrator.context_window_tokens)
		_SESSIONS[session_id] = conv

	queue: asyncio.Queue = asyncio.Queue()

	async def on_event(ev: dict) -> None:
		await queue.put(ev)

	async def run() -> None:
		try:
			# 命令直调(/search、/rag)优先:不经过 LLM,也不写入对话记忆
			if await _run_command(rt, message, on_event):
				await queue.put({'type': 'done'})
				return
			conv.add('user', message)
			if conv.needs_compaction():
				from .memory.conversation import compact_with_llm

				summary = await compact_with_llm(rt.registry.cheap(), conv, 'research assistant chat')
				conv.apply_compaction(summary)
			tool_registry = await _build_tool_registry(rt, on_event)
			reply = await rt.registry.strong().astream_agent(conv.build(_CHAT_SYSTEM), tool_registry, on_event=on_event)
			conv.add('assistant', reply)
			await queue.put({'type': 'done'})
		except Exception as e:  # noqa: BLE001 - SSE 需把异常推给前端而不是中断连接
			await queue.put({'type': 'error', 'message': f'{type(e).__name__}: {e}'})
		finally:
			await queue.put(None)  # 结束标记

	async def sse_gen() -> AsyncGenerator[str, None]:
		producer = asyncio.create_task(run())
		while True:
			ev = await queue.get()
			if ev is None:
				break
			yield f'data: {json.dumps(ev, ensure_ascii=False)}\n\n'
		await producer

	return StreamingResponse(sse_gen(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})


@router.get('/candidates')
def list_candidates() -> list[dict[str, Any]]:
	"""候选库列表(聊天工具收藏的论文)。"""
	return _get_runtime().store.list('candidates')


@router.post('/candidates')
def add_candidate(payload: dict) -> dict[str, Any]:
	"""收藏一条候选论文(去重键:DOI,否则标题小写)。"""
	key = payload.get('doi') or str(payload.get('title', '')).lower() or 'untitled'
	_get_runtime().store.put('candidates', key, payload)
	return {'ok': True}


@router.post('/rag/ingest-web')
def ingest_web_citation(payload: dict) -> dict[str, Any]:
	"""把 web 搜索结果(引用)自动下载全文 PDF 并入库 RAG。

	同步执行(下载 + 解析 + embedding + 写 faiss 分片),FastAPI 在线程池中
	运行,不阻塞 SSE 聊天事件循环;失败返回 ok=False + 原因,前端展示。

	降级链路:无开放获取 PDF 时(arXiv/openAccessPdf 都没有),转科研通文献
	互助渠道(paper-downloader MCP server)发布求助,返回求助 ID 供后续确认下载。
	"""
	from .models import Citation
	from .rag.web_ingest import ingest_citation

	cit = Citation(**{k: v for k, v in payload.items() if k in Citation.model_fields})
	try:
		return ingest_citation(cit)
	except Exception as e:  # noqa: BLE001 - 前端需要展示具体失败原因
		if '无开放获取' in str(e):
			return _request_ablesci_assist(cit)
		return {
			'ok': False,
			'added': False,
			'paper_id': '',
			'title': str(payload.get('title', '')),
			'message': f'{type(e).__name__}: {e}',
		}


def _request_ablesci_assist(cit) -> dict[str, Any]:
	"""科研通文献互助降级:发布求助(异步 MCP 调用,endpoint 在线程池中运行可安全 asyncio.run)。"""
	import asyncio

	from .search.paper_downloader import request_paper

	try:
		text = asyncio.run(request_paper(doi=cit.doi or '', title=cit.title or ''))
	except Exception as e:  # noqa: BLE001 - 降级失败也要给前端明确原因
		return {
			'ok': False,
			'added': False,
			'paper_id': '',
			'title': cit.title,
			'message': f'科研通求助发布失败: {type(e).__name__}: {e}',
		}
	m_id = re.search(r'求助 ID: ([A-Za-z0-9]+)', text)
	ok = '求助已发布' in text
	return {
		'ok': ok,
		'added': False,
		'requested': True,
		'assist_id': m_id.group(1) if m_id else '',
		'paper_id': '',
		'title': cit.title,
		'message': text,
	}


@router.post('/rag/confirm-assist')
def confirm_assist(payload: dict) -> dict[str, Any]:
	"""确认科研通求助的应助文件:接受应助 → 下载 PDF → 复用 ingest_pdf 入库 RAG。

	前端凭 ingest-web 返回的 assist_id 调用;求助尚未被应助时返回当前状态提示。
	"""
	import asyncio
	from pathlib import Path

	from .config import get_config
	from .rag.index import FaissShards
	from .rag.ingest import ingest_pdf
	from .search.paper_downloader import confirm_and_download

	assist_id = str(payload.get('assist_id', '')).strip()
	if not assist_id:
		return {'ok': False, 'message': '缺少 assist_id。'}
	try:
		text = asyncio.run(confirm_and_download(assist_id))
	except Exception as e:  # noqa: BLE001
		return {'ok': False, 'message': f'{type(e).__name__}: {e}'}

	m_path = re.search(r'保存位置: (.+)', text)
	if not m_path:
		return {'ok': False, 'message': text}

	pdf_path = Path(m_path.group(1).strip())
	cfg = get_config()
	index = FaissShards(cfg.index_path)
	try:
		paper = ingest_pdf(pdf_path, index, subfield='web')
	finally:
		index.close()
	if not paper:
		return {
			'ok': False,
			'message': f'PDF 已下载到 {pdf_path},但解析入库失败;可稍后运行 thesis-agent ingest 重试。',
		}
	return {
		'ok': True,
		'added': True,
		'paper_id': paper['paper_id'],
		'title': paper['title'],
		'message': f'已下载并入库:「{paper["title"]}」({paper["year"] or "?"})',
	}
