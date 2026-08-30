"""Web 论文搜索:arxiv 官方 API + Semantic Scholar Graph API(均合规免费)。

只取元数据与摘要,不搬运正文;命中结果进入"候选库",由用户确认后入 RAG。
"""
from __future__ import annotations

import asyncio
import os
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ..config import get_config
from ..models import Citation

_ARXIV_NS = {'a': 'http://www.w3.org/2005/Atom'}

# 可重试的 HTTP 状态:限流(429/408)与临时服务端错误(5xx)
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# 同一来源相邻两次请求的最小间隔(秒),降低免费接口(如 Semantic Scholar)的 429 概率
_RATE_LIMIT_SECONDS = 1.2
_last_call: dict[str, float] = {}


async def _throttle(source: str) -> None:
	"""同源限速:距该来源上一次请求至少 _RATE_LIMIT_SECONDS。"""
	now = time.monotonic()
	wait = _last_call.get(source, 0.0) + _RATE_LIMIT_SECONDS - now
	if wait > 0:
		await asyncio.sleep(wait)
		now = time.monotonic()
	_last_call[source] = now


async def _get_with_retry(
	client: httpx.AsyncClient,
	url: str,
	*,
	params: dict[str, Any] | None = None,
	headers: dict[str, str] | None = None,
	retries: int = 4,
	base_delay: float = 1.0,
) -> httpx.Response:
	"""带指数退避的 GET,覆盖瞬时 DNS/连接失败与 429/5xx 限流。

	间隔 = base_delay * 2^(尝试次数-1);429 时优先遵循 Retry-After 响应头(封顶 30s)。
	"""
	delay = base_delay
	for attempt in range(1, retries + 1):
		try:
			resp = await client.get(url, params=params, headers=headers)
		except httpx.TransportError:
			if attempt < retries:
				await asyncio.sleep(delay)
				delay *= 2
				continue
			raise
		if resp.status_code in _RETRYABLE_STATUS:
			if attempt < retries:
				retry_after = resp.headers.get('Retry-After')
				wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
				await asyncio.sleep(min(wait, 30))
				delay *= 2
				continue
		resp.raise_for_status()
		return resp
	raise httpx.TransportError('unreachable')


async def search_arxiv(query: str, max_results: int = 10) -> list[Citation]:
	"""arxiv API 搜索,返回 Citation 列表。"""
	cfg = get_config()
	await _throttle('arxiv')
	params = {'search_query': f'all:{query}', 'max_results': max_results}
	async with httpx.AsyncClient(timeout=30) as client:
		resp = await _get_with_retry(client, cfg.search.arxiv_api, params=params)
	root = ET.fromstring(resp.text)
	out: list[Citation] = []
	for entry in root.findall('a:entry', _ARXIV_NS):
		title = (entry.findtext('a:title', default='', namespaces=_ARXIV_NS) or '').strip()
		summary = (entry.findtext('a:summary', default='', namespaces=_ARXIV_NS) or '').strip()
		year = None
		published = entry.findtext('a:published', default='', namespaces=_ARXIV_NS)
		if published and len(published) >= 4:
			try:
				year = int(published[:4])
			except ValueError:
				pass
		authors = [a.findtext('a:name', default='', namespaces=_ARXIV_NS) for a in entry.findall('a:author', _ARXIV_NS)]
		link = entry.findtext('a:id', default='', namespaces=_ARXIV_NS)
		arxiv_id = link.rsplit('/abs/', 1)[-1].split('/')[0] if '/abs/' in link else ''
		out.append(
			Citation(
				key='',
				title=title,
				authors=[a for a in authors if a],
				year=year,
				venue='arXiv',
				doi=link,
				abstract=summary[:1500],
				extra={
					'source': 'arxiv',
					'url': link,
					'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}' if arxiv_id else '',
				},
			)
		)
		return out


async def search_semantic_scholar(query: str, max_results: int = 10) -> list[Citation]:
	"""Semantic Scholar Graph API 搜索(可选 API key 降低限流)。"""
	cfg = get_config()
	await _throttle('semantic_scholar')
	fields = 'title,authors,year,venue,externalIds,abstract,openAccessPdf'
	params = {'query': query, 'limit': max_results, 'fields': fields}
	headers = {}
	api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
	if api_key:
		headers['x-api-key'] = api_key
	async with httpx.AsyncClient(timeout=30) as client:
		resp = await _get_with_retry(client, cfg.search.semantic_scholar_api, params=params, headers=headers)
		payload: dict[str, Any] = resp.json()
	out: list[Citation] = []
	for paper in payload.get('data', []):
		ids = paper.get('externalIds') or {}
		oa = paper.get('openAccessPdf') or {}
		out.append(
			Citation(
				key='',
				title=paper.get('title', ''),
				authors=[a.get('name', '') for a in (paper.get('authors') or []) if a.get('name')],
				year=paper.get('year'),
				venue=paper.get('venue', ''),
				doi=ids.get('DOI', ''),
				abstract=(paper.get('abstract') or '')[:1500],
				extra={
					'source': 'semantic_scholar',
					'url': ids.get('ArXiv', ''),
					'pdf_url': oa.get('url', ''),
				},
			)
		)
	return out


async def search_papers(query: str, max_results: int | None = None) -> list[Citation]:
	"""组合搜索:arxiv + semantic scholar,去重(按 DOI/标题)。

	外部 API 可能限流/超时:单个来源失败不影响其他来源,尽力返回已命中的结果。
	"""
	cfg = get_config()
	limit = max_results or cfg.search.max_results
	seen: set[str] = set()
	out: list[Citation] = []
	for source in (search_arxiv, search_semantic_scholar):
		try:
			results = await source(query, limit)
		except Exception as e:
			print(f'[search] {source.__name__} 失败(跳过): {type(e).__name__}: {str(e)[:100]}')
			continue
		for c in results:
			key = c.doi or c.title.lower()
			if key in seen:
				continue
			seen.add(key)
			out.append(c)
	return out
