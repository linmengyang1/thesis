"""Web 引用一键入库:自动下载全文 PDF 并写入本地 RAG 索引。

前端在 web_search 结果卡片上点「入库」时调用本模块:
1. 从引用里解析 arXiv ID / openAccessPdf 链接,确定 PDF 下载地址;
2. 下载 PDF 到 data/papers(文件名稳定,重复入库按 DOI/路径自动去重);
3. 复用 ingest_pdf 现有管线:解析 → 分块 → embedding → faiss 分区索引。

arXiv 与 Semantic Scholar 的开放获取论文均可自动下载;无开放 PDF 的
论文会返回明确错误,提示用户手动下载后放入 data/papers 再跑入库命令。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

from ..config import get_config
from ..models import Citation
from .index import FaissShards

# arXiv ID:如 2407.11699 / 2407.11699v2(可带 /abs/ /pdf/ 前缀或 arXiv: 前缀)
_ARXIV_ID_RE = re.compile(r'(\d{4}\.\d{4,5})(v\d+)?')

# 下载超时与 UA:arXiv 偶发对默认 UA 返回 403
_TIMEOUT = 60.0
_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'


def arxiv_id_from(text: str) -> str | None:
	"""从 URL / ID 中提取 arXiv 号,找不到返回 None。"""
	if not text:
		return None
	m = _ARXIV_ID_RE.search(text)
	return m.group(1) if m else None


def resolve_pdf_url(cit: Citation) -> str:
	"""确定 PDF 下载地址:优先引用自带的 pdf_url,其次由 arXiv ID 推导。"""
	extra = cit.extra or {}
	pdf_url = str(extra.get('pdf_url') or '').strip()
	if pdf_url:
		return pdf_url
	# arxiv 结果 extra.url 是 abs 页;semantic scholar extra.url 是 arXiv ID
	arxiv_id = arxiv_id_from(str(extra.get('url') or '')) or arxiv_id_from(cit.doi or '')
	if arxiv_id:
		return f'https://arxiv.org/pdf/{arxiv_id}'
	raise ValueError(
		f'「{cit.title[:60]}」无开放获取 PDF 链接(arXiv 或 openAccessPdf),'
		'请手动下载后放入 data/papers 再运行入库命令。'
	)


def _safe_filename(arxiv_id: str | None, cit: Citation) -> str:
	"""生成稳定文件名:arXiv ID 优先,其次 DOI/标题 slug。"""
	if arxiv_id:
		return f'web_{arxiv_id}.pdf'
	base = cit.doi or cit.title or 'untitled'
	slug = re.sub(r'[^a-z0-9]+', '-', base.lower()).strip('-')[:60] or 'untitled'
	return f'web_{slug}.pdf'


def ingest_citation(cit: Citation) -> dict[str, Any]:
	"""下载引用对应的全文 PDF 并入库 RAG。返回 {ok, added, paper_id, title, message}。

	added=True 表示新入库;False 表示库中已存在(按 DOI 去重,跳过下载)。
	"""
	cfg = get_config()
	papers_dir = cfg.papers_path

	# 按 DOI 预判是否已入库,避免重复下载
	paper_id = cit.doi.replace('/', '_') if cit.doi else ''
	if paper_id:
		index = FaissShards(cfg.index_path)
		try:
			if index.meta.get_paper(paper_id):
				return {
					'ok': True,
					'added': False,
					'paper_id': paper_id,
					'title': cit.title,
					'message': f'「{cit.title}」已在本地库中,无需重复入库。',
				}
		finally:
			index.close()

	url = resolve_pdf_url(cit)
	arxiv_id = arxiv_id_from(url) or arxiv_id_from(str((cit.extra or {}).get('url') or '')) or arxiv_id_from(cit.doi or '')
	dest = papers_dir / _safe_filename(arxiv_id, cit)

	from .ingest import ingest_pdf

	resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True, headers={'User-Agent': _UA})
	resp.raise_for_status()
	if not resp.content.startswith(b'%PDF'):
		raise ValueError('下载的不是 PDF 文件,可能是访问受限页面,请手动下载后入库。')
	dest.write_bytes(resp.content)

	index = FaissShards(cfg.index_path)
	try:
		paper = ingest_pdf(dest, index, subfield='web')
	finally:
		index.close()

	if not paper:
		dest.unlink(missing_ok=True)  # 解析失败:清掉占位文件,避免批量入库反复重试
		raise ValueError(f'「{cit.title}」PDF 解析失败,未能分块入库。')
	return {
		'ok': True,
		'added': True,
		'paper_id': paper['paper_id'],
		'title': paper['title'],
		'message': f'已下载并入库:「{paper["title"]}」({paper["year"] or "?"})',
	}
