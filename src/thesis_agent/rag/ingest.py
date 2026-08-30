"""论文入库管线:PDF → 文本 → 分块 → 元数据 → embedding → faiss 分区。

分块策略:按标题层级切节,大节再按 ~500 token(约 2000 字符)子分块,
跨分块重叠 10% 并尽量落在句子边界。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_config
from .embedder import Embedder
from .index import FaissShards

# 粗略 token 估算:英文约 4 字符/token
_CHARS_PER_TOKEN = 4.0
_DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+')


def paper_id_from_path(path: Path) -> str:
	"""由文件路径生成稳定的 paper_id。"""
	return hashlib.md5(str(path.resolve()).encode('utf-8')).hexdigest()[:16]


def extract_pdf_text(path: Path) -> tuple[str, list[tuple[int, str]], dict[str, str]]:
	"""提取 PDF 纯文本、(字号,文本)跨度与内置元数据,用于标题层级检测与元数据修复。"""
	import pymupdf

	doc = pymupdf.open(str(path))
	pdf_meta = {k: (v or '') for k, v in (doc.metadata or {}).items()}
	spans: list[tuple[int, str]] = []  # (font_size, text)
	full_text: list[str] = []
	for page in doc:
		full_text.append(page.get_text())
		for block in page.get_text('dict')['blocks']:
			for line in block.get('lines', []):
				line_text = ''.join(span.get('text', '') for span in line.get('spans', []))
				if not line_text.strip():
					continue
				size = max((s.get('size', 0) for s in line.get('spans', [])), default=0)
				spans.append((size, line_text.strip()))
	doc.close()
	return '\n'.join(full_text), spans, pdf_meta


# arXiv 首页头部行(如 "arXiv:2407.11699v1 [cs.CV] 16 Jul 2024"),不是论文标题
_ARXIV_HEADER_RE = re.compile(r'^arxiv:\s?\d{4}\.\d{5}', re.IGNORECASE)
# 纯日期行(如 "March 17, 2026")
_DATE_LINE_RE = re.compile(
	r'^(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)\s+\d{1,2},?\s+\d{4}$',
	re.IGNORECASE,
)
# 归属机构/邮箱等非作者名的关键词,用于过滤启发式作者解析
_AUTHOR_STOPWORDS = re.compile(
	r'\b(Team|Inc|Ltd|University|Institute|School|Department|Research|Laboratory|Lab|Academy|Company|Corporation|Center|Centre|College|Google|Baidu|Alibaba|Microsoft|Beijing|Correspondence|Affiliations?|Contact)\b',
	re.IGNORECASE,
)
_MAX_AUTHORS = 20


def _clean_name(raw: str) -> str:
	"""去掉姓名后的上标数字/符号(如 'Yian Zhao1,2†'),去除首尾标点。"""
	name = re.sub(r'[\d†‡⋆*]+', '', raw)
	return re.sub(r'\s+', ' ', name).strip(' ,;.')


def _name_usable(name: str) -> bool:
	"""判断清洗后的名字是否像真实作者名(拉丁字母/中文、非归属关键词)。"""
	if len(name) < 3:
		return False
	if not re.fullmatch(r"[A-Za-zÀ-ɏ㐀-鿿'\-\. ]+", name):
		return False
	if _AUTHOR_STOPWORDS.search(name):
		return False
	return True


def _year_from_path(path: Path) -> int | None:
	"""从文件名中提取年份(如 CVPR_2024 / ICFHR2022 / (2026))。"""
	m = re.search(r'(19|20)\d{2}', str(path))
	return int(m.group(0)) if m else None


def _parse_authors(full_text: str, title: str, pdf_meta: dict[str, str] | None = None) -> list[str]:
	"""提取作者列表:优先 PDF 内置 author 字段(分号分隔);否则在首页标题后、Abstract 前解析。"""
	pdf_meta = pdf_meta or {}
	meta_author = (pdf_meta.get('author') or '').strip()
	if meta_author:
		authors = [a.strip() for a in meta_author.split(';') if _name_usable(a.strip())]
		if authors:
			return authors[:_MAX_AUTHORS]

	lines = [line.strip() for line in full_text.splitlines() if line.strip()]
	names: list[str] = []
	seen_title = (not title) or bool(_ARXIV_HEADER_RE.search(title))
	for line in lines[:50]:
		if not seen_title:
			if title and (line[:40] in title or title[:40] in line):
				seen_title = True
			continue
		low = line.lower()
		if low.startswith(('abstract', 'keywords', '1 ', '1.', '2 ', '2.')) or low.startswith('©'):
			break
		if _ARXIV_HEADER_RE.search(line) or _DATE_LINE_RE.search(line):
			continue
		if '@' in line or 'http' in line:
			continue
		# 作者行通常是逗号分隔的名字列表,或含 ' and '
		if (',' in line or ' and ' in line) and len(line) <= 200:
			for part in re.split(r'\s*,\s*|\s+and\s+', line):
				name = _clean_name(part)
				if name and _name_usable(name) and name not in names:
					names.append(name)
		if len(names) >= _MAX_AUTHORS:
			break
	return names


def detect_metadata(path: Path, full_text: str, spans: list[tuple[int, str]], pdf_meta: dict[str, str] | None = None) -> dict[str, Any]:
	"""抽取标题/年份/DOI/摘要/作者:优先 PDF 内置元数据,缺失时用首页文本启发式补全。

	此前标题取自首页最大字号行,常把 arXiv 头部串(如 "arXiv:2407.11699v1 [cs.CV]")误当标题,
	作者恒为空导致引用库缺失作者;本版优先 doc.metadata 并补充作者解析。
	"""
	pdf_meta = pdf_meta or {}
	lines = [line.strip() for line in full_text.splitlines() if line.strip()]

	# 标题:PDF 内置元数据优先;否则取首页最大字号、且非 arXiv 头/日期行的正文
	title = (pdf_meta.get('title') or '').strip()
	if _ARXIV_HEADER_RE.search(title) or len(title) < 8:
		title = ''
	if not title and spans:
		max_size = max((s for s, _ in spans[:80]), default=0)
		for size, text in spans[:80]:
			if size >= max_size * 0.9 and len(text) > 10 and not _ARXIV_HEADER_RE.search(text) and not _DATE_LINE_RE.search(text):
				title = text
				break
	if not title:
		# 跳过 arXiv 头 / 日期 / ResearchGate 封面等非正文行,取第一个像标题的行
		for line in lines[:30]:
			if _ARXIV_HEADER_RE.search(line) or _DATE_LINE_RE.search(line):
				continue
			if line.startswith('See discussions') or '@' in line or 'http' in line:
				continue
			if len(line) >= 12:
				title = line[:200]
				break

	year = None
	year_match = re.search(r'\b(19|20)\d{2}\b', full_text[:3000])
	if year_match:
		year = int(year_match.group(0))
	if year is None:
		year = _year_from_path(path)

	doi = ''
	doi_match = _DOI_RE.search(full_text[:6000])
	if doi_match:
		doi = doi_match.group(0)

	abstract = ''
	abstract_start = re.search(r'\bAbstract\b', full_text, re.IGNORECASE)
	abstract_end = re.search(r'\b(Introduction|1\.\s|Keywords)', full_text[abstract_start.end() :] if abstract_start else '')
	if abstract_start:
		end = abstract_start.end() + (abstract_end.start() if abstract_end else 3000)
		abstract = re.sub(r'\s+', ' ', full_text[abstract_start.end() : end]).strip()[:1500]

	return {
		'title': title,
		'year': year,
		'doi': doi,
		'abstract': abstract,
		'authors': _parse_authors(full_text, title, pdf_meta),
		'venue': '',
	}


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
	"""按段落边界把长文本切成 max_chars 左右的分块,带 overlap。"""
	paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
	if not paragraphs:
		return []
	chunks: list[str] = []
	buf = ''
	for para in paragraphs:
		if len(para) > max_chars:
			# 超长段落自身再切
			if buf:
				chunks.append(buf)
				buf = ''
			chunks.extend(_split_long_paragraph(para, max_chars, overlap_chars))
			continue
		if len(buf) + len(para) <= max_chars:
			buf = buf + '\n\n' + para if buf else para
		else:
			if buf:
				chunks.append(buf)
			# overlap:复用上一块末尾句子,保持上下文
			tail = _tail_sentences(buf, overlap_chars) if buf else ''
			buf = (tail + '\n\n' if tail else '') + para
	if buf:
		chunks.append(buf)
	return [c for c in chunks if c.strip()]


def _split_long_paragraph(para: str, max_chars: int, overlap_chars: int) -> list[str]:
	"""按句子边界切超长段落。"""
	sentences = re.split(r'(?<=[.!?])\s+', para)
	out: list[str] = []
	buf = ''
	for sent in sentences:
		if len(buf) + len(sent) <= max_chars:
			buf = buf + ' ' + sent if buf else sent
		else:
			if buf:
				out.append(buf)
			tail = _tail_sentences(buf, overlap_chars) if buf else ''
			buf = (tail + ' ' if tail else '') + sent
	if buf:
		out.append(buf)
	return out


def _tail_sentences(text: str, max_chars: int) -> str:
	"""取文本末尾约 max_chars 字符的完整句子,用于 overlap。"""
	if len(text) <= max_chars:
		return text
	trimmed = text[-max_chars:]
	first_dot = trimmed.find('. ')
	if first_dot > 0:
		trimmed = trimmed[first_dot + 1 :]
	return trimmed.strip()


def ingest_pdf(path: Path, index: FaissShards, subfield: str = 'general', shard: int | None = None, force: bool = False) -> dict[str, Any]:
	"""入库一篇 PDF:解析→分块→embedding→写入分区索引。返回该论文记录。

	force=True 时对已入库论文仅重新提取并修复元数据(标题/作者/年份等),
	保留原有分块与向量,不重复 embedding。
	"""
	cfg = get_config()
	text, spans, pdf_meta = extract_pdf_text(path)
	meta = detect_metadata(path, text, spans, pdf_meta)

	paper_id = meta['doi'].replace('/', '_') if meta['doi'] else paper_id_from_path(path)
	existing = index.meta.get_paper(paper_id)
	# 已入库则跳过(增量特性:同一 paper_id 只入一次);force 时走元数据修复
	if existing and not force:
		return existing
	if existing and force:
		updated = dict(existing)
		updated.update(
			title=meta['title'],
			authors=meta['authors'],
			year=meta['year'],
			venue=meta['venue'],
			doi=meta['doi'],
			abstract=meta['abstract'],
			path=str(path),
		)
		index.meta.upsert_paper(updated)
		return updated

	# paper_id 可能是 MD5 十六进制,也可能是 DOI 别名(含 .、字母),统一做 md5 哈希取模,保证任意格式都稳定分片
	shard = shard if shard is not None else int(hashlib.md5(str(paper_id).encode('utf-8')).hexdigest()[:8], 16) % cfg.rag.shards
	max_chars = int(cfg.rag.chunk_size_tokens * _CHARS_PER_TOKEN)
	overlap = int(max_chars * cfg.rag.chunk_overlap)
	chunks_text = chunk_text(text, max_chars, overlap)
	if not chunks_text:
		return {}

	vectors = Embedder.embed(chunks_text)
	chunks = [
		{
			'chunk_id': f'{paper_id}:{seq}',
			'paper_id': paper_id,
			'seq': seq,
			'shard': shard,
			'text': chunks_text[seq],
			'vector': np.asarray(vectors[seq], dtype='float32').tobytes(),
		}
		for seq in range(len(chunks_text))
	]
	paper = {
		'paper_id': paper_id,
		'title': meta['title'],
		'authors': meta['authors'],
		'year': meta['year'],
		'venue': meta['venue'],
		'doi': meta['doi'],
		'abstract': meta['abstract'],
		'path': str(path),
		'subfield': subfield,
		'shard': shard,
	}
	index.add_paper(paper, chunks)
	return paper


def ingest_directory(papers_dir: Path | str, index: FaissShards, subfield: str = 'general', force: bool = False) -> tuple[int, int]:
	"""批量入库一个目录下的所有 PDF。返回 (新增, 跳过) 数。"""
	added = 0
	skipped = 0
	for pdf in sorted(Path(papers_dir).rglob('*.pdf')):
		paper = ingest_pdf(pdf, index, subfield=subfield, force=force)
		if paper:
			added += 1
		else:
			skipped += 1
	return added, skipped
