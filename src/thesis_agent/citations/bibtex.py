"""BibTeX 生成与管理:Citation → .bib 文本,双向转换,生成 citation key。"""
from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any

import bibtexparser

from ..models import Citation


def _slugify(text: str) -> str:
	"""转成 BibTeX key 可用的单词。"""
	words = re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()
	return ''.join(words[:4]) if words else 'untitled'


def _normalize_surname(name: str) -> str:
	"""姓氏归一化:小写 + 去组合音调符号(NFKD)。

	保证 'Peña' 与 'pena'、'Douzon' 等在不同环节(正文引用/引用库/校验)可一致匹配。
	"""
	return ''.join(ch for ch in unicodedata.normalize('NFKD', name) if not unicodedata.combining(ch)).lower()


def make_citation_key(c: Citation) -> str:
	"""生成稳定 citation key:首作者姓氏 + 年份 + 标题前几个词。"""
	first_author = ''
	if c.authors:
		surname = _normalize_surname(c.authors[0].split()[-1])
		first_author = surname
	year = c.year if c.year else ''
	slug = _slugify(c.title)[:24] or uuid.uuid4().hex[:6]
	return f'{first_author}{year}{slug}'[:48]


def citation_to_bibtex_entry(c: Citation) -> dict[str, str]:
	key = c.key or make_citation_key(c)
	fields: dict[str, str] = {'title': c.title}
	if c.authors:
		fields['author'] = ' and '.join(c.authors)
	if c.year:
		fields['year'] = str(c.year)
	if c.venue:
		fields['journal'] = c.venue
	if c.doi:
		fields['doi'] = c.doi
	if c.abstract:
		fields['abstract'] = c.abstract
	return {'ID': key, 'ENTRYTYPE': 'article', **fields}


def citations_to_bibtex(citations: list[Citation]) -> str:
	"""将 Citation 列表序列化为 .bib 文本。"""
	bib = bibtexparser.bibdatabase.BibDatabase()
	bib.entries = [citation_to_bibtex_entry(c) for c in citations]
	return bibtexparser.dumps(bib)


def bibtex_to_citations(text: str) -> list[Citation]:
	"""解析 .bib 文本为 Citation 列表(bibtexparser v1 条目为 dict)。"""
	db = bibtexparser.loads(text)
	out: list[Citation] = []
	for entry in db.entries:
		author_field = str(entry.get('author', ''))
		out.append(
			Citation(
				key=entry.get('ID', ''),
				title=entry.get('title', ''),
				authors=[a.strip() for a in author_field.split(' and ') if a.strip()],
				year=_int_or_none(entry.get('year')),
				venue=entry.get('journal', entry.get('booktitle', '')),
				doi=entry.get('doi', ''),
				abstract=entry.get('abstract', ''),
			)
		)
	return out


def _int_or_none(value: Any) -> int | None:
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


_AUTHOR_YEAR_RE = re.compile(r'\[([^\[\]]+?),\s*(\d{4})\]')
# 正文方括号引用组:[key] / [key1; key2] / [Author et al., YYYY]
_CITATION_GROUP_RE = re.compile(r'\[([^\[\]]+)\]')
# 形如 "Author et al., YYYY" 或 "Author and Author, YYYY"
_AUTHOR_YEAR_CITE_RE = re.compile(r'^(.*?),\s*(\d{4})$')


def _first_author_surname(author_part: str) -> str:
	"""从 '[X et al./and Y, ZZZZ]' 中取首个作者的姓氏(归一化小写、去音调、去标点)。

	如 'Binmakhashen and Mahmoud' -> 'binmakhashen';'Yang et al.' -> 'yang';
	'Peña et al.' -> 'pena'(与 make_citation_key 的 key 前缀保持一致)。
	"""
	first = author_part.split(' and ', 1)[0]
	tokens = [t for t in re.split(r'\s+', first) if t.lower() not in ('et', 'al.', 'al')]
	if not tokens:
		return ''
	return re.sub(r'[^a-z0-9\-]', '', _normalize_surname(tokens[-1]))


def rewrite_author_year_citations(text: str, citations: list[Citation]) -> tuple[str, list[str]]:
	"""把正文中的 [Author et al., YYYY] 标记统一改写为 [key](引用溯源一致化)。

	返回 (改写后文本, 未匹配的警告列表)。
	"""
	key_by_surname_year: dict[tuple[str, str], str] = {}
	for c in citations:
		if not c.authors or not c.year:
			continue
		surname = _normalize_surname(c.authors[0].split()[-1])
		key_by_surname_year[(surname, str(c.year))] = c.key or ''

	warnings: list[str] = []

	def _replace(match: re.Match) -> str:
		author_part = match.group(1).strip()
		year = match.group(2)
		# 取首个作者的姓氏(支持 'A and B' / 'A et al.' 两种写法)
		surname = _first_author_surname(author_part)
		key = key_by_surname_year.get((surname, year))
		if key:
			return f'[{key}]'
		warnings.append(f'未匹配到引用的作者-年份标记: [{author_part}, {year}]')
		return match.group(0)

	return _AUTHOR_YEAR_RE.sub(_replace, text), warnings


def _looks_like_key(token: str) -> bool:
	"""粗略判断裸 token 是否像引用 key(字母数字开头,长度>=3,可含 _ : - .)。"""
	return bool(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_:\-\.]{2,}', token))


def validate_citations_in_text(text: str, citations: list[Citation]) -> dict[str, str]:
	"""校验正文中出现的引用都能在 citations 中找到,返回 {问题: 说明} 字典。

	- '[Author et al., YYYY]' 型引用按首作者姓氏+年份整体校验,不再拆词误报;
	- '[key]' / '[key1; key2]' 型校验 key 是否存在于引用库。
	"""
	keys = {c.key for c in citations if c.key}
	surname_year: dict[tuple[str, str], str] = {}
	for c in citations:
		if c.authors and c.year:
			surname_year[(_normalize_surname(c.authors[0].split()[-1]), str(c.year))] = c.key or ''

	problems: dict[str, str] = {}
	for group in _CITATION_GROUP_RE.findall(text):
		group = group.strip()
		if not group:
			continue
		ay = _AUTHOR_YEAR_CITE_RE.match(group)
		if ay:
			# 作者-年份型引用:整体核对,命中即视为有效
			surname = _first_author_surname(ay.group(1).strip())
			if (surname, ay.group(2)) not in surname_year:
				problems.setdefault(group, f'引用 [{group}] 无匹配 BibTeX 条目')
			continue
		# 裸 key 型:可能 [key] 或 [key1; key2]
		for token in group.split(';'):
			token = token.strip().lstrip('@')
			if not token or token in keys:
				continue
			if _looks_like_key(token):
				problems.setdefault(token, f'引用 [{token}] 不在 BibTeX 中')
	return problems
