"""聊天 agent 的工具注册:schema 定义 + 工具执行器。

工具以 {name: async (arguments: dict) -> str} 形式注入 acomplete_agent。
TOOL_SCHEMAS 同时用于 anthropic / openai 两种 provider 的 tool 声明。
"""
from __future__ import annotations

from ..search.paper_search import search_papers

# 工具 schema(同时兼容 anthropic input_schema 与 openai parameters)
TOOL_SCHEMAS: list[dict] = [
    {
        'name': 'web_search',
        'description': '搜索学术论文(arxiv + Semantic Scholar)。查找相关文献、调研研究现状时使用。',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': '检索关键词(英文更佳)'},
                'top_k': {'type': 'integer', 'description': '返回结果数,默认 5'},
            },
            'required': ['query'],
        },
    },
    {
        'name': 'retrieve_local',
        'description': '在本地论文库(faiss RAG)中检索。需要引用本地已有论文内容时使用。',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': '检索主题或问题'},
                'top_k': {'type': 'integer', 'description': '返回结果数,默认 5'},
            },
            'required': ['query'],
        },
    },
]


def _fmt_citations(citations: list, index_offset: int = 0) -> str:
	lines = [f'找到 {len(citations)} 条结果:']
	for i, c in enumerate(citations, 1):
		authors = '; '.join(c.authors[:3]) or '未知作者'
		source = c.venue or c.extra.get('source', '')
		lines.append(
			f'{index_offset + i}. [{c.year or "?"}] {c.title}\n'
			f'   作者: {authors} | 来源: {source or "?"} | DOI: {c.doi or "无"}'
		)
	return '\n'.join(lines)


async def run_web_search(arguments: dict) -> str:
	"""web_search 工具执行器(独立可用,不依赖 runtime)。"""
	query = str(arguments.get('query', '')).strip()
	top_k = int(arguments.get('top_k', 5) or 5)
	if not query:
		return '缺少检索关键词 query。'
	citations = await search_papers(query, max_results=max(1, top_k))
	if not citations:
		return f'未找到与 "{query}" 相关的论文。'
	return _fmt_citations(citations)


async def run_retrieve_local(arguments: dict, retriever) -> str:
	"""retrieve_local 工具执行器(依赖 runtime 的 retriever,由聊天层注入)。"""
	query = str(arguments.get('query', '')).strip()
	top_k = int(arguments.get('top_k', 5) or 5)
	if not query:
		return '缺少检索关键词 query。'
	results = retriever.retrieve(query, top_k=max(1, top_k))
	if not results:
		return f'本地论文库中未找到与 "{query}" 相关的内容。'
	lines = [f'本地检索到 {len(results)} 条相关内容:']
	for i, r in enumerate(results, 1):
		authors = '; '.join(r.authors[:3]) or '未知'
		lines.append(
			f'{i}. [{r.year or "?"}] {r.title}\n'
			f'   作者: {authors} | 来源: {r.venue or "?"}\n'
			f'   片段: {r.text[:200]}'
		)
	return '\n'.join(lines)


# 无需 runtime 的独立工具
TOOL_REGISTRY: dict = {
	'web_search': run_web_search,
}
