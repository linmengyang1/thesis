"""论文入库 CLI:把 data/papers 下的 PDF 解析、分块、向量化并写入 faiss 分区索引。

用法:
    uv run python scripts/ingest_papers.py [--papers-dir DIR] [--subfield general] [--search "query"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from thesis_agent.config import get_config
from thesis_agent.rag.index import FaissShards
from thesis_agent.rag.ingest import ingest_directory
from thesis_agent.rag.retrieve import Retriever


def main() -> None:
	parser = argparse.ArgumentParser(description='论文入库 + 索引')
	parser.add_argument('--papers-dir', help='PDF 目录(默认 data/papers)')
	parser.add_argument('--subfield', default='general', help='子领域标签,如 nlp/vision/optimization')
	parser.add_argument('--rebuild', action='store_true', help='全量重建索引')
	parser.add_argument('--search', help='入库后执行一个检索测试')
	args = parser.parse_args()

	cfg = get_config()
	papers_dir = Path(args.papers_dir) if args.papers_dir else cfg.papers_path
	index = FaissShards(cfg.index_path)

	if args.rebuild:
		index.rebuild_all()
		print('已全量重建分区索引')

	added, skipped = ingest_directory(papers_dir, index, subfield=args.subfield)
	print(f'入库完成:新增 {added} 篇,跳过 {skipped} 篇(已存在或解析失败)')
	print(f'库内论文总数: {len(index.meta.all_papers())}')

	if args.search:
		retriever = Retriever(index)
		results = retriever.retrieve(args.search, top_k=5)
		print(f'\n检索「{args.search}」Top-5:')
		for r in results:
			year = r.year or '?'
			print(f'  [{r.score:.2f}] ({year}) {r.title[:80]}')
			print(f'        {r.text[:120]}...')

	index.close()


if __name__ == '__main__':
	main()
