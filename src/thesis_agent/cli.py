"""CLI 入口:`thesis-agent run|ingest ...`。"""
from __future__ import annotations

import argparse
import asyncio


def main() -> None:
	parser = argparse.ArgumentParser(description='论文撰写多 Agent 系统')
	sub = parser.add_subparsers(dest='command', required=True)

	run_p = sub.add_parser('run', help='运行论文撰写流程')
	run_p.add_argument('--topic', required=True, help='论文主题')
	run_p.add_argument('--venue', default='', help='目标会议/期刊')
	run_p.add_argument('--thread-id', default='default', help='会话/线程 id(断点续跑用)')
	run_p.set_defaults(func=_run)

	chat_p = sub.add_parser('chat', help='进入交互式聊天助手(可随时调用 Web 搜索/本地检索工具)')
	chat_p.set_defaults(func=_chat)

	ingest_p = sub.add_parser('ingest', help='论文入库')
	ingest_p.add_argument('--papers-dir', help='PDF 目录(默认 data/papers)')
	ingest_p.add_argument('--subfield', default='general', help='子领域标签')
	ingest_p.add_argument('--rebuild', action='store_true', help='全量重建索引')
	ingest_p.add_argument('--force', action='store_true', help='强制重新提取并修复已入库论文的元数据(不重复 embedding)')
	ingest_p.add_argument('--search', help='入库后执行检索测试')
	ingest_p.set_defaults(func=_ingest)

	serve_p = sub.add_parser('serve', help='启动运行看板(FastAPI),实时展示任务板与长期记忆库')
	serve_p.add_argument('--host', default='127.0.0.1', help='监听地址(默认 127.0.0.1)')
	serve_p.add_argument('--port', type=int, default=9000, help='监听端口(默认 9000)')
	serve_p.set_defaults(func=_serve)

	args = parser.parse_args()
	args.func(args)


def _run(args: argparse.Namespace) -> None:
	from .graph.runner import run_writer, summarize

	result = asyncio.run(run_writer(args.topic, args.venue, args.thread_id))
	print(summarize(result))


def _chat(args: argparse.Namespace) -> None:
	from .chat import run_chat

	asyncio.run(run_chat())


def _ingest(args: argparse.Namespace) -> None:
	from .config import get_config
	from .rag.index import FaissShards
	from .rag.ingest import ingest_directory

	cfg = get_config()
	from pathlib import Path

	papers_dir = Path(args.papers_dir) if args.papers_dir else cfg.papers_path
	index = FaissShards(cfg.index_path)
	if args.rebuild:
		index.rebuild_all()
		print('已全量重建分区索引')
	added, skipped = ingest_directory(papers_dir, index, subfield=args.subfield, force=args.force)
	print(f'入库完成:新增 {added} 篇,跳过 {skipped} 篇;库内论文总数 {len(index.meta.all_papers())}')
	if args.force:
		print('已按 --force 重新提取并修复论文元数据(标题/作者/年份),保留原有分块与向量。')
	if args.search:
		from .rag.retrieve import Retriever

		results = Retriever(index).retrieve(args.search, top_k=5)
		for r in results:
			print(f'  [{r.score:.2f}] ({r.year or "?"}) {r.title[:70]}')
	index.close()


def _serve(args: argparse.Namespace) -> None:
	"""启动 FastAPI 看板:只读展示任务板与长期记忆库,页面每 3 秒自动刷新。"""
	import uvicorn

	from .dashboard import app

	print(f'运行看板: http://{args.host}:{args.port}  (Ctrl+C 退出)')
	uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
	main()
