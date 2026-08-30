"""论文撰写主入口:启动 langgraph 主编排,产出论文 + BibTeX + 评审报告。

用法:
    uv run python scripts/run_writer.py --topic "..." [--venue "CVPR"] [--thread-id run1]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from thesis_agent.graph.runner import run_writer, summarize


def main() -> None:
	parser = argparse.ArgumentParser(description='论文撰写多 Agent 系统')
	parser.add_argument('--topic', required=True, help='论文主题')
	parser.add_argument('--venue', default='', help='目标会议/期刊')
	parser.add_argument('--thread-id', default='default', help='会话/线程 id(断点续跑用)')
	args = parser.parse_args()

	result = asyncio.run(run_writer(args.topic, args.venue, args.thread_id))
	print(summarize(result))


if __name__ == '__main__':
	main()
