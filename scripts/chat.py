"""交互式聊天助手入口。

用法:
    uv run python scripts/chat.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from thesis_agent.chat import run_chat


def main() -> None:
	asyncio.run(run_chat())


if __name__ == '__main__':
	main()
