"""论文下载 MCP server(paper-downloader)的客户端封装。

通过 stdio 拉起 mcp_server/paper_downloader/server.py 子进程并调用其工具。
发布求助/确认下载均为低频操作,每次调用新建会话,无长驻连接复杂度。
"""
from __future__ import annotations

import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# src/thesis_agent/search/paper_downloader.py -> parents[3] = 项目根
_ROOT = Path(__file__).resolve().parents[3]
_SERVER = _ROOT / "mcp_server" / "paper_downloader" / "server.py"


async def call_tool(name: str, args: dict | None = None) -> str:
    """调用 paper-downloader MCP server 的工具,返回文本结果。"""
    params = StdioServerParameters(command=sys.executable, args=[str(_SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args or {})
    return result.content[0].text if result.content else ""


async def request_paper(doi: str = "", title: str = "", reward: int = 10) -> str:
    """在科研通发布文献求助,返回带求助 ID 的结果文本。"""
    return await call_tool("request_paper", {"doi": doi, "title": title, "reward": reward})


async def check_requests() -> str:
    """查看我的科研通求助列表。"""
    return await call_tool("check_requests", {})


async def confirm_and_download(assist_id: str) -> str:
    """确认指定求助的应助文件并下载 PDF 到 data/papers/,返回结果文本。"""
    return await call_tool("confirm_and_download", {"assist_id": assist_id})
