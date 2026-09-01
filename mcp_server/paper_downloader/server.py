"""论文下载 MCP server(科研通文献互助渠道),stdio 传输。

暴露 4 个工具,供 MCP 客户端(Trae / thesis-agent 集成链路)调用:
- check_login:          验证 Cookie 登录态,返回当前积分
- request_paper:        智能提取文献信息并发布求助(DOI/PMID/标题),返回求助 ID
- check_requests:       查看我的求助列表与状态(求助中/待确认/已完结)
- confirm_and_download: 接受应助文件并下载 PDF 到主项目 data/papers/(后续 ingest 入 RAG)

运行: .venv/Scripts/python.exe mcp_server/paper_downloader/server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablesci_client import AbleSciClient, AbleSciAuthError, AbleSciError  # noqa: E402

# 主项目根目录(mcp_server/paper_downloader/server.py -> 上两级)
_ROOT = Path(__file__).resolve().parents[2]
COOKIE_FILE = _ROOT / "mcp_server" / "paper_downloader" / ".ablesci_cookie"
DOWNLOAD_DIR = Path(os.environ.get("PAPER_DOWNLOAD_DIR") or (_ROOT / "data" / "papers"))

mcp = FastMCP(
    "paper-downloader",
    instructions="科研通文献互助渠道:发布文献求助,等待应助后确认并下载 PDF 到本地论文库。",
)


def _client() -> AbleSciClient:
    return AbleSciClient(COOKIE_FILE, DOWNLOAD_DIR)


def _err_text(e: Exception) -> str:
    if isinstance(e, AbleSciAuthError):
        return f"[登录态失效] {e} (请重新从浏览器复制 Cookie 到 mcp_server/paper_downloader/.ablesci_cookie)"
    if isinstance(e, AbleSciError):
        return f"[科研通错误] {e}"
    return f"[意外错误] {type(e).__name__}: {e}"


@mcp.tool()
async def check_login() -> str:
    """验证科研通 Cookie 登录态,返回是否有效与当前积分余额。"""
    try:
        info = await _client().check_login()
    except Exception as e:
        return _err_text(e)
    return f"登录有效,当前积分 {info['points']}。"


@mcp.tool()
async def request_paper(doi: str = "", title: str = "", pmid: str = "", reward: int = 10) -> str:
    """在科研通发布文献求助。优先用 DOI(智能提取命中率最高),其次 PMID 或英文标题。

    发布后通常数十秒到几分钟内被自动应助;到期无人应助积分自动返还。
    参数: doi 如 "10.1126/science.aba208";reward 为悬赏积分(最低 10)。
    返回求助 ID,之后用 check_requests 查状态、confirm_and_download 确认下载。
    """
    try:
        client = _client()
        if not doi and pmid:
            # PMID 走智能提取补全 title/doi,提高发布成功率
            info = await client.onekey_query(pmid)
            doi = str(info.get("doi") or "").strip()
            title = title or str(info.get("title") or "").strip()
        req = await client.create_request(doi=doi, title=title, reward=reward)
    except Exception as e:
        return _err_text(e)
    return (
        f"求助已发布: 「{req['title']}」 悬赏 {req['reward']} 积分,自动关闭时间 {req['auto_close_at']}。\n"
        f"求助 ID: {req['assist_id']}\n"
        f"详情页: {req['detail_url']}\n"
        f"应助后调用 confirm_and_download(assist_id=\"{req['assist_id']}\") 确认并下载。"
    )


@mcp.tool()
async def check_requests() -> str:
    """查看我的科研通求助列表(标题/状态/求助 ID)。"""
    try:
        items = await _client().my_requests()
    except Exception as e:
        return _err_text(e)
    if not items:
        return "当前没有任何求助记录。"
    lines = [f"- [{it['status']}] {it['title'][:60]} (ID: {it['assist_id']})" for it in items]
    return "我的求助:\n" + "\n".join(lines)


@mcp.tool()
async def confirm_and_download(assist_id: str, note: str = "") -> str:
    """接受指定求助的应助文件(积分转给应助者,求助完结),下载 PDF 到本地论文库 data/papers/。

    仅当求助状态为[待确认]时可用;下载成功后可运行 thesis-agent ingest 增量入库。
    """
    try:
        result = await _client().confirm_and_download(assist_id, note=note)
    except Exception as e:
        return _err_text(e)
    return (
        f"已接受应助并下载:「{result['title']}」\n"
        f"保存位置: {result['file']}\n"
        f"下一步: 运行 thesis-agent ingest --subfield cv 增量入库。"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
