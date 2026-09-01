"""科研通(ablesci.com)站点客户端:Cookie 登录态复用 + 求助发布/查询/确认/下载。

登录态来自浏览器导出的 Cookie 文件(.ablesci_cookie,已被 gitignore,不入库不入对话)。
写操作(发布求助/接受应助)需要页面内嵌的动态 _csrf token,流程均为:
先 GET 页面解析 token,再 POST 提交,接口返回 JSON {code, msg, data}。

已探明的站点接口(2026-08 实测):
- 积分/登录标志:  首页 HTML(「登入后的状态」注释 + 「当前拥有 N 积分」)
- 智能提取:      POST /assist/onekey-query          body: onekey
- 发布求助:      POST /assist/create                body: Assist[...] 完整表单,成功后 data.url 为详情页
- 我的求助列表:  GET  /my/assist-my                 表格行: 详情链接 + assist-badge 状态
- 求助详情:      GET  /assist/detail?id=...         状态徽章 / DOI / 文件列表(assist-file-id)
- 接受应助:      POST /assist/file-handle           body: assist_file_id, note, type=accept
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

BASE = "https://www.ablesci.com"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_TIMEOUT = 30.0

# 文献类型 radio 值:1=期刊论文(站点默认选中项,发布时绝大多数场景)
ASSIST_TYPE_JOURNAL = "1"


class AbleSciError(Exception):
    """科研通业务失败(接口返回非 0 code / 页面结构不符合预期)。"""


class AbleSciAuthError(AbleSciError):
    """登录态失效:Cookie 过期或被踢下线,需重新导出 Cookie。"""


def slugify(text: str, limit: int = 60) -> str:
    """DOI/标题转安全文件名 slug,与主项目 web_ingest 命名风格一致。"""
    base = (text or "untitled").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return slug[:limit] or "untitled"


class AbleSciClient:
    """一个客户端实例持有一份 Cookie;工具内按需创建,不跨会话复用。"""

    def __init__(self, cookie_file: str | Path, download_dir: str | Path):
        self.cookie_file = Path(cookie_file)
        self.download_dir = Path(download_dir)
        self._raw_cookie = self._load_cookie()

    def _load_cookie(self) -> str:
        if not self.cookie_file.exists():
            raise AbleSciAuthError(
                f"Cookie 文件不存在: {self.cookie_file}。请在浏览器登录科研通后,"
                "从 DevTools 复制 Cookie 粘贴到该文件(首行 # 注释可保留)。"
            )
        lines = [
            ln.strip()
            for ln in self.cookie_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        cookie = "".join(lines)
        if "_identity-frontend=" not in cookie:
            raise AbleSciAuthError("Cookie 内容无效:缺少 _identity-frontend(核心登录态),请重新复制完整 Cookie。")
        return cookie

    # ---------- 基础请求 ----------

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"User-Agent": _UA, "Referer": BASE + "/", "Cookie": self._raw_cookie},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    @staticmethod
    def _csrf_from(html: str) -> str:
        # 发布页是 <input name="_csrf"> 表单;详情页可能只嵌在 JS 变量('_csrf': '...')里
        m = re.search(r'name="_csrf"\s+value="([^"]+)"', html) or re.search(
            r"""'_csrf':\s*'([^']+)'""", html
        )
        if not m:
            raise AbleSciError("页面中未找到 _csrf token,站点结构可能已变更。")
        return m.group(1)

    @staticmethod
    def _check_json(payload: dict[str, Any], action: str) -> dict[str, Any]:
        """统一校验科研通 JSON 返回:{code:0} 成功,其他为业务失败。"""
        code = payload.get("code")
        if code not in (0, "0"):
            msg = str(payload.get("msg") or payload.get("message") or "未知错误")
            if "登录" in msg or "登陆" in msg:
                raise AbleSciAuthError(f"{action}失败(登录态失效): {msg}")
            raise AbleSciError(f"{action}失败: {msg}")
        return payload

    # ---------- 只读:登录态 / 智能提取 / 列表 / 详情 ----------

    async def check_login(self) -> dict[str, Any]:
        """验证登录态并返回积分(积分取自 /my/point 的「您当前的总积分为」字段)。"""
        async with self._client() as c:
            r = await c.get(BASE + "/")
            if r.status_code != 200:
                raise AbleSciError(f"首页请求失败: HTTP {r.status_code}")
            logged_in = "登入后的状态" in r.text
            points: int | None = None
            if logged_in:
                rp = await c.get(BASE + "/my/point")
                m = re.search(r"总积分为.*?(\d[\d,]*)", rp.text, re.S)
                if m:
                    points = int(m.group(1).replace(",", ""))
        if not logged_in:
            raise AbleSciAuthError("登录态已失效(首页未检测到登录标志),请重新导出 Cookie 粘贴到 Cookie 文件。")
        return {"logged_in": True, "points": points}

    async def onekey_query(self, onekey: str) -> dict[str, Any]:
        """智能提取:输入 DOI/PMID/标题,返回站点识别的文献信息。"""
        async with self._client() as c:
            page = await c.get(BASE + "/assist/create")
            csrf = self._csrf_from(page.text)
            r = await c.post(
                BASE + "/assist/onekey-query",
                data={"_csrf": csrf, "onekey": onekey},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        if r.status_code != 200:
            raise AbleSciError(f"智能提取请求失败: HTTP {r.status_code}")
        payload = self._check_json(r.json(), "智能提取")
        return payload.get("data") or {}

    async def my_requests(self) -> list[dict[str, Any]]:
        """我的求助列表:[{assist_id, title, status}]。"""
        async with self._client() as c:
            r = await c.get(BASE + "/my/assist-my")
        if r.status_code != 200:
            raise AbleSciError(f"我的求助列表请求失败: HTTP {r.status_code}")
        out: list[dict[str, Any]] = []
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
            m_id = re.search(r'/assist/detail\?id=([A-Za-z0-9]+)', row)
            if not m_id:
                continue
            m_title = re.search(r'<a[^>]*href="[^"]*assist/detail[^"]*"[^>]*>([^<]+)</a>', row)
            m_badge = re.search(r'assist-badge[^"]*">\s*([^<]+?)\s*<', row)
            out.append(
                {
                    "assist_id": m_id.group(1),
                    "title": (m_title.group(1).strip() if m_title else ""),
                    "status": (m_badge.group(1).strip() if m_badge else "未知"),
                }
            )
        return out

    async def detail(self, assist_id: str) -> dict[str, Any]:
        """求助详情:状态 / DOI / 标题 / 待确认文件列表。"""
        async with self._client() as c:
            r = await c.get(BASE + f"/assist/detail?id={assist_id}")
        if r.status_code != 200:
            raise AbleSciError(f"详情页请求失败: HTTP {r.status_code}")
        page = r.text
        if "assist/detail" not in page and "user/login" in page:
            raise AbleSciAuthError("登录态已失效,请重新导出 Cookie。")

        m_badge = re.search(r'assist-badge[^"]*">\s*([^<]+?)\s*<', page)
        m_doi = re.search(r'class="assist-doi">\s*([^\s<]+)', page)
        # <title>【状态】论文标题 - 科研通
        m_title = re.search(r"<title>(?:【[^】]*】)?\s*(.*?)\s*-\s*科研通\s*</title>", page)

        files: list[dict[str, Any]] = []
        # <input type="hidden" value="8Awga4" class="assist-file-id">(value 可能在 class 之前)
        for tag in re.findall(r"<input[^>]*>", page):
            if 'assist-file-id' not in tag:
                continue
            m_id2 = re.search(r'value="([^"]+)"', tag)
            if not m_id2:
                continue
            file_id = m_id2.group(1)
            files.append({"file_id": file_id, "url": BASE + f"/assist/download?id={file_id}"})

        return {
            "assist_id": assist_id,
            "status": (m_badge.group(1).strip() if m_badge else "未知"),
            "doi": (m_doi.group(1).strip() if m_doi else ""),
            "title": (m_title.group(1).strip() if m_title else ""),
            "files": files,
        }

    # ---------- 写操作:发布求助 / 接受应助 ----------

    async def create_request(
        self,
        doi: str = "",
        title: str = "",
        url: str = "",
        reward: int = 10,
        auto_close_days: int = 5,
    ) -> dict[str, Any]:
        """发布文献求助。doi 优先用于智能提取补全 title;返回 {assist_id, detail_url, title}。"""
        if not doi and not title:
            raise AbleSciError("发布求助至少需要 DOI 或 标题 之一。")

        async with self._client() as c:
            page = await c.get(BASE + "/assist/create")
            csrf = self._csrf_from(page.text)

            # DOI 在手时先用智能提取补全标题(命中率远高于纯标题)
            if doi and not title:
                try:
                    info = await self.onekey_query(doi)
                    title = str(info.get("title") or "").strip() or title
                except AbleSciError:
                    pass  # 提取失败不阻塞:只要手动有 title 仍可发布
            if not title:
                raise AbleSciError(f"无法确定文献标题(DOI {doi} 智能提取失败且未提供 title),请补充标题后重试。")

            close_at = (datetime.now() + timedelta(days=auto_close_days)).strftime("%Y-%m-%d %H:%M:%S")
            form = {
                "_csrf": csrf,
                "Assist[doi]": doi,
                "Assist[title]": title,
                "Assist[url]": url,
                "Assist[type]": ASSIST_TYPE_JOURNAL,
                "Assist[point]": str(max(10, int(reward))),
                "Assist[note]": "",
                "Assist[remark]": "",
                "Assist[suppl]": "0",
                "Assist[close_at]": close_at,
            }
            r = await c.post(
                BASE + "/assist/create",
                data=form,
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        if r.status_code != 200:
            raise AbleSciError(f"发布求助请求失败: HTTP {r.status_code}")
        payload = self._check_json(r.json(), "发布求助")
        detail_url = str((payload.get("data") or {}).get("url") or "")
        m_id = re.search(r"id=([A-Za-z0-9]+)", detail_url)
        if not m_id:
            raise AbleSciError(f"发布成功但未解析到求助 ID(返回 url: {detail_url})。")
        return {
            "assist_id": m_id.group(1),
            "detail_url": BASE + f"/assist/detail?id={m_id.group(1)}",
            "title": title,
            "reward": max(10, int(reward)),
            "auto_close_at": close_at,
        }

    async def accept_file(self, assist_id: str, file_id: str, note: str = "") -> None:
        """接受应助(确认文件,求助完结,积分转给应助者)。"""
        async with self._client() as c:
            page = await c.get(BASE + f"/assist/detail?id={assist_id}")
            csrf = self._csrf_from(page.text)
            data = {"_csrf": csrf, "assist_file_id": file_id, "note": note, "type": "accept"}
            r = await c.post(
                BASE + "/assist/file-handle",
                data=data,
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            # 站点敏感词/风险提示流程:subcode==1 时前端会让用户二次确认,这里自动重提一次
            payload = r.json()
            if payload.get("subcode") == 1:
                data["ignore_alert"] = "1"
                r = await c.post(
                    BASE + "/assist/file-handle",
                    data=data,
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )
        if r.status_code != 200:
            raise AbleSciError(f"接受应助请求失败: HTTP {r.status_code}")
        self._check_json(r.json(), "接受应助")

    async def download(self, file_id: str, doi: str = "", title: str = "", highspeed: bool = True) -> Path:
        """按科研通下载协议拉取应助文件:%PDF + 大小完整性校验后落盘 data/papers/。

        协议(中转页 /assist/download 内嵌 JS 实测逆向):
        1. GET  /assist/download?id=<file_id>          中转页,内嵌 config.csrfToken
        2. POST /file/request-download-token           type=assistFile, id=<file_id>,
                                                       channel=normal/highspeed, file_server=<线路>
               → {data: {host, token, transport, expected_size, output_filename}}
        3. GET  {host}?token=<token>                   最终 PDF 字节流

        实测注意:
        - 一次性聚合读(aread)在文件节点会中途断连,必须流式写盘;
        - 慢速线路 2.6MB 约需 5 分钟,读超时放宽到 300s;
        - 节点支持 Range(206),若未来再遇断连可升级断点续传;
        - 免费线路不稳定,按线路 2/3/4 轮换重试。

        高速通道(highspeed=True,默认开启):站点规则为积分<500 时每个文件扣 2 积分
        (同一文件不重复扣),积分>=500 免费;优先高速,失败自动回退普通线路轮换。
        """
        entry = BASE + f"/assist/download?id={file_id}"
        slug = slugify(doi or title or "ablesci")
        dest = self.download_dir / f"web_{slug}.pdf"
        part = dest.with_suffix(".part")
        # 尝试顺序:高速 → 普通线路 2/3/4
        attempts = []
        if highspeed:
            attempts.append({"channel": "highspeed", "highspeed": "1", "file_server": "0"})
        attempts += [
            {"channel": "normal", "highspeed": "0", "file_server": "2"},
            {"channel": "normal", "highspeed": "0", "file_server": "3"},
            {"channel": "normal", "highspeed": "0", "file_server": "4"},
        ]
        last_err: Exception | None = None
        for att in attempts:
            try:
                async with self._client() as c:
                    page = await c.get(entry)
                    m_csrf = re.search(r'"csrfToken":"([^"]+)"', page.text)
                    if not m_csrf:
                        raise AbleSciError("下载中转页未找到 csrfToken,站点结构可能已变更。")

                    r = await c.post(
                        BASE + "/file/request-download-token",
                        data={
                            "_csrf": m_csrf.group(1),
                            "type": "assistFile",
                            "id": file_id,
                            "channel": att["channel"],
                            "highspeed": att["highspeed"],
                            "fallback": "0",
                            "file_server": att["file_server"],
                        },
                        headers={"X-Requested-With": "XMLHttpRequest", "Referer": entry},
                    )
                    payload = self._check_json(r.json(), "获取下载令牌")
                    d = payload.get("data") or {}
                    host, token = str(d.get("host") or ""), str(d.get("token") or "")
                    if not host or not token:
                        raise AbleSciError(f"下载令牌响应缺少 host/token: {list(d.keys())}")
                    expected = int(d.get("expected_size") or 0)
                    final_url = (host if host.startswith("http") else BASE + host) + "?token=" + token

                    c.headers["Referer"] = entry
                    # 流式写盘:聚合读在该节点会断连;identity 避免压缩传输出错;慢线路放宽读超时
                    received = 0
                    async with c.stream(
                        "GET",
                        final_url,
                        headers={"Accept-Encoding": "identity"},
                        timeout=httpx.Timeout(30.0, read=300.0),
                    ) as resp:
                        if resp.status_code != 200:
                            raise AbleSciError(f"文件下载失败: HTTP {resp.status_code}")
                        part.parent.mkdir(parents=True, exist_ok=True)
                        with part.open("wb") as f:
                            async for chunk in resp.aiter_bytes(65536):
                                f.write(chunk)
                                received += len(chunk)

                if not part.read_bytes()[:4].startswith(b"%PDF"):
                    raise AbleSciError("下载内容不是 PDF(可能是登录页或错误页)。")
                if expected and received != expected:
                    raise AbleSciError(
                        f"下载不完整: {received}/{expected} 字节,通道 {att['channel']}(线路 {att['file_server']}) 中途断连。"
                    )
                part.replace(dest)
                return dest
            except (httpx.HTTPError, AbleSciError) as e:
                last_err = e
                part.unlink(missing_ok=True)
                continue  # 换下一条线路重试
        raise AbleSciError(f"三条线路下载均失败,最后错误: {last_err}")

    async def confirm_and_download(self, assist_id: str, note: str = "") -> dict[str, Any]:
        """确认最新应助文件并下载到本地库目录。已完结(24h 内)则跳过确认直接下载,可重复调用。"""
        info = await self.detail(assist_id)
        if not info["files"]:
            if info["status"] == "已完结":
                raise AbleSciError(f"求助 {assist_id} 已完结且文件已被站点清理(超 24 小时),无法再下载。")
            raise AbleSciError(f"求助 {assist_id} 当前状态[{info['status']}]没有可确认的应助文件。")
        f = info["files"][0]
        if info["status"] != "已完结":
            await self.accept_file(assist_id, f["file_id"], note=note)
        dest = await self.download(f["file_id"], doi=info["doi"], title=info["title"])
        return {"file": str(dest), "title": info["title"], "status_before": info["status"]}


def verify_cookie_storage(cookie_file: Path) -> bool:
    """开发用:检查 Cookie 文件是否包含关键登录态字段。"""
    if not cookie_file.exists():
        return False
    text = cookie_file.read_text(encoding="utf-8")
    return "_identity-frontend=" in unquote(text)
