"""运行看板 + Web 聊天服务:FastAPI 统一入口。

- `/` 返回前端(Vue)工作台(看板 + 聊天两个视图,构建产物在 frontend/dist);
- `/api/tasks`、`/api/memory` 只读展示任务板(thesis.db)与长期记忆库(memory.sqlite);
- `/api/chat/stream`、`/api/candidates` 由 chat_api 提供(SSE 流式聊天 + 候选收藏)。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import chat_api
from .config import PROJECT_ROOT, get_config

app = FastAPI(title='thesis-agent 工作台', docs_url=None, redoc_url=None)
app.include_router(chat_api.router)

# 前端构建产物(frontend/dist);不存在时(未构建)根路径回退到内置 HTML 看板
_FRONTEND_DIST = PROJECT_ROOT / 'frontend' / 'dist'
_HAS_FRONTEND = (_FRONTEND_DIST / 'index.html').exists()

if _HAS_FRONTEND:
	app.mount('/assets', StaticFiles(directory=_FRONTEND_DIST / 'assets'), name='assets')

# 任务状态 → 徽标颜色(与 TaskStatus 对齐)
_STATUS_COLORS = {
	'queued': '#6b7280',
	'in_progress': '#2563eb',
	'needs_review': '#9333ea',
	'in_revision': '#d97706',
	'approved': '#16a34a',
	'merged': '#0d9488',
	'blocked': '#dc2626',
}


def _readonly_conn(path: Path) -> sqlite3.Connection:
	"""只读 SQLite 连接:避免与正在写入的任务板互相加锁。"""
	return sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=3)


def _tasks_data(db_file: Path) -> dict[str, Any]:
	"""读取任务板 tasks 表,按 run_id 分组并做状态统计。"""
	if not db_file.exists():
		return {'runs': [], 'runs_count': 0, 'total': 0}
	conn = _readonly_conn(db_file)
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(
			'SELECT id, run_id, title, kind, status, revision_count, chapter_id, note '
			'FROM tasks ORDER BY run_id, kind'
		).fetchall()
	finally:
		conn.close()
	runs: dict[str, list[dict[str, Any]]] = {}
	for r in rows:
		runs.setdefault(r['run_id'], []).append(
			{
				'id': r['id'],
				'title': r['title'],
				'kind': r['kind'],
				'status': r['status'],
				'revision_count': r['revision_count'],
				'chapter_id': r['chapter_id'],
				'note': r['note'],
			}
		)
	run_list = []
	for run_id, tasks in runs.items():
		status_counts: dict[str, int] = {}
		for t in tasks:
			status_counts[t['status']] = status_counts.get(t['status'], 0) + 1
		run_list.append({'run_id': run_id, 'tasks': tasks, 'counts': status_counts})
	run_list.sort(key=lambda x: x['run_id'], reverse=True)
	return {'runs': run_list, 'runs_count': len(run_list), 'total': len(rows)}


def _memory_data(mem_file: Path) -> dict[str, Any]:
	"""读取长期记忆库,按 namespace 统计。"""
	if not mem_file.exists():
		return {'namespaces': [], 'total': 0}
	conn = _readonly_conn(mem_file)
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(
			'SELECT namespace, key, value, updated_at FROM memory ORDER BY namespace, updated_at DESC'
		).fetchall()
	finally:
		conn.close()
	namespaces: dict[str, list[dict[str, Any]]] = {}
	for r in rows:
		namespaces.setdefault(r['namespace'], []).append(
			{'key': r['key'], 'value': r['value'], 'updated_at': r['updated_at']}
		)
	ns_list = [{'namespace': k, 'entries': v} for k, v in sorted(namespaces.items())]
	return {'namespaces': ns_list, 'total': len(rows)}


@app.get('/api/tasks')
def api_tasks() -> dict[str, Any]:
	return _tasks_data(get_config().db_file)


@app.get('/api/memory')
def api_memory() -> dict[str, Any]:
	cfg = get_config()
	return _memory_data(cfg.db_file.parent / 'memory.sqlite')


_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>thesis-agent 运行看板</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background: #0f172a; color: #e2e8f0; }
  header { padding: 16px 24px; background: #1e293b; display: flex; align-items: baseline; gap: 16px; border-bottom: 1px solid #334155; }
  header h1 { font-size: 18px; margin: 0; }
  header .stat { color: #94a3b8; font-size: 13px; }
  .wrap { padding: 16px 24px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
  .card h2 { font-size: 15px; margin: 0 0 10px; display: flex; align-items: center; gap: 10px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; color: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #334155; }
  th { color: #94a3b8; font-weight: 500; }
  td.note { color: #f59e0b; }
  .muted { color: #64748b; }
  .ns { margin-bottom: 10px; }
  .ns summary { cursor: pointer; color: #7dd3fc; }
  pre { margin: 4px 0 0; font-size: 12px; color: #94a3b8; white-space: pre-wrap; word-break: break-all; }
</style>
</head>
<body>
<header>
  <h1>thesis-agent 运行看板</h1>
  <span class="stat" id="stamp">加载中...</span>
</header>
<div class="wrap" id="content">加载中...</div>
<script>
const COLORS = {queued:'#6b7280',in_progress:'#2563eb',needs_review:'#9333ea',in_revision:'#d97706',approved:'#16a34a',merged:'#0d9488',blocked:'#dc2626'};
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function badge(status){ return '<span class="badge" style="background:'+(COLORS[status]||'#64748b')+'">'+esc(status)+'</span>'; }
function ts(sec){ return sec ? new Date(sec*1000).toLocaleString() : ''; }
function renderTasks(data){
  if(!data.total){ return '<div class="card muted">任务板为空。运行 `thesis-agent run ...` 后此处实时展示进度。</div>'; }
  let html = '<div class="card muted">任务总数 ' + data.total + ' · 运行代次 ' + data.runs_count + '</div>';
  for(const run of data.runs){
    html += '<div class="card"><h2>run: ' + esc(run.run_id);
    for(const [k,v] of Object.entries(run.counts)){ html += ' ' + badge(k) + ' x' + v; }
    html += '</h2><table><thead><tr><th>kind</th><th>任务</th><th>章节</th><th>状态</th><th>评审轮次</th><th>备注</th></tr></thead><tbody>';
    for(const t of run.tasks){
      html += '<tr><td>' + esc(t.kind) + '</td><td>' + esc(t.title) + '</td><td>' + esc(t.chapter_id) + '</td>'
            + '<td>' + badge(t.status) + '</td><td>' + t.revision_count + '</td>'
            + '<td class="note">' + esc(t.note) + '</td></tr>';
    }
    html += '</tbody></table></div>';
  }
  return html;
}
function renderMemory(data){
  if(!data.total){ return '<div class="card muted">长期记忆库为空。</div>'; }
  let html = '<div class="card"><h2>长期记忆库(' + data.total + ' 条)</h2>';
  for(const ns of data.namespaces){
    html += '<div class="ns"><details><summary>' + esc(ns.namespace) + ' · ' + ns.entries.length + ' 条</summary>';
    for(const e of ns.entries){
      let v = esc(e.value); if(v.length > 300) v = v.slice(0,300) + '...';
      html += '<pre>[' + e.key + '] ' + v + '<br><span class="muted">更新于 ' + ts(e.updated_at) + '</span></pre>';
    }
    html += '</details></div>';
  }
  return html + '</div>';
}
async function tick(){
  const stamp = document.getElementById('stamp');
  stamp.textContent = '更新时间 ' + new Date().toLocaleTimeString();
  try{
    const [t, m] = await Promise.all([fetch('/api/tasks').then(r=>r.json()), fetch('/api/memory').then(r=>r.json())]);
    document.getElementById('content').innerHTML = renderTasks(t) + '<hr>' + renderMemory(m);
  }catch(e){
    document.getElementById('content').innerHTML = '<div class="card muted">读取失败: ' + esc(e) + '</div>';
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
def index() -> str:
	"""前端构建存在时返回 Vue 工作台,否则返回内置 HTML 看板。"""
	if _HAS_FRONTEND:
		return (_FRONTEND_DIST / 'index.html').read_text(encoding='utf-8')
	return _PAGE
