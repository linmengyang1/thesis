// 后端 API 封装:普通 fetch + SSE 流式聊天
const BASE = '/api'

async function json(url, options = {}) {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

// 看板:任务板 + 长期记忆
export const fetchTasks = () => json(`${BASE}/tasks`)
export const fetchMemory = () => json(`${BASE}/memory`)

// 候选库
export const fetchCandidates = () => json(`${BASE}/candidates`)
export const saveCandidate = (citation) =>
  json(`${BASE}/candidates`, { method: 'POST', body: JSON.stringify(citation) })

// Web 引用一键入库 RAG:后端自动下载全文 PDF 并写入本地 faiss 索引
export const ingestWebCitation = (citation) =>
  json(`${BASE}/rag/ingest-web`, { method: 'POST', body: JSON.stringify(citation) })

// SSE 流式聊天:POST body {message, session_id};返回事件流
// 事件帧:{"type":"token"|"tool_call"|"tool_result"|"done", ...}
export async function streamChat(message, sessionId, onEvent) {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE 以空行分隔事件,逐条解析
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const part of parts) {
      const dataLine = part.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      try {
        onEvent(JSON.parse(dataLine.slice(5).trim()))
      } catch (e) {
        console.warn('解析 SSE 事件失败', e)
      }
    }
  }
}
