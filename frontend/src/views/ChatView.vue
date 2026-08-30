<script setup>
// 研究助手聊天视图:SSE 流式回复 + 工具调用卡片 + 候选库面板
import { ref, nextTick, onMounted } from 'vue'
import { streamChat, fetchCandidates, saveCandidate, ingestWebCitation } from '../api.js'
import MessageBubble from '../components/MessageBubble.vue'
import ToolCard from '../components/ToolCard.vue'

const messages = ref([])      // {id, role: 'user'|'assistant'|'tool', content, name?, status?, arguments?, citations?}
const input = ref('')
const busy = ref(false)
const candidates = ref([])
const sessionId = ref('')

const quickCommands = [
  { label: '搜索论文', hint: 'search <关键词>', text: '/search ' },
  { label: '本地检索', hint: 'rag <关键词>', text: '/rag ' },
]

function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

function ensureSession() {
  if (!sessionId.value) {
    sessionId.value = localStorage.getItem('chat_session') || newId()
    localStorage.setItem('chat_session', sessionId.value)
  }
}

async function loadCandidates() {
  try {
    candidates.value = await fetchCandidates()
  } catch (e) {
    console.warn('读取候选库失败', e)
  }
}

async function scrollBottom() {
  await nextTick()
  const el = document.querySelector('.msg-list')
  if (el) el.scrollTop = el.scrollHeight
}

async function send(raw) {
  const text = (raw || input.value).trim()
  if (!text || busy.value) return
  ensureSession()
  messages.value.push({ id: newId(), role: 'user', content: text })
  input.value = ''
  busy.value = true
  await scrollBottom()

  let curAssistant = null
  try {
    await streamChat(text, sessionId.value, (ev) => {
      if (ev.type === 'tool_call') {
        messages.value.push({
          id: newId(), role: 'tool', name: ev.name,
          status: 'running', arguments: ev.arguments || {}, citations: [],
        })
      } else if (ev.type === 'tool_result') {
        const t = [...messages.value].reverse().find((m) => m.role === 'tool' && m.name === ev.name && m.status === 'running')
        if (t) {
          t.status = 'done'
          t.citations = ev.citations || []
          t.content = ev.content || ''
          // 助手通常会把结果汇入回复,这里清空卡片正文只保留结构化引用
        }
      } else if (ev.type === 'token') {
        if (!curAssistant || curAssistant.role !== 'assistant') {
          curAssistant = { id: newId(), role: 'assistant', content: '' }
          messages.value.push(curAssistant)
        }
        curAssistant.content += ev.content
      }
      scrollBottom()
    })
  } catch (e) {
    messages.value.push({ id: newId(), role: 'assistant', content: `请求失败: ${e.message}` })
  } finally {
    busy.value = false
    await scrollBottom()
  }
}

async function onFavorite(cit) {
  try {
    await saveCandidate(cit)
    await loadCandidates()
  } catch (e) {
    console.warn('收藏失败', e)
  }
}

// 一键入库:把 web 搜索到的引用自动下载全文 PDF 并写入本地 RAG
async function onIngest({ citation, index }, toolMsg) {
  if (!toolMsg.ingestStates) toolMsg.ingestStates = {}
  toolMsg.ingestStates[index] = { state: 'busy', msg: '' }
  try {
    const r = await ingestWebCitation(citation)
    toolMsg.ingestStates[index] = {
      state: r.ok ? 'done' : 'error',
      msg: r.message || (r.ok ? '已入库' : '入库失败'),
    }
  } catch (e) {
    toolMsg.ingestStates[index] = { state: 'error', msg: `请求失败: ${e.message}` }
  }
}

async function removeCandidate(idx) {
  // 候选库删除走 store.delete,后端暂未提供 DELETE 接口,此处仅提示
}

onMounted(() => {
  ensureSession()
  loadCandidates()
  messages.value.push({
    id: newId(), role: 'assistant',
    content: '你好,我是论文研究助手。可以让我搜索学术论文(web_search)或检索你的本地论文库(retrieve_local),也可以直接提问。试试输入 "search 文档版面分析" 或点击下方快捷按钮。',
  })
})
</script>

<template>
  <div class="chat">
    <div class="msg-list">
      <MessageBubble v-for="m in messages" :key="m.id" :msg="m" @favorite="onFavorite" @ingest="(e) => onIngest(e, m)" />
      <div v-if="busy && !messages.some((m) => m.role === 'tool' && m.status === 'running')" class="thinking">思考中...</div>
    </div>

    <div class="quick">
      <button v-for="c in quickCommands" :key="c.text" @click="input = c.text" :title="c.hint">{{ c.label }}</button>
    </div>

    <div class="input-row">
      <input
        v-model="input"
        placeholder="输入消息; /search 关键词 或 /rag 关键词 直接调工具"
        :disabled="busy"
        @keydown.enter="send()"
      />
      <button :disabled="busy || !input.trim()" @click="send()">发送</button>
    </div>

    <div class="candidates">
      <div class="cand-title">候选库({{ candidates.length }})</div>
      <div v-if="candidates.length === 0" class="cand-empty">在搜索结果卡片上点"收藏"后显示在这里。</div>
      <div v-for="(c, i) in candidates" :key="i" class="cand-item">
        <div>[{{ c.year || '?' }}] {{ c.title }}</div>
        <div class="cand-meta">{{ (c.authors || []).slice(0, 3).join('; ') || '未知作者' }} | {{ c.venue || '?' }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.msg-list { flex: 1; overflow-y: auto; padding: 16px 24px; }
.thinking { color: #64748b; font-size: 13px; padding: 8px 0; }
.quick { display: flex; gap: 8px; padding: 0 24px 8px; }
.quick button {
  background: #1e293b; border: 1px solid #334155; color: #7dd3fc;
  border-radius: 6px; padding: 4px 12px; font-size: 13px;
}
.quick button:hover { border-color: #2563eb; }
.input-row { display: flex; gap: 8px; padding: 8px 24px 12px; }
.input-row input {
  flex: 1; background: #1e293b; border: 1px solid #334155; color: #e2e8f0;
  border-radius: 8px; padding: 10px 12px; font-size: 14px; outline: none;
}
.input-row input:focus { border-color: #2563eb; }
.input-row button {
  background: #2563eb; color: #fff; border: none; border-radius: 8px; padding: 0 20px;
}
.input-row button:disabled { opacity: 0.5; cursor: not-allowed; }
.candidates { border-top: 1px solid #334155; padding: 8px 24px 12px; max-height: 150px; overflow-y: auto; }
.cand-title { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
.cand-empty { font-size: 12px; color: #64748b; }
.cand-item { font-size: 13px; padding: 4px 0; border-bottom: 1px dashed #1e293b; }
.cand-meta { font-size: 12px; color: #64748b; }
</style>
