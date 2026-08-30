<script setup>
// 工具调用卡片:展示工具名/状态/参数,结果以结构化引用列表展示(可收藏、web 结果可一键入库)
import { computed } from 'vue'

const props = defineProps({
  name: { type: String, default: '' },
  status: { type: String, default: 'running' },
  arguments: { type: Object, default: () => ({}) },
  citations: { type: Array, default: () => [] },
  showIngest: { type: Boolean, default: false },
  ingestStates: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['favorite', 'ingest'])

const title = computed(() => {
  if (props.name === 'web_search') return 'Web 搜索(arxiv + Semantic Scholar)'
  if (props.name === 'retrieve_local') return '本地论文库检索(faiss RAG)'
  return props.name
})
const query = computed(() => props.arguments?.query || '')

// 单条引用的入库状态:{state: 'idle'|'busy'|'done'|'error', msg}
function st(i) {
  return props.ingestStates?.[i] || { state: 'idle', msg: '' }
}
const ingestLabel = (i) => {
  const s = st(i).state
  if (s === 'busy') return '入库中…'
  if (s === 'done') return '已入库'
  if (s === 'error') return '重试'
  return '入库'
}
</script>

<template>
  <div :class="['tool', status]">
    <div class="head">
      <span class="dot"></span>
      <span class="name">{{ title }}</span>
      <span v-if="query" class="query">"{{ query }}"</span>
      <span class="state">{{ status === 'running' ? '执行中...' : '完成' }}</span>
    </div>
    <div v-if="status === 'done' && citations.length > 0" class="items">
      <div v-for="(c, i) in citations" :key="i" class="item">
        <span class="idx">{{ i + 1 }}</span>
        <div class="body">
          <div class="ititle">[{{ c.year || '?' }}] {{ c.title }}</div>
          <div class="imeta">{{ (c.authors || []).slice(0, 3).join('; ') || '未知作者' }} | {{ c.venue || c.journal || '?' }}</div>
          <div v-if="st(i).msg" class="ingest-msg" :class="st(i).state">{{ st(i).msg }}</div>
        </div>
        <button v-if="showIngest" class="fav ingest" :disabled="st(i).state === 'busy'" @click="emit('ingest', { citation: c, index: i })">
          {{ ingestLabel(i) }}
        </button>
        <button class="fav" @click="emit('favorite', c)">收藏</button>
      </div>
    </div>
    <div v-else-if="status === 'done'" class="empty">无结果。</div>
  </div>
</template>

<style scoped>
.tool {
  background: #111827; border: 1px solid #334155; border-radius: 8px;
  padding: 10px 12px; max-width: 640px;
}
.tool.running { border-color: #2563eb; }
.tool.done { border-color: #334155; }
.head { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #64748b; animation: pulse 1.2s infinite;
}
.tool.done .dot { background: #16a34a; animation: none; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.name { color: #e2e8f0; font-weight: 600; }
.query { color: #94a3b8; }
.state { margin-left: auto; color: #94a3b8; }
.items { margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }
.item { display: flex; align-items: flex-start; gap: 8px; }
.idx { color: #64748b; font-size: 12px; min-width: 16px; text-align: right; }
.body { flex: 1; min-width: 0; }
.ititle { font-size: 13px; color: #e2e8f0; }
.imeta { font-size: 12px; color: #64748b; }
.fav {
  background: transparent; border: 1px solid #334155; color: #7dd3fc;
  border-radius: 6px; padding: 2px 8px; font-size: 12px; flex-shrink: 0;
}
.fav:hover { border-color: #2563eb; }
.fav.ingest { color: #86efac; }
.fav.ingest:disabled { opacity: 0.5; cursor: not-allowed; }
.ingest-msg { font-size: 12px; margin-top: 2px; }
.ingest-msg.done { color: #86efac; }
.ingest-msg.error { color: #f87171; }
.empty { color: #64748b; font-size: 13px; margin-top: 6px; }
</style>
