<script setup>
// 运行看板视图:轮询 /api/tasks 与 /api/memory 实时展示撰写进度
import { ref, onMounted, onUnmounted } from 'vue'
import { fetchTasks, fetchMemory } from '../api.js'

const tasks = ref({ runs: [], runs_count: 0, total: 0 })
const memory = ref({ namespaces: [], total: 0 })
const stamp = ref('')

const STATUS_COLORS = {
  queued: '#6b7280', in_progress: '#2563eb', needs_review: '#9333ea',
  in_revision: '#d97706', approved: '#16a34a', merged: '#0d9488', blocked: '#dc2626',
}

async function tick() {
  stamp.value = new Date().toLocaleTimeString()
  try {
    const [t, m] = await Promise.all([fetchTasks(), fetchMemory()])
    tasks.value = t
    memory.value = m
  } catch (e) {
    console.warn('读取看板失败', e)
  }
}

let timer = null
onMounted(() => {
  tick()
  timer = setInterval(tick, 3000)
})
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div class="board">
    <div class="hint">任务总数 {{ tasks.total }} · 运行代次 {{ tasks.runs_count }} · 更新时间 {{ stamp }}</div>

    <div v-if="tasks.total === 0" class="empty">任务板为空。运行 `thesis-agent run ...` 后此处实时展示进度。</div>

    <div v-for="run in tasks.runs" :key="run.run_id" class="card">
      <h3>
        run: {{ run.run_id }}
        <span v-for="(v, k) in run.counts" :key="k" class="badge" :style="{ background: STATUS_COLORS[k] || '#64748b' }">
          {{ k }} x{{ v }}
        </span>
      </h3>
      <table>
        <thead>
          <tr><th>kind</th><th>任务</th><th>章节</th><th>状态</th><th>评审轮次</th><th>备注</th></tr>
        </thead>
        <tbody>
          <tr v-for="t in run.tasks" :key="t.id">
            <td>{{ t.kind }}</td>
            <td>{{ t.title }}</td>
            <td>{{ t.chapter_id }}</td>
            <td><span class="badge" :style="{ background: STATUS_COLORS[t.status] || '#64748b' }">{{ t.status }}</span></td>
            <td>{{ t.revision_count }}</td>
            <td class="note">{{ t.note }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h3>长期记忆库({{ memory.total }} 条)</h3>
      <div v-if="memory.total === 0" class="empty">长期记忆库为空。</div>
      <details v-for="ns in memory.namespaces" :key="ns.namespace" class="ns">
        <summary>{{ ns.namespace }} · {{ ns.entries.length }} 条</summary>
        <pre v-for="e in ns.entries" :key="e.key">[{{ e.key }}] {{ (e.value || '').length > 300 ? e.value.slice(0, 300) + '...' : e.value }}
<span class="muted">更新于 {{ new Date(e.updated_at * 1000).toLocaleString() }}</span></pre>
      </details>
    </div>
  </div>
</template>

<style scoped>
.board { padding: 16px 24px; overflow-y: auto; flex: 1; }
.hint { color: #94a3b8; font-size: 13px; margin-bottom: 12px; }
.empty { color: #64748b; padding: 12px 0; }
.card {
  background: #1e293b; border: 1px solid #334155; border-radius: 8px;
  padding: 14px 16px; margin-bottom: 16px;
}
h3 { font-size: 15px; margin: 0 0 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; color: #fff; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #334155; }
th { color: #94a3b8; font-weight: 500; }
td.note { color: #f59e0b; }
.ns { margin-bottom: 10px; }
.ns summary { cursor: pointer; color: #7dd3fc; }
pre { margin: 4px 0 0; font-size: 12px; color: #94a3b8; white-space: pre-wrap; word-break: break-all; }
.muted { color: #64748b; }
</style>
