<script setup>
// 应用外壳:顶部导航 + 按 hash 切换看板/聊天视图
import { ref, computed, onMounted } from 'vue'
import DashboardView from './views/DashboardView.vue'
import ChatView from './views/ChatView.vue'

const route = ref(location.hash || '#/dashboard')

function parseHash() {
  const h = location.hash.replace(/^#/, '') || '/dashboard'
  route.value = h
}

onMounted(() => {
  window.addEventListener('hashchange', parseHash)
  parseHash()
})

const current = computed(() => route.value.split('?')[0])
</script>

<template>
  <nav class="topnav">
    <span class="brand">thesis-agent 工作台</span>
    <a href="#/dashboard" :class="{ active: current === '/dashboard' }">运行看板</a>
    <a href="#/chat" :class="{ active: current === '/chat' }">研究助手</a>
  </nav>
  <main class="main">
    <DashboardView v-if="current === '/dashboard'" />
    <ChatView v-else-if="current === '/chat'" />
    <div v-else class="empty">未知页面: {{ current }}</div>
  </main>
</template>

<style scoped>
.topnav {
  display: flex; align-items: center; gap: 20px;
  padding: 12px 24px; background: #1e293b;
  border-bottom: 1px solid #334155;
}
.brand { font-weight: 600; color: #f1f5f9; }
.topnav a {
  color: #94a3b8; text-decoration: none; font-size: 14px;
  padding: 6px 12px; border-radius: 6px;
}
.topnav a:hover { color: #e2e8f0; }
.topnav a.active { background: #2563eb; color: #fff; }
.main { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.empty { padding: 40px; color: #64748b; }
</style>
