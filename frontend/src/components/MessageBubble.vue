<script setup>
// 单条消息气泡:用户 / 助手(带 markdown 渲染)
import { computed } from 'vue'
import { marked } from 'marked'
import ToolCard from './ToolCard.vue'

const props = defineProps({
  msg: { type: Object, required: true },
})
const emit = defineEmits(['favorite', 'ingest'])

const html = computed(() => {
  if (props.msg.role !== 'assistant') return ''
  return marked.parse(props.msg.content || '', { breaks: true })
})
</script>

<template>
  <div v-if="msg.role === 'tool'" class="row">
    <ToolCard
      :name="msg.name"
      :status="msg.status"
      :arguments="msg.arguments"
      :citations="msg.citations"
      :show-ingest="msg.name === 'web_search'"
      :ingest-states="msg.ingestStates"
      @favorite="emit('favorite', $event)"
      @ingest="emit('ingest', $event)"
    />
  </div>
  <div v-else :class="['row', msg.role === 'user' ? 'right' : 'left']">
    <div :class="['bubble', msg.role]">
      <div v-if="msg.role === 'assistant'" class="md" v-html="html"></div>
      <div v-else>{{ msg.content }}</div>
    </div>
  </div>
</template>

<style scoped>
.row { display: flex; margin: 10px 0; }
.row.right { justify-content: flex-end; }
.bubble { max-width: 72%; padding: 10px 14px; border-radius: 10px; font-size: 14px; line-height: 1.6; }
.bubble.user { background: #2563eb; color: #fff; }
.bubble.assistant { background: #1e293b; border: 1px solid #334155; }
.md :deep(pre) { background: #0f172a; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12px; }
.md :deep(code) { background: #0f172a; padding: 1px 4px; border-radius: 4px; font-size: 12px; }
.md :deep(pre code) { background: none; padding: 0; }
</style>
