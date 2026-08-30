"""对话记忆:子 agent 的聊天式上下文 + token 预算 + 截断 + 压缩。

设计背景:论文的搜索/调查/撰写是迭代对话,子 agent 应当"记住"
自己之前的轮次。但对话历史会增长,超窗时用两级策略控制:
  1) 硬截断(truncate_*):对注入的素材/引用按预算截掉多余部分;
  2) 压缩摘要(compact_with_llm):对话历史本身超窗时,把旧轮次
     用 LLM 压成 compacted_memory,保留最近 keep_recent 条。
"""
from __future__ import annotations

from typing import Any

_CHARS_PER_TOKEN = 4.0  # 英文粗略估算:4 字符/token


def estimate_tokens(text: str) -> int:
	"""粗略估算文本的 token 数。"""
	if not text:
		return 0
	return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate_text(text: str, max_chars: int) -> str:
	"""硬截断:超长部分截掉并标注。"""
	if len(text) <= max_chars:
		return text
	return text[:max_chars] + f'\n...[截断 {len(text) - max_chars} 字符]'


def truncate_parts(parts: list[str], budget_tokens: int) -> str:
	"""按 token 预算合并多个注入块:按序保留,超预算截断。

	内部统一按字符计数:预算 = budget_tokens * 每 token 字符数。
	"""
	budget_chars = max(1, int(budget_tokens * _CHARS_PER_TOKEN))
	used_chars = 0
	kept: list[str] = []
	for part in parts:
		if not part:
			continue
		remaining = budget_chars - used_chars
		if remaining <= 0:
			break
		trimmed = truncate_text(part, remaining)
		kept.append(trimmed)
		used_chars += len(trimmed)
	return '\n\n'.join(kept)


class ConversationMemory:
	"""单个 agent(或 agent+任务键)的对话历史,带预算与压缩。

	- messages: 当前保留的最近消息(压缩后)
	- compacted: 已被压缩掉的旧轮次摘要列表
	序列化形态(to_state/from_state)用于存入 ThesisState(随 Checkpointer 持久化)。
	"""

	def __init__(self, budget_tokens: int = 15000, keep_recent: int = 8, max_compacted: int = 3) -> None:
		self.budget_tokens = budget_tokens
		self.keep_recent = keep_recent  # 压缩后保留的最近消息条数
		self.max_compacted = max_compacted  # 压缩摘要块上限,防止多轮修订后无限累积
		self.messages: list[dict[str, str]] = []
		self.compacted: list[str] = []

	def add(self, role: str, content: str) -> None:
		self.messages.append({'role': role, 'content': content})

	def total_tokens(self) -> int:
		"""消息 + 压缩摘要的总 token 估算(压缩块同样占用上下文窗口)。"""
		return sum(estimate_tokens(m['content']) for m in self.messages) + sum(
			estimate_tokens(s) for s in self.compacted
		)

	def needs_compaction(self) -> bool:
		"""历史超预算且有旧消息可压缩时,需要压缩。"""
		return self.total_tokens() > self.budget_tokens and len(self.messages) > self.keep_recent

	def older_messages(self) -> list[dict[str, str]]:
		"""返回将被压缩的旧消息(最近 keep_recent 条以外的部分)。"""
		if len(self.messages) <= self.keep_recent:
			return []
		return self.messages[: -self.keep_recent]

	def apply_compaction(self, summary: str) -> None:
		"""应用一次压缩:旧消息替换为摘要。"""
		if not summary or len(self.messages) <= self.keep_recent:
			return
		self.compacted.append(summary)
		# 限制压缩块条数:超过上限时丢弃最旧块(信息价值最低)
		if len(self.compacted) > self.max_compacted:
			self.compacted = self.compacted[-self.max_compacted :]
		self.messages = self.messages[-self.keep_recent :]

	def build(self, system_prompt: str) -> list[dict[str, str]]:
		"""组装发给 LLM 的消息:系统提示 + 压缩摘要(最近 max_compacted 块)+ 最近消息。"""
		out: list[dict[str, str]] = [{'role': 'system', 'content': system_prompt}]
		# 防御性截取:兼容历史 checkpoint 中已超过上限的压缩块
		recent_blocks = self.compacted[-self.max_compacted :]
		if recent_blocks:
			block = '\n'.join(f'<compacted_memory>\n{s}\n</compacted_memory>' for s in recent_blocks)
			out.append({'role': 'user', 'content': block})
		out.extend(self.messages)
		return out

	def to_state(self) -> dict[str, Any]:
		return {'messages': self.messages, 'compacted': self.compacted}

	@classmethod
	def from_state(cls, data: dict[str, Any] | None, budget_tokens: int = 15000) -> 'ConversationMemory':
		cm = cls(budget_tokens=budget_tokens)
		if data:
			cm.messages = data.get('messages', [])
			cm.compacted = data.get('compacted', [])
		return cm


async def compact_with_llm(llm, conversation: ConversationMemory, context_desc: str) -> str:
	"""用 LLM 把对话的旧消息压成摘要。

	压缩是尽力而为:失败时返回空串,调用方跳过压缩(历史保持原样,
	由下一轮重试或硬截断兜底),不阻断撰写主流程。
	"""
	older = conversation.older_messages()
	if not older:
		return ''
	system = (
		'You are compressing an agent conversation for a research paper writing system. '
		'Produce a concise summary that preserves: task requirements, key facts, decisions, '
		'progress, unresolved issues, and next steps. Return plain text only.'
	)
	history = '\n'.join(f"[{m['role']}]: {m['content']}" for m in older)
	user = f'Context: {context_desc}\n\nHistory to compress:\n{history}'
	try:
		resp = await llm.acomplete([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
		return (resp or '').strip()
	except Exception:
		return ''
