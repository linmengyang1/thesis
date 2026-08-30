"""混合策略模型工厂:strong/medium/cheap 三档,统一异步 chat 接口。

provider 支持 anthropic / openai / deepseek(OpenAI 兼容协议)。
API 密钥从环境变量读取:ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY。
"""
from __future__ import annotations

import os
from typing import Any

from ..config import Config, LlmConfig, LlmTierConfig, get_config

# 标准消息格式:[{'role': 'system' | 'user' | 'assistant', 'content': str}]
ChatMessage = dict[str, str]


class ChatModel:
	"""单一档位模型的统一封装。"""

	def __init__(self, tier: str, cfg: LlmTierConfig) -> None:
		self.tier = tier
		self.cfg = cfg
		self.provider = cfg.provider.lower()
		self._client: Any = None

	def _get_client(self) -> Any:
		if self._client is not None:
			return self._client
		if self.provider == 'anthropic':
			from anthropic import AsyncAnthropic

			self._client = AsyncAnthropic(api_key=self.cfg.api_key or os.getenv('ANTHROPIC_API_KEY'))
		else:  # openai / deepseek 均走 OpenAI 兼容协议
			from openai import AsyncOpenAI

			base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
			api_key = self.cfg.api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
			if not api_key:
				raise RuntimeError(f'{self.provider} 未配置 API 密钥,请检查 config.yaml 或 .env')
			if self.provider == 'deepseek':
				self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
			else:
				self._client = AsyncOpenAI(api_key=api_key)
		return self._client

	async def acomplete(self, messages: list[ChatMessage]) -> str:
		"""给定标准消息列表,返回模型补全文本。"""
		client = self._get_client()
		if self.provider == 'anthropic':
			system_text = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
			conversation = [
				{'role': m['role'], 'content': m['content']}
				for m in messages
				if m['role'] in ('user', 'assistant')
			]
			resp = await client.messages.create(
				model=self.cfg.model,
				max_tokens=8192,
				temperature=self.cfg.temperature,
				system=system_text or None,
				messages=conversation,
			)
			return ''.join(block.text for block in resp.content if block.type == 'text')
		resp = await client.chat.completions.create(
			model=self.cfg.model,
			temperature=self.cfg.temperature,
			messages=messages,  # type: ignore[arg-type]
		)
		return resp.choices[0].message.content or ''

	async def acomplete_agent(
		self,
		messages: list[ChatMessage],
		tool_registry: dict[str, Any],
		max_tool_rounds: int = 4,
	) -> str:
		"""带工具调用的 agent 循环。

		tool_registry: {工具名: async (arguments: dict) -> str},执行结果回填给模型。
		内部处理 anthropic / openai 两种 tool 格式;返回最终文本(工具往返过程是瞬态的,
		不写入对话记忆)。工具执行异常时把错误文本回填,不中断循环。
		"""
		import json

		client = self._get_client()
		tools = self._provider_tools(list(tool_registry.keys()))
		last_text = ''
		for _ in range(max_tool_rounds):
			if self.provider == 'anthropic':
				system_text = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
				conversation = [m for m in messages if m['role'] in ('user', 'assistant')]
				resp = await client.messages.create(
					model=self.cfg.model,
					max_tokens=8192,
					temperature=self.cfg.temperature,
					system=system_text or None,
					messages=conversation,  # type: ignore[arg-type]
					tools=tools or None,
				)
				last_text = ''.join(b.text for b in resp.content if b.type == 'text')
				tool_uses = [b for b in resp.content if b.type == 'tool_use']
				if not tool_uses:
					return last_text
				# 附加 assistant 的 tool_use 块
				messages.append(
					{
						'role': 'assistant',
						'content': [
							*([{'type': 'text', 'text': last_text}] if last_text else []),
							*[
								{'type': 'tool_use', 'id': b.id, 'name': b.name, 'input': b.input}
								for b in tool_uses
							],
						],
					}
				)
				tool_results = []
				for b in tool_uses:
					tool_results.append(
						{'type': 'tool_result', 'tool_use_id': b.id, 'content': await self._run_tool(tool_registry, b.name, b.input)}
					)
				messages.append({'role': 'user', 'content': tool_results})
			else:  # openai / deepseek
				resp = await client.chat.completions.create(
					model=self.cfg.model,
					temperature=self.cfg.temperature,
					messages=messages,  # type: ignore[arg-type]
					tools=tools or None,
				)
				msg = resp.choices[0].message
				last_text = msg.content or ''
				calls = msg.tool_calls or []
				if not calls:
					return last_text
				messages.append(msg)  # assistant 消息含 tool_calls
				for c in calls:
					try:
						args = json.loads(c.function.arguments or '{}')
					except json.JSONDecodeError:
						args = {}
					messages.append(
						{'role': 'tool', 'tool_call_id': c.id, 'content': await self._run_tool(tool_registry, c.function.name, args)}
					)
		return last_text

	async def _run_tool(self, tool_registry: dict[str, Any], name: str, arguments: dict) -> str:
		"""执行工具并返回文本结果;异常时返回错误文本。"""
		fn = tool_registry.get(name)
		if fn is None:
			return f'工具不存在: {name}'
		try:
			result = await fn(arguments)
			return str(result)
		except Exception as e:
			return f'工具 {name} 执行失败: {type(e).__name__}: {e}'

	async def astream_agent(
		self,
		messages: list[ChatMessage],
		tool_registry: dict[str, Any],
		on_event=None,
		max_tool_rounds: int = 4,
	) -> str:
		"""带工具调用的流式 agent 循环(SSE 聊天用)。

		- OpenAI 兼容协议(deepseek/openai):逐 token 推 `{'type':'token'}` 事件;
		  工具调用前推 `{'type':'tool_call','name','arguments'}`,执行后推 `{'type':'tool_result','name','content'}`。
		- anthropic 协议:工具轮与最终轮非流式(一次推全文 token 事件),接口形态保持一致。
		on_event:async (event: dict) -> None;为 None 时只返回最终文本。
		"""
		import json

		client = self._get_client()
		tools = self._provider_tools(list(tool_registry.keys()))
		last_text = ''
		for _ in range(max_tool_rounds):
			if self.provider == 'anthropic':
				system_text = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
				conversation = [m for m in messages if m['role'] in ('user', 'assistant')]
				resp = await client.messages.create(
					model=self.cfg.model,
					max_tokens=8192,
					temperature=self.cfg.temperature,
					system=system_text or None,
					messages=conversation,  # type: ignore[arg-type]
					tools=tools or None,
				)
				last_text = ''.join(b.text for b in resp.content if b.type == 'text')
				tool_uses = [b for b in resp.content if b.type == 'tool_use']
				if not tool_uses:
					if on_event:
						await on_event({'type': 'token', 'content': last_text})
					return last_text
				messages.append(
					{
						'role': 'assistant',
						'content': [
							*([{'type': 'text', 'text': last_text}] if last_text else []),
							*[
								{'type': 'tool_use', 'id': b.id, 'name': b.name, 'input': b.input}
								for b in tool_uses
							],
						],
					}
				)
				tool_results = []
				for b in tool_uses:
					if on_event:
						await on_event({'type': 'tool_call', 'name': b.name, 'arguments': b.input})
					result = await self._run_tool(tool_registry, b.name, b.input)
					if on_event:
						await on_event({'type': 'tool_result', 'name': b.name, 'content': result})
					tool_results.append({'type': 'tool_result', 'tool_use_id': b.id, 'content': result})
				messages.append({'role': 'user', 'content': tool_results})
				continue

			# OpenAI 兼容协议:流式累积正文与工具调用增量
			tool_parts: list[dict[str, str]] = []
			stream = await client.chat.completions.create(
				model=self.cfg.model,
				temperature=self.cfg.temperature,
				messages=messages,  # type: ignore[arg-type]
				tools=tools or None,
				stream=True,
			)
			last_text = ''
			async for chunk in stream:
				if not chunk.choices:
					continue
				delta = chunk.choices[0].delta
				if delta.content:
					last_text += delta.content
					if on_event:
						await on_event({'type': 'token', 'content': delta.content})
				if delta.tool_calls:
					for tc in delta.tool_calls:
						idx = tc.index
						while len(tool_parts) <= idx:
							tool_parts.append({'id': '', 'name': '', 'arguments': ''})
						if tc.id:
							tool_parts[idx]['id'] = tc.id
						if tc.function:
							if tc.function.name:
								tool_parts[idx]['name'] += tc.function.name
							if tc.function.arguments:
								tool_parts[idx]['arguments'] += tc.function.arguments
			if not tool_parts:
				return last_text
			# 工具调用:回填 assistant 消息 → 执行工具 → 继续下一轮
			messages.append(
				{
					'role': 'assistant',
					'content': last_text or None,
					'tool_calls': [
						{
							'id': p['id'],
							'type': 'function',
							'function': {'name': p['name'], 'arguments': p['arguments']},
						}
						for p in tool_parts
					],
				}
			)
			for p in tool_parts:
				try:
					args = json.loads(p['arguments'] or '{}')
				except json.JSONDecodeError:
					args = {}
				if on_event:
					await on_event({'type': 'tool_call', 'name': p['name'], 'arguments': args})
				result = await self._run_tool(tool_registry, p['name'], args)
				if on_event:
					await on_event({'type': 'tool_result', 'name': p['name'], 'content': result})
				messages.append({'role': 'tool', 'tool_call_id': p['id'], 'content': result})
		return last_text

	def _provider_tools(self, names: list[str]) -> list[dict[str, Any]]:
		"""把工具名转成 provider 的工具 schema。schema 由调用方通过同名常量注入。"""
		from .tools import TOOL_SCHEMAS

		schemas = [s for s in TOOL_SCHEMAS if s.get('name') in names]
		if self.provider == 'anthropic':
			return [
				{
					'name': s['name'],
					'description': s.get('description', ''),
					'input_schema': s.get('parameters', {'type': 'object', 'properties': {}}),
				}
				for s in schemas
			]
		return [
			{
				'type': 'function',
				'function': {
					'name': s['name'],
					'description': s.get('description', ''),
					'parameters': s.get('parameters', {'type': 'object', 'properties': {}}),
				},
			}
			for s in schemas
		]


class ModelRegistry:
	"""按档位返回 ChatModel。"""

	def __init__(self, cfg: LlmConfig | None = None) -> None:
		self.cfg = cfg or get_config().llm
		self._models: dict[str, ChatModel] = {}

	def get(self, tier: str) -> ChatModel:
		tier_cfg: LlmTierConfig = getattr(self.cfg, tier)
		if tier not in self._models:
			self._models[tier] = ChatModel(tier, tier_cfg)
		return self._models[tier]

	def strong(self) -> ChatModel:
		return self.get('strong')

	def medium(self) -> ChatModel:
		return self.get('medium')

	def cheap(self) -> ChatModel:
		return self.get('cheap')


# 模块级单例
_REGISTRY: ModelRegistry | None = None


def get_registry(config: Config | None = None) -> ModelRegistry:
	global _REGISTRY
	if _REGISTRY is None:
		_REGISTRY = ModelRegistry(config.llm if config else None)
	return _REGISTRY


def chat(tier: str, messages: list[ChatMessage]) -> str:
	"""同步便捷入口(内部跑事件循环),供 CLI / 脚本使用。"""
	import asyncio

	return asyncio.run(get_registry().get(tier).acomplete(messages))
