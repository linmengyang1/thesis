"""节点通用工具:LLM 输出的 JSON 提取与容错解析。"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
	"""从 LLM 输出中提取第一个合法 JSON 对象(容忍 markdown 代码块包裹)。"""
	if not text:
		return None
	# 去掉 markdown 代码块围栏
	text = re.sub(r'```(?:json)?', '', text).strip()
	for match in re.finditer(r'\{', text):
		try:
			obj, _ = json.JSONDecoder().raw_decode(text[match.start() :])
			if isinstance(obj, dict):
				return obj
		except (json.JSONDecodeError, ValueError):
			continue
	return None


def extract_list(text: str, key: str) -> list[Any]:
	"""从 LLM 输出的 JSON 中取指定 key 的列表。"""
	obj = extract_json(text)
	if obj is None:
		return []
	value = obj.get(key)
	return value if isinstance(value, list) else []
