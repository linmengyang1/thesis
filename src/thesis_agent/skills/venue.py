"""Venue 风格技能:按目标会议/期刊加载写作规范,注入 Drafter 提示词。

知识包为 skills/venue_styles.yaml;匹配规则:venue 字符串(小写)含某条目的
keywords 之一即命中,均未命中回退 default。设计为纯规则匹配 + 零 LLM 成本。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_YAML_PATH = Path(__file__).resolve().parent / 'venue_styles.yaml'

_SYSTEMATIC_KEYS = ('tense', 'voice', 'length_hint', 'citations', 'structure', 'terminology')
_LABELS = {
    'tense': 'Tense',
    'voice': 'Voice',
    'length_hint': 'Length',
    'citations': 'Citations',
    'structure': 'Structure',
    'terminology': 'Terminology',
}


@lru_cache(maxsize=1)
def _load_styles() -> dict[str, Any]:
    with _YAML_PATH.open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def match_venue(venue: str) -> str:
    """venue 字符串匹配风格条目名,未命中返回 'default'。"""
    v = (venue or '').lower()
    if not v:
        return 'default'
    for name, style in _load_styles().items():
        if name == 'default':
            continue
        for kw in style.get('keywords', []):
            if str(kw).lower() in v:
                return name
    return 'default'


def load_venue_guidelines(venue: str) -> str:
    """返回格式化的 venue 写作规范文本块(英文,直接可拼进 Drafter 提示词)。"""
    styles = _load_styles()
    name = match_venue(venue)
    style = styles.get(name) or styles.get('default') or {}
    lines = [f'[venue: {name}]']
    for key in _SYSTEMATIC_KEYS:
        if style.get(key):
            lines.append(f'- {_LABELS[key]}: {style[key]}')
    return '\n'.join(lines)
