"""自进化记忆:把评审教训沉淀为可复用的写作经验,并做反馈采纳追踪。

机制:
1. record_review_outcome:每次评审后,把章节类型的教训写入 writing_experience。
2. get_experiences:写作时检索同类章节的历史教训,注入提示词。
3. get_recurring_issues:统计同类问题重复出现次数,重复多则优先警示。
4. feedback 采纳追踪:记录每个问题是否在修订中被采纳。
"""
from __future__ import annotations

import hashlib
from typing import Any

from .store import MemoryStore

_NS_EXPERIENCE = 'writing_experience'
_NS_FEEDBACK = 'feedback'


def _lesson_key(chapter_kind: str) -> str:
	return f'kind:{chapter_kind}'


def _issue_key(chapter_id: str, issue: str) -> str:
	"""稳定 key:用 md5 代替内置 hash(PYTHONHASHSEED 跨进程随机化会让同一问题生成不同 key)。"""
	digest = hashlib.md5(issue.encode('utf-8')).hexdigest()[:12]
	return f'{chapter_id}:{digest}'


def record_review_outcome(
	store: MemoryStore,
	chapter_kind: str,
	approved: bool,
	issues: list[str],
	revision_count: int,
) -> None:
	"""把一次评审的教训写入经验库(按章节类型聚合)。"""
	if not issues:
		return
	key = _lesson_key(chapter_kind)
	entry = store.get(_NS_EXPERIENCE, key) or {'chapter_kind': chapter_kind, 'lessons': {}}
	lessons: dict[str, int] = entry.setdefault('lessons', {})
	for issue in issues:
		lesson = issue.strip()[:300]
		if not lesson:
			continue
		lessons[lesson] = lessons.get(lesson, 0) + 1
	entry['last_approved'] = approved
	entry['last_revision_count'] = revision_count
	store.put(_NS_EXPERIENCE, key, entry)


def get_experiences(store: MemoryStore, chapter_kind: str, limit: int = 5) -> list[str]:
	"""检索同类章节的历史教训,按重复次数排序。"""
	key = _lesson_key(chapter_kind)
	entry = store.get(_NS_EXPERIENCE, key)
	if not entry:
		return []
	lessons = entry.get('lessons', {})
	ranked = sorted(lessons.items(), key=lambda x: x[1], reverse=True)
	return [f'[{count}x] {lesson}' for lesson, count in ranked[:limit]]


def get_recurring_issues(store: MemoryStore, chapter_kind: str, min_count: int = 2) -> list[str]:
	"""返回重复出现 >= min_count 次的教训(高频问题优先警示)。"""
	key = _lesson_key(chapter_kind)
	entry = store.get(_NS_EXPERIENCE, key)
	if not entry:
		return []
	return [f'[{count}x] {lesson}' for lesson, count in entry.get('lessons', {}).items() if count >= min_count]


def record_feedback_adoption(store: MemoryStore, chapter_id: str, issues: list[str], adopted: bool) -> None:
	"""记录评审问题是否在修订中被采纳。"""
	for issue in issues:
		store.put(
			_NS_FEEDBACK,
			_issue_key(chapter_id, issue),
			{'chapter_id': chapter_id, 'issue': issue[:300], 'adopted': adopted},
		)


def feedback_summary(store: MemoryStore, chapter_id: str) -> dict[str, Any]:
	"""某章节的反馈采纳汇总。"""
	records = [r for r in store.list(_NS_FEEDBACK) if r.get('chapter_id') == chapter_id]
	adopted = sum(1 for r in records if r.get('adopted') is True)
	return {'total': len(records), 'adopted': adopted, 'pending': sum(1 for r in records if r.get('adopted') is None)}
