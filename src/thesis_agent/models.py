"""共享数据模型:引用、评审意见、章节大纲等。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
	"""一条文献引用(对应 BibTeX 条目)。"""

	key: str = ''  # citation key,如 'vaswani2017attention'(可为空,入库后生成)
	title: str
	authors: list[str] = Field(default_factory=list)
	year: int | None = None
	venue: str = ''
	doi: str = ''
	paper_id: str = ''  # 关联本地论文库的 paper_id(溯源用)
	abstract: str = ''
	extra: dict = Field(default_factory=dict)


class ReviewVerdict(BaseModel):
	"""评审结论。approved 通过;否则意见必须可操作。"""

	approved: bool
	score: int = Field(default=0, ge=0, le=10)
	issues: list[str] = Field(default_factory=list)  # 具体问题
	acceptance_checks: dict[str, bool] = Field(default_factory=dict)  # 验收标准逐项
	revision_suggestion: str = ''  # 给 Drafter 的修订建议


class FeedbackEntry(BaseModel):
	"""评审反馈 + 采纳追踪(自进化记忆的输入)。"""

	task_id: str
	chapter_id: str
	issue: str
	adopted: bool | None = None  # None=未决, True/False=是否采纳
	occurrences: int = 1  # 同类问题出现次数


class ChapterOutline(BaseModel):
	"""章节大纲条目。"""

	chapter_id: str  # 如 'intro' / 'method'
	title: str
	description: str
	kind: Literal['intro', 'related_work', 'method', 'experiments', 'conclusion', 'other'] = 'other'
	deps: list[str] = Field(default_factory=list)  # 依赖的章节 id(如 experiments 依赖 method)
	keywords: list[str] = Field(default_factory=list)  # 检索用关键词


class Outline(BaseModel):
	"""完整论文大纲:章节树 + 全局信息。"""

	topic: str
	venue: str
	chapters: list[ChapterOutline] = Field(default_factory=list)

	def chapter_ids(self) -> list[str]:
		return [c.chapter_id for c in self.chapters]
