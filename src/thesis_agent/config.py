"""项目配置:加载 config.yaml + .env,提供类型化访问与路径解析。

优先级:config.yaml 为基准,.env 中的密钥,环境变量可覆盖模型名。
"""
from __future__ import annotations

from pathlib import Path

import os

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# huggingface.co 在国内直连不稳定,默认走镜像;已在环境变量设置时尊重用户配置
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 模型权重已本地缓存完整,默认完全离线加载,避免 huggingface_hub 的 xet 网络校验
# 在国内直连官方 CDN 时长时间卡顿;进程级设置,不影响系统环境变量与其他项目
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# src/thesis_agent/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LlmTierConfig(BaseModel):
	"""单档模型配置。provider: anthropic | openai | deepseek。"""

	provider: str = 'deepseek'
	model: str = 'deepseek-chat'
	temperature: float = 0.7
	api_key: str | None = None  # 优先用档位内配置,缺省回退到环境变量


class LlmConfig(BaseModel):
	strong: LlmTierConfig = Field(default_factory=lambda: LlmTierConfig(model='claude-sonnet-5', provider='anthropic'))
	medium: LlmTierConfig = Field(default_factory=lambda: LlmTierConfig())
	cheap: LlmTierConfig = Field(default_factory=lambda: LlmTierConfig(temperature=0.8))


class RagConfig(BaseModel):
	chunk_size_tokens: int = 500
	chunk_overlap: float = 0.1
	top_k_candidate: int = 100
	top_k_final: int = 20
	embedding_model: str = 'BAAI/bge-m3'
	rerank_model: str = 'BAAI/bge-reranker-base'
	shards: int = 8


class SearchConfig(BaseModel):
	arxiv_api: str = 'https://export.arxiv.org/api/query'
	semantic_scholar_api: str = 'https://api.semanticscholar.org/graph/v1/paper/search'
	max_results: int = 10


class OrchestratorConfig(BaseModel):
	max_review_rounds: int = 3
	context_window_tokens: int = 15000


class ThesisConfig(BaseModel):
	topic: str = ''
	venue: str = ''
	language: str = 'en'


class Config(BaseModel):
	"""顶层配置,路径字段均为相对项目根目录的字符串。"""

	data_dir: str = 'data'
	papers_dir: str = 'data/papers'
	index_dir: str = 'data/index'
	db_path: str = 'data/db/thesis.db'
	output_dir: str = 'data/output'
	experiments_dir: str = 'data/experiments'
	llm: LlmConfig = Field(default_factory=LlmConfig)
	rag: RagConfig = Field(default_factory=RagConfig)
	search: SearchConfig = Field(default_factory=SearchConfig)
	orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
	thesis: ThesisConfig = Field(default_factory=ThesisConfig)

	def _resolve(self, rel: str) -> Path:
		p = Path(rel)
		return p if p.is_absolute() else PROJECT_ROOT / p

	@property
	def papers_path(self) -> Path:
		return self._resolve(self.papers_dir)

	@property
	def index_path(self) -> Path:
		return self._resolve(self.index_dir)

	@property
	def db_file(self) -> Path:
		return self._resolve(self.db_path)

	@property
	def output_path(self) -> Path:
		return self._resolve(self.output_dir)

	@property
	def experiments_path(self) -> Path:
		return self._resolve(self.experiments_dir)

	@classmethod
	def load(cls) -> 'Config':
		"""从项目根 config.yaml 加载,并用 .env / 环境变量覆盖模型名。"""
		load_dotenv(PROJECT_ROOT / '.env')
		config_file = PROJECT_ROOT / 'config.yaml'
		data: dict = {}
		if config_file.exists():
			data = yaml.safe_load(config_file.read_text(encoding='utf-8')) or {}
		cfg = cls.model_validate(data)
		# 环境变量覆盖模型名
		overrides = {
			'strong': 'THESIS_STRONG_MODEL',
			'medium': 'THESIS_MEDIUM_MODEL',
			'cheap': 'THESIS_CHEAP_MODEL',
		}
		for tier, env_name in overrides.items():
			val = os.getenv(env_name)
			if val:
				setattr(getattr(cfg.llm, tier), 'model', val)
		return cfg


# 模块级单例(懒加载避免 import 开销)
_CONFIG: Config | None = None


def get_config() -> Config:
	global _CONFIG
	if _CONFIG is None:
		_CONFIG = Config.load()
	return _CONFIG
