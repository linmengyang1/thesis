"""本地 embedding 与 rerank 后端(基于 sentence-transformers)。

BGE-M3 生成 1024 维稠密向量;bge-reranker 作为 cross-encoder 精排。
模型首次使用会从 HuggingFace 下载;国内网络可在 .env 设置
HF_ENDPOINT=https://hf-mirror.com 走镜像。
"""
from __future__ import annotations

import os
import threading

from ..config import get_config


class Embedder:
	"""BGE-M3 稠密向量编码器,线程安全单例。"""

	_model = None
	_lock = threading.Lock()

	@classmethod
	def _load(cls):
		if cls._model is None:
			with cls._lock:
				if cls._model is None:
					from sentence_transformers import SentenceTransformer

					model_name = get_config().rag.embedding_model
					cls._model = SentenceTransformer(model_name, device='cpu')
		return cls._model

	@classmethod
	def embed(cls, texts: list[str]) -> list[list[float]]:
		"""批量编码,返回归一化后的向量列表。"""
		model = cls._load()
		vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
		return [list(map(float, v)) for v in vectors]

	@classmethod
	def embed_one(cls, text: str) -> list[float]:
		return cls.embed([text])[0]


class Reranker:
	"""bge-reranker cross-encoder 精排器。"""

	_model = None
	_lock = threading.Lock()

	@classmethod
	def _load(cls):
		if cls._model is None:
			with cls._lock:
				if cls._model is None:
					from sentence_transformers import CrossEncoder

					model_name = get_config().rag.rerank_model
					cls._model = CrossEncoder(model_name, max_length=512)
		return cls._model

	@classmethod
	def score(cls, query: str, documents: list[str]) -> list[float]:
		"""返回 query 与每个 document 的相关性分数(越高越相关)。"""
		model = cls._load()
		pairs = [(query, doc) for doc in documents]
		scores = model.predict(pairs, show_progress_bar=False)
		return [float(s) for s in scores]


def check_backend() -> None:
	"""健康检查:确认本地模型可加载(供阶段0验收用)。"""
	os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
	Embedder.embed_one('attention is all you need')
	print(f'embedding 后端 OK,向量维度={len(Embedder.embed_one("test"))}')
