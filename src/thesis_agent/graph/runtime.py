"""运行时服务容器:节点通过它访问 LLM 分级、RAG、任务板等共享服务。

节点函数保持纯净(只依赖 state + runtime),便于单测与替换。
"""
from __future__ import annotations

from ..config import Config, get_config
from ..llm.factory import ModelRegistry
from ..memory.store import MemoryStore
from ..rag.index import FaissShards
from ..rag.retrieve import Retriever
from .task_board import TaskBoard


class ThesisRuntime:
	def __init__(self, config: Config | None = None) -> None:
		self.config = config or get_config()
		self.registry = ModelRegistry(self.config.llm)
		self.index = FaissShards(self.config.index_path)
		self.retriever = Retriever(self.index)
		self.task_board = TaskBoard(self.config.db_file)
		self.store = MemoryStore()  # 长期记忆(跨会话)

	def close(self) -> None:
		self.index.close()
		self.task_board.close()
		self.store.close()
