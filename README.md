# 基于 LangGraph 的多智能体论文撰写系统（thesis-multiagent）

> 主从协同多智能体 + 本地 FAISS RAG + 三级记忆（短期/长期/自进化）+ 防幻觉引用治理的科研论文写作系统。

输入论文主题，系统自动完成 **规划 → 检索 → 撰写 → 评审 → 修订 → 收尾校验** 的完整闭环，产出论文全文、BibTeX 参考文献与一致性校验报告。

## 核心特性

### 1. 主从多智能体编排（LangGraph）
- 主编排（Orchestrator）不写内容，只负责任务分发与流程驱动，协调 5 个角色子 Agent：

| Agent | 职责 | 模型档位 |
|-------|------|----------|
| Planner | 生成章节大纲、展开任务板 | strong |
| Researcher | 本地 RAG + Web 学术检索，产出带溯源的素材 | medium |
| Drafter | 按章节撰写初稿（可并行多实例） | cheap |
| Critic | 结构化评审（逻辑/引用/结构） | strong |
| Editor | 整合章节、构建引用库、润色 | medium |

- 条件边 `Send` API 按章节依赖并行扇出撰写与评审；SQLite 任务板驱动"评审-修订"闭环（默认 3 轮，超限自动升级人工）
- `run_id` 代际隔离 + `AsyncSqliteSaver` Checkpointer，支持断点续跑

### 2. 本地 RAG 分层检索
- **入库管线**：PDF 解析（保留标题层级与内置元数据）→ 按段落边界分块（约 500 token，10% 重叠，句子对齐）→ BGE-M3 向量化（1024 维）→ FAISS 分片索引 + SQLite 元数据
- **分层检索**：FAISS 向量 + BM25 关键词双路召回（RRF 融合）Top-100 → 元数据过滤（年份/子领域）→ bge-reranker 精排 Top-20；关键词路补齐方法名/缩写等精确匹配盲区
- **增量索引**：新论文只重建所属分片，不触碰其他分片
- **Web 补充检索**：arxiv / Semantic Scholar 官方 API，指数退避重试 + 同源限速
- **自动全文获取**：无开放获取 PDF 的文献自动降级至文献互助渠道（见第 3 节）

### 3. 论文自动获取 MCP Server
无开放获取全文的文献（IEEE / Springer 等付费库），自动降级调用科研通文献互助平台：

- **触发**：Web 工作台「入库」时解析不到开放 PDF → 自动发布悬赏求助（DOI 智能提取，通常数十秒内被自动应助）
- **下载**：令牌化下载协议，高速通道优先（2.6MB 约 1.7s，积分 ≥500 免费）+ 三线路容错重试 + 大小完整性校验，校验通过才落盘
- **闭环**：应助文件自动落入 `data/papers/` 并复用既有管线入库 RAG，检索引用即刻可用
- **实现**：独立 FastMCP stdio server（`mcp_server/paper_downloader/`），登录态由本地 Cookie 文件承载（已 gitignore 不入库）；主项目经 MCP 客户端调用，亦可挂载到任意 MCP 宿主（Trae / Claude Desktop）

### 4. 三级记忆系统
| 层级 | 实现 | 用途 |
|------|------|------|
| 短期记忆 | 对话式上下文 + Checkpointer 持久化 | 修订轮次记忆接续；硬截断 + LLM 压缩摘要两级超窗控制 |
| 长期记忆 | SQLite 命名空间 KV（WAL 并发安全） | 引用库/反馈/经验跨会话存储 |
| 自进化记忆 | 评审教训沉淀 → 撰写时注入 | 按章节类型聚合经验，按重复频次注入提示词，避免重复犯错 |

### 5. 可复用技能包（Skills）
按需注入对应 agent 的知识/流程，与提示词解耦，均为零成本或低成本设计：

| 技能 | 注入点 | 作用 |
|------|--------|------|
| venue-style | Drafter | 按 `--venue` 匹配 CVPR/IEEE/学位论文等写作规范（时态/语态/篇幅/术语），YAML 知识包 + 纯规则匹配，零 LLM 成本 |
| query-expansion | Researcher | LLM 生成同义改写/缩写展开/相关术语 3 类查询变体，多路检索结果去重合并，失败自动回退原始查询 |
| terminology-consistency | Finalize | 规则检查全文缩写规范（使用未定义 / 重复定义），问题并入一致性报告 |

### 6. 防幻觉引用治理闭环
四道防线，强制每个论断有文献依据：
```
写前限制：Drafter 只能引用素材白名单内的论文
评审核查：Critic 携带素材逐条核对引用
汇总裁决：Editor 过滤离题论文，改写失败上报
收尾校验：规则引擎全文核对 引用 ↔ BibTeX 一一对应 + LLM 结构检查
```

### 7. Web 工作台
FastAPI 统一服务 + Vue 3 前端：实时任务看板（3 秒轮询任务板/记忆库）+ SSE 流式研究助手聊天（支持 `/search`、`/rag` 直调工具与结果收藏）。

## 架构总览

```mermaid
graph TD
    subgraph 编排层
        ORCH[Orchestrator<br/>StateGraph 状态机]
        TB[任务板 TaskBoard<br/>SQLite · run_id 隔离]
    end
    subgraph 子Agent层
        PL[Planner] --> RS[Researcher] --> DR[Drafter · 并行] --> CR[Critic] --> ED[Editor]
    end
    subgraph 记忆层
        ST[短期 Checkpointer]
        LT[长期 MemoryStore]
        EX[自进化经验库]
    end
    subgraph 数据层
        RAG[FAISS 分区 + 元数据]
        CIT[引用库 BibTeX]
    end
    ORCH --> TB
    ORCH --> PL
    DR --> ST
    DR --> EX
    RS --> RAG
    ED --> CIT
```

## 技术栈

Python 3.11+ · LangGraph · MCP (FastMCP) · FAISS · BM25 (rank-bm25) · BGE-M3 / bge-reranker · PyMuPDF · SQLite · FastAPI · Vue 3 / Vite · DeepSeek API（OpenAI 兼容协议）

## 快速开始

```powershell
# 1) 安装依赖（uv 管理，国内走阿里云镜像）
uv sync
.venv\Scripts\Activate.ps1

# 2) 配置密钥（DeepSeek 必填）
copy .env.example .env

# 3) 把论文 PDF 放入 data/papers/，入库建索引
thesis-agent ingest --subfield nlp

# 4) 运行论文撰写（相同 thread-id 可断点续跑）
thesis-agent run --topic "论文主题" --venue "目标会议" --thread-id run1

# 5) Web 工作台（默认 http://127.0.0.1:9000）
thesis-agent serve
```

产出物（`data/output/`）：
- `final_paper.md` — 最终论文
- `references.bib` — 参考文献
- `consistency_check.txt` — 一致性校验报告

可选：启用**论文自动获取**（无开放 PDF 时自动经科研通互助下载）——登录 [ablesci.com](https://www.ablesci.com/) 后，将浏览器 Cookie 粘贴到 `mcp_server/paper_downloader/.ablesci_cookie`（该文件已被 gitignore），Web 工作台「入库」即自动生效。

## 目录结构

```
thesis-multiagent/
├── config.yaml               # 模型分级 / RAG / 编排参数
├── src/thesis_agent/
│   ├── graph/                # LangGraph 编排（orchestrator + 7 类节点 + 任务板）
│   ├── rag/                  # PDF 入库 / FAISS 分片索引 / 分层检索
│   ├── memory/               # 长期记忆 / 对话记忆 / 自进化经验
│   ├── skills/               # 可复用技能包（venue 风格 YAML + 加载器）
│   ├── citations/            # BibTeX 管理与引用校验
│   ├── eval.py               # 离线评测（检索命中率 / 引用有效率 / 评审轮次分布）
│   ├── search/               # arxiv / Semantic Scholar 检索 + 论文下载 MCP 客户端
│   ├── llm/                  # 模型分级工厂 + 工具调用
│   ├── chat_api.py           # Web 聊天后端（SSE 流式）
│   └── dashboard.py          # FastAPI 运行看板
├── mcp_server/               # 论文下载 MCP server（科研通文献互助渠道，独立进程）
├── frontend/                 # Vue 3 工作台（看板 + 聊天）
├── scripts/                  # 入库 / 撰写 / 聊天 CLI 脚本
└── docs/                     # 技术文档 + 问题审查与修复记录
```

## 文档

- [技术文档](docs/技术文档.md) — 完整架构设计、记忆系统、RAG 管线、引用治理实现说明
- [问题审查与修复记录](docs/问题审查与修复记录.md) — 三轮审查迭代修复的 13 项工程问题（并发安全、引用治理、索引重建等）

## 说明

- 本项目仅用于学习与研究，生成内容需人工审核后方可使用
- Windows / WSL2 均可运行；FAISS 中文路径已做兼容处理
