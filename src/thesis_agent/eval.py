"""离线评测:检索命中率 / 引用有效率 / 评审轮次分布,输出汇总报告。

三个评测器相互独立,任一数据源缺失时跳过并说明原因:
1. eval_retrieval  —— 检索质量:HitRate@5/10/20 与 MRR。
   支持两种模式:自检索采样(无标注,用论文标题做查询,检查能否召回该论文自身)
   与自定义标注集(JSONL,每行 {"query": str, "relevant": [paper_id, ...]})。
2. eval_citations  —— 引用有效率:最新 final_paper.md + references.bib 逐组核对,
   统计引用组总数、无匹配引用明细、孤儿 BibTeX 条目。
3. eval_review     —— 评审轮次分布:任务板各状态计数与 revision_count 分布,
   反映"评审-修订"闭环的回炉成本。

用法:
    thesis-agent eval                      # 三项全跑(自检索默认采样 50 篇)
    thesis-agent eval --limit 100          # 提高自检索采样数
    thesis-agent eval --queries eval.jsonl # 使用自定义标注集
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

from .citations.bibtex import bibtex_to_citations, validate_citations_in_text
from .config import get_config

_KS = (5, 10, 20)


# ---------- 检索评测 ----------

def self_retrieval_queries(index: Any, limit: int, seed: int = 42) -> list[dict[str, Any]]:
    """自检索评测集:随机采样论文,以标题为查询、自身 paper_id 为相关结果。

    标题的措辞与论文 chunks 高度重叠,若连标题都召回不了自身,说明检索管线
    存在退化;该模式无需人工标注,适合作为回归基线。
    """
    papers = [p for p in index.meta.all_papers() if p.get('title')]
    rng = random.Random(seed)
    if len(papers) > limit:
        papers = rng.sample(papers, limit)
    return [{'query': p['title'], 'relevant': [p['paper_id']]} for p in papers]


def load_query_file(path: Path) -> list[dict[str, Any]]:
    """加载自定义标注集:JSONL,每行 {"query": str, "relevant": [paper_id, ...]}。"""
    queries: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        obj = json.loads(line)
        if obj.get('query') and obj.get('relevant'):
            queries.append({'query': obj['query'], 'relevant': list(obj['relevant'])})
    return queries


def eval_retrieval(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """跑检索评测:对每个查询取 top-max(ks) 结果,统计 HitRate@k 与 MRR。"""
    from .rag.index import FaissShards
    from .rag.retrieve import Retriever

    cfg = get_config()
    index = FaissShards(cfg.index_path)
    top_k = max(_KS)
    hits_at = {k: 0 for k in _KS}
    rr_sum = 0.0
    per_query: list[dict[str, Any]] = []
    try:
        retriever = Retriever(index)
        for q in queries:
            results = retriever.retrieve(q['query'], top_k=top_k)
            ranked_ids = list(dict.fromkeys(r.paper_id for r in results))
            relevant = set(q['relevant'])
            first_rank = next((i + 1 for i, pid in enumerate(ranked_ids) if pid in relevant), 0)
            for k in _KS:
                if any(pid in relevant for pid in ranked_ids[:k]):
                    hits_at[k] += 1
            rr_sum += (1.0 / first_rank) if first_rank else 0.0
            per_query.append({
                'query': q['query'][:60],
                'first_rank': first_rank,
                'returned': len(ranked_ids),
            })
    finally:
        index.close()

    n = max(1, len(queries))
    return {
        'num_queries': len(queries),
        'hit_rate': {f'@{k}': hits_at[k] / n for k in _KS},
        'mrr': rr_sum / n,
        'per_query': per_query,
    }


# ---------- 引用评测 ----------

def eval_citations(output_path: Path) -> dict[str, Any]:
    """引用有效率:正文引用组逐组核对 BibTeX;另统计未被正文引用的孤儿条目。"""
    paper_path = output_path / 'final_paper.md'
    bib_path = output_path / 'references.bib'
    if not paper_path.exists() or not bib_path.exists():
        return {'skipped': f'缺少 {paper_path.name} 或 {bib_path.name}(先运行 thesis-agent run 生成产物)'}

    text = paper_path.read_text(encoding='utf-8')
    citations = bibtex_to_citations(bib_path.read_text(encoding='utf-8'))
    keys = {c.key for c in citations if c.key}
    problems = validate_citations_in_text(text, citations)

    # 孤儿条目:BibTeX 里有、正文从未引用
    cited_keys: set[str] = set()
    for group in re.findall(r'\[([^\[\]]+)\]', text):
        for token in group.split(';'):
            token = token.strip().lstrip('@')
            if token in keys:
                cited_keys.add(token)
    orphans = sorted(keys - cited_keys)

    return {
        'paper': paper_path.name,
        'bibtex_entries': len(citations),
        'cited_unique': len(cited_keys),
        'invalid_groups': problems,
        'invalid_count': len(problems),
        'orphan_entries': orphans,
    }


# ---------- 评审分布评测 ----------

def eval_review(db_file: Path) -> dict[str, Any]:
    """评审轮次分布:任务板状态计数 + revision_count 分布(全部 run 合并)。"""
    if not db_file.exists():
        return {'skipped': f'任务板数据库不存在: {db_file}(先运行 thesis-agent run)'}
    conn = sqlite3.connect(str(db_file))
    try:
        rows = conn.execute('SELECT status, revision_count FROM tasks').fetchall()
    finally:
        conn.close()
    if not rows:
        return {'skipped': '任务板为空(先运行 thesis-agent run)'}

    status_count: dict[str, int] = {}
    rev_dist: dict[int, int] = {}
    for status, rev in rows:
        status_count[status] = status_count.get(status, 0) + 1
        rev_dist[int(rev)] = rev_dist.get(int(rev), 0) + 1
    draft_tasks = sum(v for k, v in status_count.items() if k != 'queued')
    total_rev = sum(k * v for k, v in rev_dist.items())
    return {
        'total_tasks': len(rows),
        'status_count': dict(sorted(status_count.items(), key=lambda x: -x[1])),
        'revision_dist': dict(sorted(rev_dist.items())),
        'avg_revisions': round(total_rev / max(1, draft_tasks), 2),
    }


# ---------- 汇总报告 ----------

def run_eval(retrieval_limit: int = 50, queries_file: Path | None = None) -> str:
    """跑全部评测并生成文本报告(同时打印与落盘 data/output/eval_report.txt)。"""
    cfg = get_config()
    lines: list[str] = ['===== thesis-agent 离线评测报告 =====']

    # 1) 检索
    lines.append('\n--- 1. 检索评测(HitRate@5/10/20 与 MRR) ---')
    try:
        queries = load_query_file(queries_file) if queries_file else None
        if queries is None:
            from .rag.index import FaissShards

            index = FaissShards(cfg.index_path)
            try:
                queries = self_retrieval_queries(index, retrieval_limit)
            finally:
                index.close()
            lines.append(f'模式: 自检索采样 {len(queries)} 篇(无标注,标题召回自身;--queries 可指定标注集)')
        else:
            lines.append(f'模式: 自定义标注集 {queries_file.name}({len(queries)} 条)')
        result = eval_retrieval(queries)
        lines.append(f"查询数: {result['num_queries']}")
        for k, v in result['hit_rate'].items():
            lines.append(f'  HitRate{k}: {v:.1%}')
        lines.append(f"  MRR: {result['mrr']:.3f}")
    except Exception as e:  # noqa: BLE001 - 单项失败不阻塞其他评测
        lines.append(f'[跳过] 检索评测失败: {type(e).__name__}: {e}')

    # 2) 引用
    lines.append('\n--- 2. 引用有效率 ---')
    try:
        result = eval_citations(cfg.output_path)
        if 'skipped' in result:
            lines.append(f"[跳过] {result['skipped']}")
        else:
            invalid = result['invalid_count']
            cited = result['cited_unique']
            rate = (1 - invalid / cited) * 100 if cited else 100.0
            lines.append(f"BibTeX 条目: {result['bibtex_entries']} | 正文唯一引用: {cited} | 引用有效率: {rate:.1f}%")
            lines.append(f"孤儿条目(库有正文未引): {len(result['orphan_entries'])}")
            for token, why in list(result['invalid_groups'].items())[:10]:
                lines.append(f'  [无效] {why}')
    except Exception as e:  # noqa: BLE001
        lines.append(f'[跳过] 引用评测失败: {type(e).__name__}: {e}')

    # 3) 评审分布
    lines.append('\n--- 3. 评审-修订轮次分布 ---')
    try:
        result = eval_review(cfg.db_file)
        if 'skipped' in result:
            lines.append(f"[跳过] {result['skipped']}")
        else:
            lines.append(f"任务总数: {result['total_tasks']} | 平均修订轮次: {result['avg_revisions']}")
            lines.append(f"状态分布: {result['status_count']}")
            lines.append(f"轮次分布(revision_count: 任务数): {result['revision_dist']}")
    except Exception as e:  # noqa: BLE001
        lines.append(f'[跳过] 评审评测失败: {type(e).__name__}: {e}')

    report = '\n'.join(lines)
    report_path = cfg.output_path / 'eval_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + '\n', encoding='utf-8')
    return report + f'\n\n(报告已保存: {report_path})'
