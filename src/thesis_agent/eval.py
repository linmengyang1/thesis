"""离线评测:检索命中率 / 引用有效率 / 评审轮次分布,输出汇总报告。

四个评测器相互独立,任一数据源缺失时跳过并说明原因:
1. eval_retrieval  —— 检索质量:HitRate@5/10/20 与 MRR,双路(hybrid)与单路(single) A/B 对比。
   三种标注模式按优先级自动选择:
   a) 自定义标注集(--queries,JSONL 每行 {"query": str, "relevant": [paper_id, ...]});
   b) 引用图弱标注(无人工标注时):query=论文标题,relevant=自身 + 库内被其引用的论文
      (通过 DOI 在论文全文中的出现自动建立引用关系);
   c) 自检索采样(兜底):用论文标题召回自身,作为回归基线。
2. eval_citations  —— 引用有效率:最新 final_paper.md + references.bib 逐组核对,
   统计引用组总数、无匹配引用明细、孤儿 BibTeX 条目。
3. eval_review     —— 评审轮次分布:任务板各状态计数与 revision_count 分布,
   反映"评审-修订"闭环的回炉成本。

用法:
    thesis-agent eval                      # 三项全跑(检索评测自动 A/B 对比)
    thesis-agent eval --limit 100          # 提高自检索兜底模式的采样数
    thesis-agent eval --queries eval.jsonl # 使用自定义标注集
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

from .citations.bibtex import bibtex_to_citations, validate_citations_in_text
from .config import get_config

_KS = (5, 10, 20)

# 参考文献里的 DOI 形态(扫描论文全文,建立引用图弱标注)
_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"\'<>]+')


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


def _norm_text(text: str) -> str:
    """归一化:小写 + 压缩连续空白(PDF 提取的断行/多空格统一)。"""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def citation_graph_queries(index: Any, max_queries: int = 200) -> list[dict[str, Any]]:
    """引用图弱标注:利用库内论文的互引关系自动生成标注对(DOI + 标题双通道)。

    对每篇论文 A:其全文中出现其他库内论文 B 的 DOI 或 标题(归一化子串匹配)
    → 视为 A 引用了 B → 生成 {query: A.title, relevant: 被引的库内论文}。
    排除查询论文自身:标题词与自身 chunk 高度重叠会掩盖区分度,排除后
    才是真正有挑战的跨论文检索测试。弱标注偏置:引用了不等于强相关,
    但作为跨论文召回的代理指标远强于自检索模式。
    """
    papers = index.meta.all_papers()
    by_doi = {p['doi'].lower(): p for p in papers if p.get('doi')}
    title_by_paper = {p['paper_id']: _norm_text(p['title']) for p in papers if p.get('title')}
    # 短标题子串误匹配风险高,只保留足够长的标题参与匹配
    matchable = {pid: t for pid, t in title_by_paper.items() if len(t) >= 25}

    text_by_paper: dict[str, str] = {}
    for r in index.meta.all_chunks_light():
        text_by_paper[r['paper_id']] = text_by_paper.get(r['paper_id'], '') + '\n' + r['text']

    queries: list[dict[str, Any]] = []
    for a in papers:
        if not a.get('title'):
            continue
        a_text = _norm_text(text_by_paper.get(a['paper_id'], ''))
        if not a_text:
            continue
        found = {m.group(0).lower().rstrip('.,;)') for m in _DOI_RE.finditer(a_text)}
        relevant = set()
        for doi in found:
            if doi in by_doi:
                relevant.add(by_doi[doi]['paper_id'])
        for pid, title in matchable.items():
            if pid != a['paper_id'] and title in a_text:
                relevant.add(pid)
        relevant.discard(a['paper_id'])
        if relevant:
            queries.append({'query': a['title'], 'relevant': sorted(relevant)})
        if len(queries) >= max_queries:
            break
    return queries


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


def eval_retrieval(queries: list[dict[str, Any]], hybrid: bool = True) -> dict[str, Any]:
    """跑检索评测:对每个查询取 top-max(ks) 结果,统计 HitRate@k 与 MRR。

    hybrid=True 走 BM25+向量双路召回;False 仅向量单路,用于 A/B 对比。
    """
    from .rag.index import FaissShards
    from .rag.retrieve import Retriever

    cfg = get_config()
    index = FaissShards(cfg.index_path)
    top_k = max(_KS)
    hits_at = {k: 0 for k in _KS}
    rr_sum = 0.0
    per_query: list[dict[str, Any]] = []
    try:
        retriever = Retriever(index, hybrid=hybrid)
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

def run_eval(retrieval_limit: int = 50, queries_file: Path | None = None, skip_llm: bool = False) -> str:
    """跑全部评测并生成文本报告(同时打印与落盘 data/output/eval_report.txt)。"""
    cfg = get_config()
    lines: list[str] = ['===== thesis-agent 离线评测报告 =====']

    # 1) 检索(标注模式自动选择 + 双路/单路 A/B 对比)
    lines.append('\n--- 1. 检索评测(HitRate@5/10/20 与 MRR,双路 vs 单路 A/B) ---')
    try:
        from .rag.index import FaissShards

        if queries_file:
            queries = load_query_file(queries_file)
            mode_desc = f'自定义标注集 {queries_file.name}({len(queries)} 条)'
        else:
            index = FaissShards(cfg.index_path)
            try:
                queries = citation_graph_queries(index)
                if len(queries) >= 5:
                    mode_desc = '引用图弱标注(query=标题,relevant=库内被引论文,DOI/标题匹配,排除自身)'
                else:
                    queries = self_retrieval_queries(index, retrieval_limit)
                    mode_desc = f'自检索采样 {len(queries)} 篇(兜底回归基线)'
            finally:
                index.close()

        results_by_mode = {}
        for name, hybrid in (('双路(hybrid)', True), ('单路(single)', False)):
            results_by_mode[name] = eval_retrieval(queries, hybrid=hybrid)

        # 对比表(含提升幅度)
        hy, sg = results_by_mode['双路(hybrid)'], results_by_mode['单路(single)']
        lines.append(f'标注模式: {mode_desc} | 查询数: {len(queries)}')
        header = f"{'方案':<14}" + ''.join(f'{f"HitRate@{k}":>12}' for k in _KS) + f"{'MRR':>10}"
        lines.append(header)
        for name, r in results_by_mode.items():
            row = f'{name:<14}' + ''.join(f'{r["hit_rate"][f"@{k}"]:>11.1%}' for k in _KS) + f'{r["mrr"]:>10.3f}'
            lines.append(row)
        delta = f"{'提升(双路-单路)':<12}" + ''.join(
            f'{(hy["hit_rate"][f"@{k}"] - sg["hit_rate"][f"@{k}"]) * 100:>+11.1f}pp' for k in _KS
        ) + f'{hy["mrr"] - sg["mrr"]:>+10.3f}'
        lines.append(delta)
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

    # 4) 生成质量(LLM-as-judge)+ 5) Critic 检出率:都需要 LLM,合并进同一个事件循环
    if not skip_llm:

        async def _llm_evals() -> tuple[dict[str, Any], dict[str, Any]]:
            from .graph.runtime import ThesisRuntime

            rt = ThesisRuntime()
            quality = await eval_quality(cfg.output_path, rt)
            critic = await eval_critic_robustness(cfg.output_path, rt)
            return quality, critic

        try:
            quality, critic = asyncio.run(_llm_evals())

            # 渲染第 4 节:生成质量
            sec = ['\n--- 4. 生成质量(LLM-as-judge,各维 1-5 分) ---']
            if 'skipped' in quality:
                sec.append(f"[跳过] {quality['skipped']}")
            else:
                sec.append(
                    f"章节均分: structure={quality['avg']['structure']:.2f} "
                    f"argumentation={quality['avg']['argumentation']:.2f} "
                    f"language={quality['avg']['language']:.2f}"
                )
                for name, s in quality['chapters']:
                    sec.append(
                        f"  - {name[:44]}: structure={s['structure']} argumentation={s['argumentation']} language={s['language']}"
                    )
            lines.extend(sec)

            # 渲染第 5 节:Critic 检出率
            sec = ['\n--- 5. Critic 检出率(预埋缺陷测试) ---']
            if 'skipped' in critic:
                sec.append(f"[跳过] {critic['skipped']}")
            else:
                if not critic['baseline_approved']:
                    sec.append('注意: 对照组原文被拒(历史产物引用标记与 BibTeX key 不一致),检出判定仍以关键词匹配为准')
                sec.append(f"对照组原文: approved={critic['baseline_approved']}")
                for d, v in critic['detections'].items():
                    sec.append(f"  [{d}] 检出={'✓' if v else '✗'}")
                sec.append(f"检出率: {critic['detected_count']}/{critic['total']}")
            lines.extend(sec)
        except Exception as e:  # noqa: BLE001
            lines.append(f'\n--- 4/5. LLM 评测(生成质量 + Critic 检出率) ---\n[跳过] 失败: {type(e).__name__}: {e}')

    report = '\n'.join(lines)
    report_path = cfg.output_path / 'eval_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + '\n', encoding='utf-8')
    return report + f'\n\n(报告已保存: {report_path})'


# ---------- 生成质量评测(LLM-as-judge) ----------

_JUDGE_SYSTEM = """You are an expert reviewer scoring a thesis chapter on 3 dimensions, each 1-5:
- structure: coherent paragraphs and clear logical flow
- argumentation: claims supported by reasoning and citations, adequate depth
- language: academic tone, fluency, grammar
Return JSON ONLY: {"structure": n, "argumentation": n, "language": n, "comment": "one sentence"}"""


def _split_chapters(paper_text: str, min_chars: int = 800) -> list[tuple[str, str]]:
    """把 final_paper.md 按 '## ' 切成 (章节名, 正文);排除 References 与过短段。"""
    out: list[tuple[str, str]] = []
    for part in re.split(r'^## ', paper_text, flags=re.M):
        if '\n' not in part:
            continue
        title, _, body = part.partition('\n')
        title = title.strip()
        if 'reference' in title.lower() or len(body) < min_chars:
            continue
        out.append((title, body.strip()))
    return out


async def eval_quality(output_path: Path, rt: Any) -> dict[str, Any]:
    """LLM-as-judge:strong 模型按 rubric 给每个章节打分,返回各维均分与逐章明细。"""
    paper_path = output_path / 'final_paper.md'
    if not paper_path.exists():
        return {'skipped': f'缺少 {paper_path.name}(先运行 thesis-agent run 生成产物)'}
    chapters = _split_chapters(paper_path.read_text(encoding='utf-8'))
    if not chapters:
        return {'skipped': '未能从产物中切分出章节'}

    from .graph.utils import extract_json

    llm = rt.registry.strong()
    chapters_out: list[tuple[str, dict[str, Any]]] = []
    sums = {'structure': 0.0, 'argumentation': 0.0, 'language': 0.0}
    for name, body in chapters:
        resp = await llm.acomplete(
            [
                {'role': 'system', 'content': _JUDGE_SYSTEM},
                {'role': 'user', 'content': f'Chapter: {name}\n\n{body[:6000]}\n\nScores:'},
            ]
        )
        data = extract_json(resp) or {}
        s = {
            'structure': int(data.get('structure') or 0),
            'argumentation': int(data.get('argumentation') or 0),
            'language': int(data.get('language') or 0),
        }
        chapters_out.append((name, s))
        for k in sums:
            sums[k] += s[k]
    n = max(1, len(chapters_out))
    return {
        'chapters': chapters_out,
        'avg': {k: v / n for k, v in sums.items()},
    }


# ---------- Critic 检出率评测 ----------

# 缺陷注入模板:在章节末尾追加一段含特定缺陷的文字
_DEFECT_TEMPLATES = {
    'ungrounded': 'Recent extensive experiments demonstrate that the proposed approach uniformly outperforms all state-of-the-art methods by a significant margin across every benchmark.',
    'fake_citation': 'Furthermore, end-to-end parsing frameworks have been shown to generalize robustly across domains [zhou2024nonexistent].',
    'contradiction': 'However, the overall accuracy decreases substantially compared with the baseline methods reported above, and the improvement is negligible.',
}

# 各缺陷的检出关键词(在 Critic 的 issues/suggestion 文本中匹配)
_DETECT_KEYWORDS = {
    'ungrounded': ('ungrounded', 'without citation', 'no citation', 'missing citation', 'unsupported', 'not cited', 'lacks citation'),
    'fake_citation': ('invented', 'not in the whitelist', 'not in whitelist', 'not appear', 'nonexistent', 'fake', 'not found', 'absent'),
    'contradiction': ('contradict', 'inconsisten', 'conflict'),
}


def _detects(defect_type: str, verdict_text: str) -> bool:
    text = verdict_text.lower()
    return any(k in text for k in _DETECT_KEYWORDS[defect_type])


async def eval_critic_robustness(output_path: Path, rt: Any, repeat: int = 1) -> dict[str, Any]:
    """Critic 一致性测试:对照(原文)+ 三类预埋缺陷,统计检出率。

    缺陷类型:ungrounded(无引用强断言)/ fake_citation(不在白名单的假引用)/
    contradiction(前后矛盾)。检出判定:Critic 的 issues/suggestion 文本
    命中对应关键词。对照组预期 approved=True(原文无缺陷)。
    """
    paper_path = output_path / 'final_paper.md'
    if not paper_path.exists():
        return {'skipped': f'缺少 {paper_path.name}(先运行 thesis-agent run 生成产物)'}
    chapters = _split_chapters(paper_path.read_text(encoding='utf-8'))
    if not chapters:
        return {'skipped': '未能从产物中切分出章节'}

    from .graph.nodes.review import _CRITIC_SYSTEM
    from .graph.utils import extract_json

    # 取最长章节作为被测样本;白名单用 references.bib 的 key
    _, draft = max(chapters, key=lambda c: len(c[1]))
    bib_path = output_path / 'references.bib'
    whitelist = ''
    if bib_path.exists():
        keys = [c.key for c in bibtex_to_citations(bib_path.read_text(encoding='utf-8')) if c.key]
        whitelist = 'Citation whitelist:\n' + '\n'.join(keys[:40])

    async def run_critic(text: str) -> dict[str, Any]:
        resp = await rt.registry.strong().acomplete(
            [
                {'role': 'system', 'content': _CRITIC_SYSTEM},
                {
                    'role': 'user',
                    'content': f'Chapter description: related work survey\n{whitelist}\n\nChapter draft:\n{text}\n\nVerdict:',
                },
            ]
        )
        return extract_json(resp) or {}

    # 对照组
    baseline = await run_critic(draft)
    baseline_approved = bool(baseline.get('approved'))

    detections: dict[str, bool] = {}
    for defect, extra in _DEFECT_TEMPLATES.items():
        detected = 0
        for _ in range(max(1, repeat)):
            verdict = await run_critic(draft + '\n\n' + extra)
            issues_text = ' '.join(str(i) for i in (verdict.get('issues') or [])) + ' ' + str(verdict.get('revision_suggestion', ''))
            if _detects(defect, issues_text):
                detected += 1
        detections[defect] = detected >= max(1, repeat)

    total = len(detections)
    return {
        'baseline_approved': baseline_approved,
        'detections': detections,
        'detected_count': sum(detections.values()),
        'total': total,
    }
