# ☄️ CoMeT — Cognitive Memory Tree

**Lossless structured memory for LLM agents. Summaries index; raw is preserved.**

Scattered conversations, tool outputs, and documents flow through a dual-speed sensor + compacter pipeline and land as navigable `MemoryNode`s — each holding a short summary, a retrieval trigger, lazy-generated detail, and the full raw content. Agents read at the shallowest tier that answers the question and only pay the token cost for depth they actually need.

CoMeT is the memory substrate for [CoBrA](https://github.com/Dirac-Robot/CoBrA) and is consumable standalone via Python API, LangChain tools, or an MCP server.

> Looking for a drop-in memory layer for Claude Code specifically?
> [**CoMeT-CC**](https://github.com/Dirac-Robot/comet-cc) runs as a local
> TLS proxy that CC routes through, so every outgoing `/v1/messages`
> request gets trim + retrieval injection without modifying Claude Code
> itself. Lighter than full CoMeT (single-session scope, no consolidation),
> zero API key — drives sensor/compacter via `claude -p` using your
> existing Claude subscription.

---

## Highlights

- **3-tier progressive retrieval** — summary → lazy detailed summary → raw
- **Dual-path RAG** — separate embeddings for `summary` (WHAT) and `trigger` (WHEN), fused with RRF + graph-aware 2-hop link traversal
- **LanceDB triple-table index** — disk-resident (mmap) vector store over summary / trigger / raw
- **Multi-modality compacting** — dialog, code artifacts, images, execution traces, external content each get tailored summary/trigger rules
- **Snapshot-protected consolidation** — dedup + cross-link + tag normalization + synthesis, all reversible on failure
- **Per-session briefs** — rule-hint-rationale guidance layer rewritten each compaction
- **Handoff / inherited memory** — curated IMPORTANCE:HIGH carry-over + per-chunk synthesis on session transition
- **Provider-agnostic LLMs** — OpenAI · Anthropic · Google · Ollama · vLLM, with short aliases
- **MCP server** — expose memory tools to any MCP-capable client via `comet-mcp`

---

## Architecture

```
User / tool / document
     │
     ▼
┌─────────────┐                      ┌───────────┐
│ CognitiveSensor │ ─────────────▶ │ L1 Buffer │
│  (SLM, fast)    │  load + flow    └─────┬─────┘
└─────────────┘                            │ topic_shift | high_load | buffer_overflow
                                           ▼
                                    ┌───────────┐
                                    │ Compacter │  Main LLM (modality-aware policy)
                                    └─────┬─────┘
                       summary + trigger + recall + importance + tags (+ brief)
                                           ▼
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                              ▼
            ┌──────────────┐                              ┌─────────────────┐
            │ MemoryStore  │  JSON KV + sessions +        │   VectorIndex   │  LanceDB
            │  (atomic)    │  briefs + inherited memory   │  summary/trig/  │  triple-table
            └──────┬───────┘  + snapshots                 │      raw        │
                   │                                      └────────┬────────┘
                   │                                               │
                   │                    ┌──────────────────────────┘
                   ▼                    ▼
             ┌─────────────┐    ┌──────────────┐     ┌──────────────┐
             │  Retriever  │◀──▶│  ScoreFusion │     │ Consolidator │
             │ QueryAnalyzer│   │  RRF + sim    │     │ dedup+link+  │
             │ + 2-hop link │   │  fusion_α     │     │ synthesize   │
             └─────────────┘    └──────────────┘     └──────────────┘
```

### Dual-Speed Layer

| Layer | Speed | Model | Output |
|-------|-------|-------|--------|
| **Sensor** | fast | `slm_model` | L1Memory (raw-pass-through) + `CognitiveLoad{logic_flow, load_level, redundancy_detected}` |
| **Compacter** | slow | `main_model` | `MemoryNode` with summary, trigger, recall_mode, importance, topic_tags, optional session_brief |

Compaction triggers when `buffer_size ≥ min_l1_buffer` **and** one of: `logic_flow == BROKEN` (topic shift) / `load_level ≥ load_threshold` / `buffer_size ≥ max_l1_buffer`. The originating reason is stored on the node (`compaction_reason`).

### 3-Tier Progressive Retrieval

| Tier | API | Content | Token Cost |
|------|-----|---------|------------|
| 1 | `retrieve()` / `retrieve_dual()` | summary + trigger + tags + links | minimal |
| 2 | `get_detailed_summary(node_id)` | 3–8-sentence detailed summary (lazy-generated via SLM, cached on node) | medium, one-time |
| 3 | `get_raw_content(node_id)` / `read_memory(node_id, depth=2)` | full original text | full |

Tier 2 is generated on first request from raw content and stored back on the node. Subsequent reads are free.

### MemoryNode Schema

```python
class MemoryNode(BaseModel):
    node_id: str                      # mem_YYYYMMDD_HHMMSS_<6hex>
    session_id: str | None
    depth_level: int                  # 0=summary, 1=detail, 2=raw/virtual
    recall_mode: Literal['passive', 'active', 'both']
    topic_tags: list[str]             # content tags + ORIGIN:/FLAG:/IMPORTANCE: meta
    summary: str                      # factual index, multi-topic separated by "; "
    detailed_summary: str | None      # Tier 2 cache
    trigger: str                      # "When I ..." — 2–4 anchors, retrieval-oriented
    content_key: str                  # pointer into raw store
    raw_location: str                 # path to raw .txt
    links: list[str]                  # cross-references to other node_ids
    source_links: list[str]           # file/image paths referenced by this node
    capsule: str                      # action-capsule prefix (e.g. "[ACT_FETCH] web_read (ok)")
    created_at: datetime
    compaction_reason: str | None     # topic_shift | high_load | buffer_overflow | forced | external
```

### Recall Mode

| Mode | Behavior |
|------|----------|
| `passive` | Always in the rendered context window |
| `active` | Retrieved on-demand via semantic search (default) |
| `both` | Always in context **and** searchable |

### Dual-Path RAG + Graph Traversal

1. **QueryAnalyzer** (SLM) decomposes the query → `semantic_query` (WHAT) + `search_intent` (WHEN) + `urgency` + `risk_level`
2. **Triple-table vector search** (LanceDB, cosine distance):
   - summary table ← `semantic_query`
   - trigger table ← `search_intent`
   - raw table ← both (fallback path)
3. **ScoreFusion** — `combined = 0.6 · RRF + 0.4 · max_similarity`; path weights split by `fusion_alpha` with `raw_search_weight` carved out
4. **Graph-aware re-ranking** — for each top-K result, follow outgoing `links` (2-hop, decay 0.5 / 0.25). Nodes referenced by multiple top-K results get a refcount bonus (0.3× sum)
5. `risk_level=high` attaches an inline warning to the retrieval tool's response, telling the agent summaries are likely insufficient and to open raw

Triggers are written from the agent's perspective (`"When I need to verify ..."`), not user-centric, so retrieval matches even without an explicit user query rephrase.

### Multi-Modality Compacting

`MemoryCompacter.compact(...)` accepts either a template name or a `policy` object with `modality ∈ {dialog, artifact_code, artifact_image, execution_trace, external}`. Each modality gets tailored summary / trigger / recall_mode instructions:

- **dialog** — factual index; retrieval scenarios; may emit a full-rewrite `session_brief`
- **artifact_code** — start with language/type; trigger anchors on exports to modify
- **artifact_image** — visual content, dimensions, format; trigger = visual-verification need
- **execution_trace** — tool name, success/failure, key values; trigger = verify exact results
- **external** — web search / doc ingest; active recall; URL/path preservation

Policy-driven compaction is used by CoBrA to label tool outputs correctly; standalone users can stick with the `compacting.txt` default template.

### Meta-Tag System

Alongside content tags, nodes carry structured meta-tags with fixed prefixes, priority-ranked for terse rendering:

| Axis | Prefix | Examples | Purpose |
|------|--------|----------|---------|
| Origin | `ORIGIN:` | `USER`, `WEB_SEARCH`, `FILE_EDIT`, `SUBAGENT_RESULT`, `PROJECT_GOAL` | Where the content came from |
| Action | `FLAG:ACT_` | `FAIL`, `EDIT`, `EXECUTE`, `DIAGNOSE`, `FETCH`, `PLAN`, `DECIDE`, `NONE` | What kind of action this turn represents |
| Kind | `FLAG:` | `SKILL`, `USER_REJECT`, `USER_FEEDBACK`, `PASSIVE` | Semantic flags |
| Importance | `IMPORTANCE:` | `HIGH`, `MED`, `LOW` | Prior on raw-reopen likelihood; drives handoff curation |

Only the highest-priority tag per axis is rendered in context windows (`(O:USER A:EDIT F:FEEDBACK I:H)`).

### Consolidation Pipeline

`consolidate()` runs (snapshot-protected — failure → full restore + index rebuild):

1. **Dedup** — if `1 − cosine ≥ 0.32`, merge newer into older (keep older `node_id`, bounded by `MAX_MERGE_PER_KEEPER=5`)
2. **Summary/trigger regeneration** — SLM rewrites the keeper's summary+trigger to cover both sources (`merge_summary.txt`)
3. **Cross-link** — if `1 − cosine ≥ 0.45` (and not dupes), add bidirectional `links`
4. **Tag normalization** — merge variants by case-insensitive / substring match (meta-prefixed tags excluded)
5. **Prune** — drop dangling link ids

### Synthesize — Virtual Knowledge Hubs

`synthesize()` is a separate cross-session pass that forms a chandelier of virtual parent nodes:

1. Union-Find over pairwise summary similarity (`≥ 0.22`, cluster size 2–8)
2. SLM validates each cluster is a coherent knowledge unit (`synthesis_validate.txt`)
3. SLM generates a unified summary + trigger (`synthesis_create.txt`)
4. New `MemoryNode` (depth 2) with bidirectional links to all sources, indexed into the vector store, carrying its source sessions' ids

### Sessions, Briefs, and Handoff

- **Sessions** — every node is tagged with a `session_id`. `sessions.json` tracks status (`active` / `closed`), timestamps, and `node_ids`. `list_sessions()` / `list_session_memories(session_id)` / `close_session()` are public.
- **Session briefs** — at most one per session, stored as `{base}/session_briefs/{session_id}.md`, full-rewritten on each DIALOG compaction (never appended). Fixed skeleton — `## Active Work Context`, `## Hints`. ≤ 1500 chars. Out-of-band rewrites go through `regenerate_brief(reason=...)`.
- **Inherited memory** — `save_inherited_memory(new_sid, source_sid, node_ids, synthesis_node_ids)` persists a curated IMPORTANCE:HIGH carry-over + per-chunk synthesis nodes at `{base}/inherited_memory/{sid}.json`. The successor renders each in its own harness block.
- **Pinned external nodes** — `pin_node(node_id)` surfaces cross-session nodes in the current session's `get_session_context()` output under an `[External Nodes]` header.

### Snapshot Protection

`MemoryStore.create_snapshot(label)` copies `index.json`, `sessions.json`, and all node files into `{base}/.snapshot/{label}/`. `consolidate()` and `synthesize()` wrap their mutations in a snapshot and restore-on-failure. On startup, `CoMeT._recover_pending_snapshots()` detects any leftover snapshots and rolls the store back + rebuilds the vector index.

---

## Quick Start

### Session memory

```python
from comet import CoMeT, scope

@scope
def main(config):
    memo = CoMeT(config)

    memo.add('B200 4대로 월드모델 학습 가능할까?')
    memo.add('2B면 충분하고 커봐야 8B')
    memo.add('DPO 데이터는 negative를 syntax error로 구성했어')
    memo.force_compact()

    for entry in memo.list_memories():
        print(memo.read_memory(entry['node_id'], depth=0))

    tools = memo.get_tools()
    # → get_memory_index, read_memory_node, search_memory, retrieve_memory (if retrieval enabled)

main()
```

### Cross-session RAG + 3-tier retrieval

```python
@scope
def main(config):
    memo = CoMeT(config)

    memo.add('JWT 액세스 15분, 리프레시 7일')
    memo.force_compact()

    # Tier 1: summaries
    results = memo.retrieve('토큰 만료 설정?')
    for r in results:
        print(f'[{r.node.node_id}] score={r.relevance_score:.3f} — {r.node.summary}')

    nid = results[0].node.node_id

    # Tier 2: lazy detailed summary
    print(memo.get_detailed_summary(nid))

    # Tier 3: raw
    print(memo.get_raw_content(nid))
```

### Documents + background ingestion

```python
# Sync
nodes = memo.add_document(web_page_text, source='https://example.com/article')

# Background (non-blocking) with completion callback
memo.add_document(
    long_pdf_text,
    source='paper.pdf',
    background=True,
    on_complete=lambda created: print(f'ingested {len(created)} nodes'),
)
memo.drain()  # wait for all queued jobs
```

Chunks split at paragraph → line → sentence boundaries with overlap; duplicates are detected via content hash.

### External content (tool outputs, web results)

```python
# Bypasses the L1 buffer — becomes an L2 node, linked to the next turn compaction
node = memo.add_external(
    tool_result_text,
    source_tag='WEB_SEARCH',
    template_name='compacting_external',
    source_links=['https://example.com/page'],
)
```

### Session lifecycle

```python
# Curated carry-over for a successor session
high = memo.list_high_importance_nodes(limit=10)

# Close + consolidate session-scoped dedup / links
result = memo.close_session()  # {'status': 'done', 'merged': ..., 'linked': ..., 'session_id': ...}

# Pin external nodes into another session's rendered context
other = CoMeT(config, session_id='successor')
for n in high:
    other.pin_node(n['node_id'])
print(other.get_session_context())  # includes [External Nodes] header
```

### Cross-session synthesis

```python
virtuals = memo.synthesize()  # cluster + SLM validate + virtual-node creation
```

---

## Agent tools

`memo.get_tools()` returns LangChain-compatible `BaseTool`s:

| Tool | Returns |
|------|---------|
| `get_memory_index()` | Rendered context window (passive/both first, then recent active) |
| `read_memory_node(node_id)` | Full raw data for a node |
| `search_memory(tag)` | Node ids matching a topic tag |
| `retrieve_memory(summary_query, trigger_query)` | Dual-path RAG with risk_level warning; returns summary/trigger/tags/links only (read the node for raw) |

The retrieval tool annotates the response with a `⚠️ HIGH RISK` prefix when `risk_level=high`, steering the agent toward `read_memory_node` when exact values / wording matter.

---

## MCP server

CoMeT ships an [MCP](https://modelcontextprotocol.io/) server for external agent integration:

```bash
pip install "comet-memory[mcp]"
comet-mcp        # from pyproject scripts
# or: python -m comet.mcp_server
```

Store location is configurable via `COMET_STORE_PATH`.

**Resources**

- `memory://nodes` — rendered memory index
- `memory://sessions` — session registry with metadata

**Tools** — `get_memory_index`, `read_memory_node`, `search_memory`, `retrieve_memory` (same semantics as the LangChain tools).

---

## Configuration ([ato](https://github.com/Dirac-Robot/ato))

```python
# comet/config.py — defaults
@scope.observe(default=True)
def default(comet: ADict):
    comet.slm_model  = 'gpt-4o-mini'     # Sensor, QueryAnalyzer, detail, synthesis, merge
    comet.main_model = 'gpt-4o'          # Compacter

    comet.llm = ADict(provider='openai')

    comet.language = 'the same language as the user'

    comet.compacting = ADict(
        load_threshold = 4,   # CognitiveLoad >= N triggers compaction
        max_l1_buffer  = 10,  # hard buffer cap
        # min_l1_buffer defaults to 3
    )

    comet.storage = ADict(
        type='json',
        base_path='./memory_store',
        raw_path='./memory_store/raw',
    )

    comet.consolidation = ADict(
        min_tag_overlap       = 2,
        cross_link_threshold  = 0.45,
        # cross_session_min_tag_overlap=1, cross_session_link_threshold=0.40
        # merge_threshold=0.32, cluster_threshold=0.22
    )

    comet.retrieval = ADict(
        embedding_model   = 'text-embedding-3-small',
        vector_backend    = 'lance',
        vector_db_path    = './memory_store/vectors',
        fusion_alpha      = 0.5,   # summary (α) vs trigger (1-α) weight
        rrf_k             = 5,
        raw_search_weight = 0.2,   # weight of raw-path signal in fusion
        top_k             = 5,
        rerank            = False,
    )
```

### Provider presets

```python
@scope.observe()
def anthropic(comet: ADict):
    comet.llm.provider = 'anthropic'
    comet.slm_model  = 'claude-3-5-haiku-latest'
    comet.main_model = 'claude-3-5-sonnet-latest'

@scope.observe()
def google(comet: ADict):
    comet.llm.provider = 'google'
    comet.slm_model = comet.main_model = 'gemini-2.0-flash'

@scope.observe()
def local_slm(comet: ADict):          # Ollama
    comet.llm.provider = 'ollama'
    comet.llm.base_url = 'http://localhost:11434/v1'
    comet.slm_model = 'gemma2:9b'
    comet.retrieval.embedding_provider = 'ollama'
    comet.retrieval.embedding_model    = 'nomic-embed-text'

@scope.observe()
def vllm(comet: ADict):
    comet.llm.provider = 'vllm'
    comet.llm.base_url = 'http://localhost:8000/v1'

@scope.observe()
def aggressive(comet: ADict):
    comet.compacting.load_threshold = 3
    comet.compacting.max_l1_buffer  = 5
```

Providers resolve from (1) short alias (`sonnet`, `opus`, `haiku`, `flash`, `pro`, `gpt`, `mini`, `codex`), (2) explicit `provider/model` prefix, (3) `config.llm.provider`, (4) `openai` fallback. Supported: `openai`, `anthropic`, `google`, `ollama`, `vllm`.

```bash
# Default
python main.py

# Anthropic + aggressive compaction
python main.py anthropic aggressive
```

---

## Benchmarks

Two suites under `benchmark/` compare CoMeT against a naive full-conversation-summary baseline, both using the judging-LLM pattern:

| Suite | File | Shape |
|-------|------|-------|
| Standard | `benchmark/run_benchmark.py` | 10 topics · short turns · one-hop recall |
| Hard | `benchmark/run_benchmark_hard.py` | 20 topics · needle-in-haystack · numerically adjacent distractors |

Both report `CoMeT vs baseline` accuracy and character-level compression. Run with:

```bash
python -m benchmark.run_benchmark
python -m benchmark.run_benchmark_hard
```

Results are written to `benchmark_results.json` / `benchmark_results_hard.json`.

---

## Project structure

```
comet/
├── orchestrator.py      # CoMeT — pipeline facade, sessions, pin, briefs, handoff, tools
├── sensor.py            # CognitiveSensor — L1 extract + load/flow/redundancy (SLM)
├── compacter.py         # MemoryCompacter — policy-aware L1→MemoryNode (Main LLM)
├── consolidator.py      # dedup + cross-link + tag-norm + synthesize (snapshot-protected)
├── retriever.py         # QueryAnalyzer + ScoreFusion + 2-hop link traversal
├── vector_index.py      # LanceDB triple-table (summary / trigger / raw) + rerank
├── storage.py           # JSON KV + sessions.json + session_briefs/ + inherited_memory/ + snapshots
├── schemas.py           # MemoryNode, CognitiveLoad, L1Memory, RetrievalResult
├── llm_factory.py       # Multi-provider chat/embed factory + model aliases
├── mcp_server.py        # FastMCP server — resources + tools
├── config.py            # ato scope presets (default / anthropic / google / local_slm / vllm / aggressive)
└── templates/
    ├── compacting.txt            # default dialog compacter
    ├── compacting_base.txt       # policy-injected base (modality-aware)
    ├── compacting_code.txt       # code-artifact modality
    ├── compacting_external.txt   # external content modality
    ├── compacting_file.txt       # file read/write modality
    ├── cognitive_load.txt        # sensor: logic_flow / load_level / redundancy
    ├── l1_extraction.txt         # L1 extraction (legacy — sensor now pass-through)
    ├── query_analysis.txt        # QueryAnalyzer decomposition
    ├── merge_summary.txt         # post-dedup summary+trigger regen
    ├── merge_trigger.txt         # (merge helper)
    ├── synthesis_validate.txt    # cluster coherence check
    ├── synthesis_create.txt      # virtual node generation
    └── consolidation_assessment.txt  # redundancy trigger for auto-consolidation

benchmark/
├── run_benchmark.py / run_benchmark_hard.py
└── synthetic_data{,_hard,_ultra}.py

tests/                   # test_comet, test_rag, test_comparison, test_extended,
                         # test_agent_retrieval, test_snapshot, test_storage_bugs
```

---

## Installation

```bash
pip install comet-memory                       # core (OpenAI + LanceDB)
pip install "comet-memory[providers]"          # + Anthropic, Google
pip install "comet-memory[mcp]"                # + FastMCP server
pip install "comet-memory[providers,mcp]"      # everything
```

Python ≥ 3.11. License: [PolyForm Noncommercial 1.0.0](LICENSE) — free for
personal, research, and nonprofit use; commercial use requires a separate
license from the author.
