# ☄️ CoMeT — Cognitive Memory Tree

**Lossless structured memory for AI agents.**

> **Recent Updates**  
> - 🚀 **3-Tier Progressive Retrieval**: Short summary → Lazy detailed summary → Raw content  
> - 🔗 **[GCRI](https://github.com/Dirac-Robot/GCRI) Integration**: In-session memory for multi-agent reasoning with auto-ingest  
> - 📄 **Document Ingestion**: `add_document()` for chunked ingestion of large texts  

CoMeT compresses long conversations into a navigable tree of memory nodes.  
Unlike naive summarization that loses details, CoMeT preserves full raw content behind structured summaries — agents read summaries first, then progressively drill deeper only when needed.

## Architecture

```
User Input / Document
     │
     ▼
┌─────────┐    SLM (fast)     ┌───────────┐
│  Sensor  │ ───────────────▶ │ L1 Buffer │
└─────────┘   entity/intent   └─────┬─────┘
                                    │ cognitive load trigger
                                    ▼
                              ┌───────────┐
                              │ Compacter │  LLM (slow)
                              └─────┬─────┘
                                    │ summary + trigger + recall_mode + tags
                                    ▼
                         ┌──────────┴──────────┐
                         │                     │
                   ┌───────────┐        ┌─────────────┐
                   │   Store   │        │ VectorIndex │  ChromaDB
                   │  depth 0-2│        │  full raw   │  summary + trigger
                   └───────────┘        └──────┬──────┘
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              ┌───────────────────────────────┐
                              │         3-Tier Retrieval       │
                              │  T1: Summary  (always cached)  │
                              │  T2: Detailed  (lazy, on-demand)│
                              │  T3: Raw       (full original)  │
                              └───────────────────────────────┘
```

### Dual-Speed Layer
- **Fast (Sensor)**: SLM extracts entities/intent per turn, detects topic shifts via cognitive load assessment
- **Slow (Compacter)**: Main LLM structures accumulated L1 buffer into `MemoryNode` with summary, trigger, recall mode, and topic tags

### 3-Tier Progressive Retrieval

Agents retrieve information at increasing depth, paying token cost only when needed:

| Tier | Method | Content | Token Cost |
|------|--------|---------|------------|
| 1 | `retrieve` | Short summary + trigger + node_id | Minimal |
| 2 | `get_detailed_summary` | 3–8 sentence detailed summary | Medium (lazy-generated, then cached) |
| 3 | `get_raw_content` | Full original content | Full |

**Lazy Detailed Summary**: Tier 2 summaries are generated on first request from raw content via SLM, then cached in the node. Subsequent calls return the cached version at zero additional cost.

### Recall Mode

Each memory node is classified by `recall_mode` at compaction time:

| Mode | Behavior | Examples |
|------|----------|----------|
| `passive` | Always included in context window | User identity, persistent preferences |
| `active` | Retrieved on-demand via semantic search | Factual details, decisions, events |
| `both` | Always in context + searchable via RAG | Core constraints with retrievable details |

### Dual-Path RAG Retrieval

CoMeT embeds both `summary` (what the node contains) and `trigger` (when to recall it) into separate vector collections. At query time:

1. **QueryAnalyzer** decomposes the query into `semantic_query` + `search_intent`
2. **Summary path**: matches what the information is about
3. **Trigger path**: matches when the information would be needed
4. **ScoreFusion** (Reciprocal Rank Fusion): merges results from both paths

Triggers are written from the **LLM's perspective** (`"내가 ~정보가 필요할 때"`) rather than user-centric, enabling broader semantic matching even without explicit user requests.

### Document Ingestion

Large documents and tool outputs can be ingested directly via `add_document()`:

```python
nodes = memo.add_document(
    content=long_text,
    source='tool:search_web',
    chunk_size=2000,
    chunk_overlap=200
)
```

Text is split into overlapping chunks at sentence/line boundaries, each processed through the Sensor → Compacter pipeline. Full raw content is stored in the vector store without truncation.

### Consolidation

Cross-session deduplication, linking, and tag normalization:

1. **Dedup**: Detect and merge semantically similar nodes
2. **Cross-link**: Create bidirectional links between related (non-duplicate) nodes
3. **Tag normalization**: Unify variant tags that refer to the same concept

### Topic-Aware Auto-Linking
Nodes share a global topic tag set. The compacter reuses existing tags when possible, enabling automatic bidirectional linking between related nodes across different conversation segments.

## Benchmark (52 turns, 5 conversations, 10 questions)

| Method | Context Cost | Accuracy |
|--------|-------------|----------|
| Full Context Injection | 5,198 chars (100%) | 10/10 |
| **CoMeT** | **1,397 chars (27%)** | **9/10** |
| Naive Summary | 1,179 chars (23%) | 1/10 |

- CoMeT uses **27% of the tokens** while retaining **90% accuracy**
- 6/10 questions required **link traversal** (agent read 2-3 nodes)
- Cross-topic questions: CoMeT 5/5 vs Naive 0/5

## Quick Start

### Session Memory (within a conversation)

```python
from comet import CoMeT, scope

@scope
def main(config):
    memo = CoMeT(config)

    # Add conversation turns
    memo.add("B200 4대로 월드모델 학습 가능할까?")
    memo.add("2B면 충분하고 커봐야 8B")
    memo.add("DPO 데이터는 negative를 syntax error로 구성했어")

    # Force compact remaining buffer
    memo.force_compact()

    # Navigation
    for node in memo.list_memories():
        print(memo.read_memory(node['node_id'], depth=0))

    # Agent tools (LangChain compatible)
    tools = memo.get_tools()
    # → get_memory_index, read_memory_node, search_memory

main()
```

### Cross-Session RAG Retrieval

```python
from comet import CoMeT, scope

@scope
def main(config):
    config.retrieval.vector_db_path = './memory_store/vectors'

    memo = CoMeT(config)

    # Ingest turns (auto-indexed to VectorIndex on compaction)
    memo.add("JWT 액세스 토큰 만료는 15분, 리프레시는 7일로 설정")
    memo.force_compact()

    # Semantic retrieval across all sessions
    results = memo.retrieve("토큰 만료 설정이 어떻게 되어있어?")
    for r in results:
        print(f"[{r.node.node_id}] score={r.relevance_score:.4f}")
        print(f"  {r.node.summary}")

main()
```

### 3-Tier Progressive Retrieval

```python
# Tier 1: Short summary scan
results = memo.retrieve("LangGraph architecture")
# → [mem_xxx] (score=0.85) LangGraph 프레임워크 아키텍처 요약

# Tier 2: Lazy detailed summary (generated on first call, cached after)
detailed = memo.get_detailed_summary("mem_xxx")
# → "LangGraph provides graph-based orchestration with checkpointing..."

# Tier 3: Full raw content (only when needed)
raw = memo.get_raw_content("mem_xxx")
# → [complete original text]
```

### Document Ingestion

```python
# Ingest large documents (auto-chunked)
nodes = memo.add_document(
    content=web_search_result,
    source='https://example.com/article'
)
```

## GCRI Integration

CoMeT serves as the in-session memory layer for [GCRI](https://github.com/Dirac-Robot/GCRI) (Graph-based Collective Reasoning Intelligence), a multi-agent reasoning framework.

### 3-Tier Tool Pipeline

GCRI agents access CoMeT through three progressively deeper tools:

| Tool | Tier | Description |
|------|------|-------------|
| `retrieve_from_memory(query)` | 1 | Search → short summaries + node IDs |
| `read_detailed_summary(node_id)` | 2 | Lazy-generated detailed summary (cached) |
| `read_raw_memory(node_id)` | 3 | Full original content from vector store |

### Auto-Ingest

Long tool outputs (> 1500 chars) are automatically ingested into CoMeT. A rolling window ensures agents can immediately see recent results:

- **First 2 outputs**: Returned raw in full, silently stored in CoMeT
- **3rd output onward**: Replaced with node_id reference — agents use `read_detailed_summary` or `read_raw_memory` to access

### Memory Agent Context

GCRI's Memory Agent receives CoMeT's context window in its prompts, enabling it to leverage in-session knowledge when extracting active constraints and updating external memory on successful task completion.

## Configuration ([ato](https://github.com/Dirac-Robot/ato))

```python
# comet/config.py
@scope.observe(default=True)
def default(config):
    config.slm_model = 'gpt-4o-mini'
    config.main_model = 'gpt-4o'
    config.compacting.load_threshold = 3
    config.compacting.max_l1_buffer = 5

    # RAG retrieval (enabled when retrieval block exists)
    config.retrieval.embedding_model = 'text-embedding-3-small'
    config.retrieval.vector_backend = 'chroma'
    config.retrieval.vector_db_path = './memory_store/vectors'
    config.retrieval.top_k = 5

@scope.observe()
def local_slm(config):
    config.slm_model = 'ollama/gemma3:4b'

@scope.observe()
def aggressive(config):
    config.compacting.load_threshold = 2
    config.compacting.max_l1_buffer = 3
```

```bash
# Use default
python main.py

# Local SLM + aggressive compacting
python main.py local_slm aggressive
```

## Project Structure

```
comet/
├── orchestrator.py    # CoMeT main class (3-tier retrieval, document ingestion)
├── sensor.py          # L1 extraction + cognitive load (SLM)
├── compacter.py       # L1→L2 structuring + auto-linking (LLM)
├── storage.py         # JSON key-value store + navigation
├── schemas.py         # MemoryNode, L1Memory, CognitiveLoad, RetrievalResult
├── config.py          # ato scope configuration
├── vector_index.py    # ChromaDB dual-collection vector store (full raw storage)
├── retriever.py       # QueryAnalyzer + ScoreFusion + Retriever
├── consolidator.py    # Dedup + cross-link + tag normalization
└── templates/
    ├── cognitive_load.txt   # Cognitive load judgment prompt
    ├── compacting.txt       # Memory structuring prompt
    ├── l1_extraction.txt    # Fast-layer entity/intent extraction
    └── query_analysis.txt   # Query decomposition prompt
```
