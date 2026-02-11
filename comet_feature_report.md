# CoMeT (Cognitive Memory Technology) Feature Report

## 1) Design Philosophy

CoMeT is designed drawing on cognitive science principles emphasizing a dual-speed memory system that mimics human cognitive memory layering and dynamic resolution. It specifically models memory as an active, hierarchical system with a fast (L1) memory layer processing raw input into distilled representations and a slower (L2+) layer that compacts these representations into structured memory nodes embedding "what" (content summary) and "when" (trigger and recall context) information.

This layered approach differentiates it fundamentally from conventional Retrieval-Augmented Generation (RAG) or typical flat vector-store systems, which tend to embed and index raw or chunked text without explicit modeling of episodic triggers or cognitive load dynamics.

- The fast layer, implemented by the `CognitiveSensor` in `comet/sensor.py`, uses a small language model (SLM) to extract distilled content, key entities, and user intent per conversational turn. It also assesses cognitive load and flow logic to detect topic shifts or overload.
- The slow layer, managed by the `MemoryCompacter` (`comet/compacter.py`), uses a larger LLM to compact buffered L1 memories into semantic summaries with triggers, recall modes (passive/active/both), and topic tags. These become structured `MemoryNode` instances.

The architecture integrates auto-linking of related nodes by topic to navigate the memory graph and uses a triple-path embedding index (summary, trigger, raw) for retrieval, enabling dual-path query analysis and recall (comet/retriever.py, comet/vector_index.py).

In sum, CoMeT is called "Cognitive Memory Technology" because it operationalizes cognitive memory principles (dual-speed processing, cognitive load management, structured episodic recall) into a memory system for AI agents, diverging from purely data-driven or flat retrieval architectures.

---

## 2) Memory Lifecycle

1. **Raw Input Ingestion:** The `CoMeT` orchestrator (`comet/orchestrator.py`) accepts raw text or message turns.

2. **L1 Extraction:** The `CognitiveSensor` extracts distilled `L1Memory` entries per turn, capturing concise core content, entities, and intent. It also assesses cognitive load by examining recent L1 contents and current input to detect topic continuity or disruption.

3. **Buffering:** Extracted L1 entries accumulate in the orchestrator's `_l1_buffer`.

4. **Compaction Trigger:** Based on cognitive load signals (logic flow "BROKEN" or high load level) or buffer size exceeding configured max, compaction is triggered.

5. **Compaction:** The `MemoryCompacter` concatenates buffered L1 contents as raw data and prompts its LLM to produce a structured summary, a trigger phrase defining when to recall this memory, recall mode (passive/active/both), and topic tags. This output is wrapped in a `MemoryNode`.

6. **Storage:** The `MemoryNode` and raw data are stored by `MemoryStore` (`comet/storage.py`) in a JSON key-value structure with separate raw text files for lazy loading and ease of inspection.

7. **Vector Index Updates:** `VectorIndex` (`comet/vector_index.py`) maintains distinct embedding collections for summaries, triggers, and raw data, allowing multifaceted retrieval.

8. **Consolidation:** Periodically or on demand, `Consolidator` (`comet/consolidator.py`) deduplicates similar nodes, cross-links related but non-duplicate nodes, and normalizes topic tags for consistent taxonomy.

Skipping compaction would result in storing raw, potentially redundant and unstructured data chunks leading to poor memory navigation, inflated storage, and inefficient, imprecise retrieval.

---

## 3) Retrieval Architecture

CoMeT implements a layered, dual-path retrieval:

- **Query Decomposition:** The `QueryAnalyzer` in `comet/retriever.py` uses an SLM to split user queries into a semantic component ("what" the user wants) and a situational intent or trigger component ("when" or "why" it might be needed).

- **VectorIndex Collections:** Separate Chromadb collections embed the summaries, triggers, and raw data of memory nodes.

- **Dual-path Search:** Searches are independently run on summary and trigger collections using semantic and situational queries respectively.

- **Raw Content Fallback:** Searches over raw text embeddings provide additional fallback retrieval, weighted moderately lower.

- **Score Fusion:** Results from all paths are merged by Reciprocal Rank Fusion (`ScoreFusion` in retriever.py) balancing rank and embedding similarity for precise relevance ordering.

- **Linked Node Expansion:** Retrieved nodes' links are traversed to expand result context, supporting navigation across related topics.

This layered retrieval surpasses naive flat vector search by capturing both factual content and episodic recall signals, leading to higher precision and dynamic recall appropriately tuned for AI agent contexts.

---

## 4) Storage & Persistence

CoMeT uses `MemoryStore` (`comet/storage.py`) as a file-based JSON key-value store rather than a traditional database, for several reasons:

- **Transparency and Portability:** JSON and raw text files are human-readable and easily inspectable.
- **Separation of Raw and Metadata:** Enables lazy loading of raw text, avoiding loading large blobs unnecessarily.
- **Simplicity:** Avoids database dependencies and complexities.

Tradeoffs include potentially reduced performance and concurrency control for massive datasets.

The schemas defined in `comet/schemas.py` using Pydantic support strong typing, validation, and version evolution.

---

## 5) Sensor & Change Detection

`CognitiveSensor` is the fast L1 memory extractor and cognitive load assessor. It extracts distilled content from each conversational turn, identifying entities and user intents. It also computes cognitive load indicating whether current input maintains or breaks logic flow and estimates load level.

Change detection here is essential for dynamically triggering compaction: recognizing topic shifts, high cognitive load, or buffer fullness permits the system to maintain memory usability, pruning raw L1 buffers into structured, navigable memory nodes.

---

## 6) Orchestrator Design

The `CoMeT` class in `comet/orchestrator.py` orchestrates the entire memory lifecycle:

- **Input Handling:** via `add` method which normalizes inputs and passes them to the sensor for L1 extraction and load assessment.
- **Buffer Management:** Accumulates L1 extractions in a buffer, triggers compaction per configured conditions.
- **Compaction and Storage:** Calls the `MemoryCompacter` to create `MemoryNode`s, saves them via `MemoryStore` and updates the vector index.
- **Session Management:** Tracks nodes within a session for consolidation.
- **Navigation and Retrieval Interfaces:** Supports reading memory at various depths, searching by tags, and multilayer retrieval via `Retriever`.

The orchestrator supports two paths:
- **Full pipeline:** Includes sensor processing, compaction, storing, indexing, and retrieval.
- **Direct storage path:** Implied path for injecting raw data directly, bypassing sensor extraction for faster but less structured input.

This allows balancing between structured fidelity and ingestion speed depending on application needs.

---

## 7) Benchmarking Strategy

The benchmark suite (referenced in the design docs and `benchmark_real.py`) evaluates CoMeT on 119-turn conversations with 20 questions testing both single-topic and cross-topic retrieval.

Capabilities measured include:
- Retrieval accuracy compared to full-context baseline
- Efficiency measured by token usage and latency
- Effectiveness of link traversal for complex queries

Findings demonstrate CoMeT achieves about 27% of full context token usage while preserving about 90% of retrieval accuracy. It outperforms naive summarization significantly, especially in cross-topic Q&A accuracy. The benchmark also compares session-memory and RAG modes of CoMeT for different cost-quality tradeoffs.

---

## 8) Configuration & Extensibility

CoMeT configures via `ato` scope system (`comet/config.py`):

- Models for fast L1 SLM (sensor & query analyzer) and slow main LLM (compacter) are configurable.
- Thresholds for cognitive load and buffer size trigger compaction tuning.
- Storage paths and backend type (default JSON) are configurable.
- Retrieval parameters like embedding model, vector backend, fusion weights, and top-k results are adjustable.

Extension points include:
- Replacing or customizing the sensor or main LLM models.
- Implementing alternative storage backends by conforming to the `MemoryStore` interface.
- Modifying retrieval strategies by adjusting or replacing the retriever, query analyzer, or vector index parameters.

This design supports easy adaptation to various use cases and deployment environments.

---

# End of Report

