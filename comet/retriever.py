"""Retriever: summary/raw vector search + ScoreFusion + graph expansion.

The former dual-path (summary WHAT / trigger WHEN) design and its SLM
QueryAnalyzer were removed after a consumption-side ablation measured zero
contribution from the trigger channel: quality, economy, and raw-escalation
behavior were identical with the channel silenced. Retrieval is now a single
semantic pass over summaries with a raw-content fallback channel, which also
drops one SLM round-trip per retrieval.
"""
from typing import Optional

from ato.adict import ADict
from loguru import logger

from comet.schemas import RetrievalResult
from comet.storage import MemoryStore
from comet.vector_index import VectorIndex, ScoredResult


class ScoreFusion:
    """Reciprocal Rank Fusion (RRF) for merging summary/raw search results."""

    def __init__(self, config: ADict):
        self._k = config.retrieval.get('rrf_k', 5)
        self._raw_weight = config.retrieval.get('raw_search_weight', 0.2)

    def fuse(
        self,
        summary_results: list[ScoredResult],
        raw_results: Optional[list[ScoredResult]] = None,
    ) -> list[ScoredResult]:
        k = self._k
        if raw_results:
            w_summary = 1.0 - self._raw_weight
            w_raw = self._raw_weight
        else:
            w_summary = 1.0
            w_raw = 0.0

        rrf_scores: dict[str, float] = {}
        sim_scores: dict[str, float] = {}

        for result in summary_results:
            rrf_scores[result.node_id] = rrf_scores.get(result.node_id, 0.0)
            rrf_scores[result.node_id] += w_summary*(1.0/(k+result.rank+1))
            sim = max(0.0, 1.0-result.score)
            sim_scores[result.node_id] = max(sim_scores.get(result.node_id, 0.0), sim)

        if raw_results:
            for result in raw_results:
                rrf_scores[result.node_id] = rrf_scores.get(result.node_id, 0.0)
                rrf_scores[result.node_id] += w_raw*(1.0/(k+result.rank+1))
                sim = max(0.0, 1.0-result.score)
                sim_scores[result.node_id] = max(sim_scores.get(result.node_id, 0.0), sim)

        combined: dict[str, float] = {}
        for node_id in rrf_scores:
            combined[node_id] = rrf_scores[node_id]*0.6 + sim_scores.get(node_id, 0.0)*0.4

        fused = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        return [
            ScoredResult(node_id=node_id, score=score, rank=rank)
            for rank, (node_id, score) in enumerate(fused)
        ]


class Retriever:
    """Unified retrieval interface combining VectorIndex and ScoreFusion."""

    def __init__(self, config: ADict, store: MemoryStore, vector_index: VectorIndex):
        self._config = config
        self._vector_index = vector_index
        self._fusion = ScoreFusion(config)
        self._store = store

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        if top_k is None:
            top_k = self._config.retrieval.top_k

        if self._vector_index.count == 0:
            logger.warning('VectorIndex is empty, no results to retrieve')
            return []

        search_k = min(top_k*3, self._vector_index.count)
        summary_hits = self._vector_index.search_by_summary(query, search_k)
        raw_hits = self._vector_index.search_by_raw(query, search_k)

        fused = self._fusion.fuse(summary_hits, raw_hits or None)
        top_results = fused[:top_k]

        retrieval_results = []
        seen_ids = set()
        for scored in top_results:
            node = self._store.get_node(scored.node_id)
            if node is None:
                logger.warning(f'Node {scored.node_id} not found in store, skipping')
                continue
            retrieval_results.append(RetrievalResult(
                node=node,
                relevance_score=scored.score,
                rank=scored.rank,
            ))
            seen_ids.add(scored.node_id)

        # Reinforcement signal: record the primary (pre-graph-expansion)
        # matches so the dream reinforced-decay pass can strengthen
        # frequently-recalled nodes and fade unused ones. Graph-expanded
        # neighbours below are associative, not genuine matches, so they are
        # not reinforced. Cheap in-memory counter; never break retrieval on it.
        if seen_ids:
            try:
                self._store.record_recall_hits(list(seen_ids))
            except Exception as e:
                logger.debug(f'record_recall_hits failed (non-fatal): {e}')
            # Persistent recall journal — independent of the lossy in-memory
            # reinforcement buffer above (which only reaches node metadata when
            # a dream pass drains it, and is dropped entirely on restart, so
            # node.recall_count is near-zero and unusable for analysis). One
            # structured line per retrieval makes recall frequency measurable
            # after the fact. Cheap (one log line, off the critical result
            # path); the CoBrA log sink captures ``extra``.
            logger.bind(
                event='memory.recall',
                n_recalled=len(seen_ids),
                node_ids=sorted(seen_ids),
            ).info(f'Memory recall: {len(seen_ids)} node(s)')

        # ── Graph-aware re-ranking ──
        # Count how many top-K results link to each unseen node.
        # Nodes referenced by multiple results are likely important
        # even if they didn't score high in vector search.
        link_refcount: dict[str, float] = {}
        for result in retrieval_results:
            for link_id in result.node.links:
                if link_id not in seen_ids:
                    link_refcount[link_id] = link_refcount.get(link_id, 0.0) + result.relevance_score

        # ── Multi-hop link traversal (2-hop, decaying weight) ──
        hop1_decay = 0.5
        hop2_decay = 0.25

        linked_results = []
        hop1_ids: set[str] = set()

        # Hop 1: direct links from top-K results
        for result in retrieval_results:
            for link_id in result.node.links:
                if link_id in seen_ids:
                    continue
                linked_node = self._store.get_node(link_id)
                if linked_node is None:
                    continue
                seen_ids.add(link_id)
                hop1_ids.add(link_id)
                # Boost score if multiple results reference this node
                base_score = result.relevance_score * hop1_decay
                refcount_bonus = link_refcount.get(link_id, 0.0)
                score = base_score + refcount_bonus * 0.3
                linked_results.append(RetrievalResult(
                    node=linked_node,
                    relevance_score=score,
                    rank=len(retrieval_results) + len(linked_results),
                ))

        # Hop 2: links of hop-1 nodes (cross-session only for relevance)
        for lr in linked_results:
            if lr.node.node_id not in hop1_ids:
                continue
            for link_id in lr.node.links:
                if link_id in seen_ids:
                    continue
                linked_node = self._store.get_node(link_id)
                if linked_node is None:
                    continue
                seen_ids.add(link_id)
                linked_results.append(RetrievalResult(
                    node=linked_node,
                    relevance_score=lr.relevance_score * hop2_decay,
                    rank=len(retrieval_results) + len(linked_results),
                ))

        retrieval_results.extend(linked_results)

        n_linked = len(linked_results)
        n_hop1 = len(hop1_ids)
        logger.info(
            f'Retrieved {len(retrieval_results)} nodes '
            f'({n_hop1} hop-1, {n_linked - n_hop1} hop-2 via links) '
            f'(query="{query[:40]}...")'
        )
        return retrieval_results

    def rebuild_index(self):
        all_entries = self._store.list_all()
        nodes = []
        raw_contents = []
        for entry in all_entries:
            node = self._store.get_node(entry['node_id'])
            if node:
                nodes.append(node)
                raw = self._store.get_raw(node.content_key) or ''
                raw_contents.append(raw)

        if not nodes:
            logger.warning('No nodes to index')
            return

        self._vector_index.reset()
        self._vector_index.upsert_batch(nodes, raw_contents)
        logger.info(f'Rebuilt VectorIndex with {len(nodes)} nodes')
