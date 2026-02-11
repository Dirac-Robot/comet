"""Retriever: QueryAnalyzer + ScoreFusion + unified retrieval interface."""
from typing import Literal, Optional

from ato.adict import ADict
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field

from comet.schemas import MemoryNode, RetrievalResult
from comet.storage import MemoryStore
from comet.templates import load_template
from comet.vector_index import VectorIndex, ScoredResult


class AnalyzedQuery(BaseModel):
    semantic_query: str = Field(
        description='What information is being sought (for summary matching)'
    )
    search_intent: str = Field(
        description='What situation/context triggered this need (for trigger matching)'
    )
    urgency: Literal['low', 'medium', 'high'] = Field(default='medium')


class QueryAnalyzer:
    """SLM-based query decomposition into semantic_query + search_intent."""

    def __init__(self, config: ADict):
        self._llm: BaseChatModel = ChatOpenAI(model=config.slm_model)
        self._structured_llm = self._llm.with_structured_output(AnalyzedQuery)

    def analyze(self, query: str) -> AnalyzedQuery:
        prompt = load_template('query_analysis').format(query=query)
        return self._structured_llm.invoke(prompt)


class ScoreFusion:
    """Reciprocal Rank Fusion (RRF) for merging multi-path search results."""

    def __init__(self, config: ADict):
        self._alpha = config.retrieval.fusion_alpha
        self._k = config.retrieval.get('rrf_k', 5)
        self._raw_weight = config.retrieval.get('raw_search_weight', 0.2)

    def fuse(
        self,
        summary_results: list[ScoredResult],
        trigger_results: list[ScoredResult],
        raw_results: Optional[list[ScoredResult]] = None,
    ) -> list[ScoredResult]:
        k = self._k

        if raw_results:
            scale = 1.0-self._raw_weight
            w_summary = self._alpha*scale
            w_trigger = (1.0-self._alpha)*scale
            w_raw = self._raw_weight
        else:
            w_summary = self._alpha
            w_trigger = 1.0-self._alpha
            w_raw = 0.0

        rrf_scores: dict[str, float] = {}
        sim_scores: dict[str, float] = {}

        for result in summary_results:
            rrf_scores[result.node_id] = rrf_scores.get(result.node_id, 0.0)
            rrf_scores[result.node_id] += w_summary*(1.0/(k+result.rank+1))
            sim = max(0.0, 1.0-result.score)
            sim_scores[result.node_id] = max(sim_scores.get(result.node_id, 0.0), sim)

        for result in trigger_results:
            rrf_scores[result.node_id] = rrf_scores.get(result.node_id, 0.0)
            rrf_scores[result.node_id] += w_trigger*(1.0/(k+result.rank+1))
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
    """Unified retrieval interface combining QueryAnalyzer, VectorIndex, and ScoreFusion."""

    def __init__(self, config: ADict, store: MemoryStore, vector_index: VectorIndex):
        self._config = config
        self._analyzer = QueryAnalyzer(config)
        self._vector_index = vector_index
        self._fusion = ScoreFusion(config)
        self._store = store

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        if top_k is None:
            top_k = self._config.retrieval.top_k

        if self._vector_index.count == 0:
            logger.warning('VectorIndex is empty, no results to retrieve')
            return []

        analyzed = self._analyzer.analyze(query)
        logger.debug(
            f'QueryAnalyzer: semantic="{analyzed.semantic_query}" '
            f'intent="{analyzed.search_intent}" urgency={analyzed.urgency}'
        )

        return self.retrieve_dual(analyzed.semantic_query, analyzed.search_intent, top_k)

    def retrieve_dual(
        self,
        summary_query: str,
        trigger_query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        if top_k is None:
            top_k = self._config.retrieval.top_k

        if self._vector_index.count == 0:
            logger.warning('VectorIndex is empty, no results to retrieve')
            return []

        search_k = min(top_k*3, self._vector_index.count)
        summary_hits = self._vector_index.search_by_summary(summary_query, search_k)
        trigger_hits = self._vector_index.search_by_trigger(trigger_query, search_k)

        raw_hits_s = self._vector_index.search_by_raw(summary_query, search_k)
        raw_hits_t = self._vector_index.search_by_raw(trigger_query, search_k)
        raw_merged: dict[str, ScoredResult] = {}
        for hit in raw_hits_s + raw_hits_t:
            existing = raw_merged.get(hit.node_id)
            if existing is None or hit.score < existing.score:
                raw_merged[hit.node_id] = hit
        raw_hits = sorted(raw_merged.values(), key=lambda x: x.score)
        for rank, hit in enumerate(raw_hits):
            hit.rank = rank

        fused = self._fusion.fuse(summary_hits, trigger_hits, raw_hits or None)
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

        linked_results = []
        for result in retrieval_results:
            for link_id in result.node.links:
                if link_id in seen_ids:
                    continue
                linked_node = self._store.get_node(link_id)
                if linked_node is None:
                    continue
                seen_ids.add(link_id)
                linked_results.append(RetrievalResult(
                    node=linked_node,
                    relevance_score=result.relevance_score*0.5,
                    rank=len(retrieval_results)+len(linked_results),
                ))
        retrieval_results.extend(linked_results)

        n_linked = len(linked_results)
        logger.info(
            f'Retrieved {len(retrieval_results)} nodes ({n_linked} via links) '
            f'(summary="{summary_query[:30]}..." trigger="{trigger_query[:30]}...")'
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
