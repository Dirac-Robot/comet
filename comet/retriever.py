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
    """Reciprocal Rank Fusion (RRF) for merging dual-path search results."""

    def __init__(self, config: ADict):
        self._alpha = config.retrieval.fusion_alpha

    def fuse(
        self,
        summary_results: list[ScoredResult],
        trigger_results: list[ScoredResult],
    ) -> list[ScoredResult]:
        k = 60

        rrf_scores: dict[str, float] = {}

        for result in summary_results:
            rrf_scores[result.node_id] = rrf_scores.get(result.node_id, 0.0)
            rrf_scores[result.node_id] += self._alpha * (1.0/(k+result.rank+1))

        for result in trigger_results:
            rrf_scores[result.node_id] = rrf_scores.get(result.node_id, 0.0)
            rrf_scores[result.node_id] += (1.0-self._alpha) * (1.0/(k+result.rank+1))

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

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

        search_k = min(top_k*3, self._vector_index.count)
        summary_hits = self._vector_index.search_by_summary(analyzed.semantic_query, search_k)
        trigger_hits = self._vector_index.search_by_trigger(analyzed.search_intent, search_k)

        fused = self._fusion.fuse(summary_hits, trigger_hits)
        top_results = fused[:top_k]

        retrieval_results = []
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

        logger.info(f'Retrieved {len(retrieval_results)} nodes for query: "{query[:50]}..."')
        return retrieval_results

    def rebuild_index(self):
        all_entries = self._store.list_all()
        nodes = []
        for entry in all_entries:
            node = self._store.get_node(entry['node_id'])
            if node:
                nodes.append(node)

        if not nodes:
            logger.warning('No nodes to index')
            return

        self._vector_index.reset()
        self._vector_index.upsert_batch(nodes)
        logger.info(f'Rebuilt VectorIndex with {len(nodes)} nodes')
