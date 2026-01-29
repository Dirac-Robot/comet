import uuid
from typing import Optional

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from erase.schemas import ERASEState, MemoryChunk, ScoredChunks


class ERASE:
    """
    Explicit Retention And Selective Erasure

    A dual-scored memory system that explicitly models:
    - Retention Score: How important is this information?
    - Erasure Score: Should this be excluded? (trivial OR must-exclude)
    
    Both scores are independent (don't sum to 1).
    High erasure overrides high retention.
    
    Key insight: Scores are QUERY-CONDITIONAL.
    The same chunk may have different erasure scores for different queries.
    """

    def __init__(self, config: ADict):
        self._config = config
        self._llm: BaseChatModel = None
        self._graph = None
        self._build()

    def _build(self):
        self._llm = ChatOpenAI(model=self._config.model)

        graph = StateGraph(ERASEState)
        graph.add_node('chunk_and_score', self.chunk_and_score)
        graph.add_node('apply_threshold', self.apply_threshold)

        graph.add_edge(START, 'chunk_and_score')
        graph.add_edge('chunk_and_score', 'apply_threshold')
        graph.add_edge('apply_threshold', END)

        self._graph = graph.compile()

    @property
    def config(self) -> ADict:
        return self._config

    @property
    def graph(self):
        return self._graph

    def chunk_and_score(self, state: ERASEState) -> dict:
        """Chunk input text and score each chunk for retention and erasure."""
        structured_llm = self._llm.with_structured_output(ScoredChunks)

        query = state.get('query')
        query_context = ""
        if query:
            query_context = f"""
CURRENT QUERY/CONTEXT: "{query}"

Score erasure_score based on whether each chunk should be EXCLUDED when answering this specific query.
Even important information should have high erasure if it's irrelevant or potentially confusing for THIS query.
"""

        prompt = f"""Analyze the following text and break it into meaningful memory chunks.
For each chunk, assign two INDEPENDENT scores (they don't need to sum to 1):

- retention_score (0.0-1.0): How important is this information IN GENERAL?
  - 1.0 = critical fact (names, dates, decisions, key numbers)
  - 0.0 = trivial detail

- erasure_score (0.0-1.0): Should this be EXCLUDED from the current context?
  - Use this for info that IS important but should NOT be retrieved NOW
  - Examples: off-topic facts, outdated info, sensitive data, context-inappropriate
  - 1.0 = must exclude from current context
  - 0.0 = safe to include
{query_context}
Key insight: High retention + High erasure = "Important, but not now/here"

Assign unique IDs to each chunk.

Text to analyze:
{state['input_text']}"""

        result: ScoredChunks = structured_llm.invoke(prompt)

        for chunk in result.chunks:
            if not chunk.id:
                chunk.id = str(uuid.uuid4())[:8]

        return {'chunks': result.chunks}

    def apply_threshold(self, state: ERASEState) -> dict:
        """Filter chunks based on retention and erasure thresholds."""
        retention_threshold = self._config.threshold.retention
        erasure_threshold = self._config.threshold.erasure

        committed = []
        for chunk in state['chunks']:
            keep = (
                chunk.retention_score >= retention_threshold and
                chunk.erasure_score < erasure_threshold
            )
            if keep:
                committed.append(chunk)

        return {
            'committed_memory': committed,
            'iteration': state.get('iteration', 0)+1
        }

    def invoke(self, text: str, query: Optional[str] = None) -> list[MemoryChunk]:
        """Process text and return committed memories (filtered by threshold).
        
        Args:
            text: The input text to analyze and chunk
            query: Optional query/context to condition the erasure scoring
        """
        initial_state: ERASEState = {
            'input_text': text,
            'query': query,
            'chunks': [],
            'committed_memory': [],
            'iteration': 0
        }
        result = self._graph.invoke(initial_state)
        return result['committed_memory']

    def score_all(self, text: str, query: Optional[str] = None) -> list[MemoryChunk]:
        """Score all chunks without threshold filtering.
        
        Use this to see scores for ALL chunks, including excluded ones.
        Useful for debugging and analysis.
        
        Args:
            text: The input text to analyze and chunk
            query: Optional query/context to condition the erasure scoring
        """
        initial_state: ERASEState = {
            'input_text': text,
            'query': query,
            'chunks': [],
            'committed_memory': [],
            'iteration': 0
        }
        result = self._graph.invoke(initial_state)
        return result['chunks']  # Return ALL chunks, not just committed

    def __call__(self, text: str, query: Optional[str] = None) -> list[MemoryChunk]:
        return self.invoke(text, query)

