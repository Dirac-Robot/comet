import uuid
from typing import Optional

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from como.schemas import CoMoState, MemoryChunk, ScoredChunks


class CoMo:
    """
    CoMo - Contextual Memory

    A dual-scored memory system that explicitly models:
    - Retention Score: How important is this information?
    - Exclusion Score: Should this be excluded? (trivial OR must-exclude)
    
    Both scores are independent (don't sum to 1).
    High exclusion overrides high retention.
    
    Key insight: Scores are QUERY-CONDITIONAL.
    The same chunk may have different exclusion scores for different queries.
    """

    def __init__(self, config: ADict):
        self._config = config
        self._llm: BaseChatModel = None
        self._graph = None
        self._build()

    def _build(self):
        self._llm = ChatOpenAI(model=self._config.model)

        graph = StateGraph(CoMoState)
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
        
        if query:
            prompt = f"""You are a document filter. For each text chunk, decide:

### is_relevant
- true: Related to the query topic
- false: Completely unrelated

### should_exclude
- false (KEEP): Chunk doesn't have to be excluded.
- true (EXCLUDE): Chunk has to be excluded.

**QUERY: "{query}"**

## EXAMPLES
Query: "Who developed Linux?"
- "Linux was developed by Linus Torvalds in 1991." → is_relevant=true, should_exclude=false (KEEP: has the answer "Linus Torvalds")
- "Linux is open-source and free to use." → is_relevant=true, should_exclude=true (EXCLUDE: about Linux, but no answer to "Who")
- "Python is a programming language." → is_relevant=false, should_exclude=true (EXCLUDE: unrelated)

## TEXT TO ANALYZE:
{state['input_text']}

Break into meaningful chunks and classify each."""
        else:
            prompt = f"""Analyze the following text and break it into meaningful memory chunks.
For each chunk, assign:

- is_relevant: true for most chunks (no specific query to filter by)
- should_exclude: true only for sensitive or clearly temporary info

Assign unique IDs to each chunk.

Text to analyze:
{state['input_text']}"""

        result: ScoredChunks = structured_llm.invoke(prompt)

        for chunk in result.chunks:
            if not chunk.id:
                chunk.id = str(uuid.uuid4())[:8]

        return {'chunks': result.chunks}

    def apply_threshold(self, state: ERASEState) -> dict:
        """Filter chunks based on binary is_relevant and should_exclude flags."""
        committed = []
        for chunk in state['chunks']:
            # Keep if relevant AND not excluded
            if chunk.is_relevant and not chunk.should_exclude:
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

