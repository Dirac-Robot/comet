"""Single-scored memory baseline with its own prompt (fair comparison)."""
import uuid
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from ato.adict import ADict

from como.schemas import MemoryChunk


class SingleScoredChunks(BaseModel):
    """LLM output schema for single-scored chunking."""
    chunks: list[MemoryChunk] = Field(description='List of scored memory chunks')


class SingleScoredMemory:
    """
    Traditional single-scored memory baseline.
    
    Uses a separate, simpler prompt that only asks for relevance/importance score.
    This is a fair baseline for comparing against ERASE's dual-scored approach.
    """
    
    def __init__(self, config: ADict):
        self._config = config
        self._llm = ChatOpenAI(model=config.model)
        self._threshold = config.threshold.retention
    
    def score_and_retrieve(self, text: str, query: str) -> list[MemoryChunk]:
        """Score chunks using single-score prompt and filter by threshold."""
        structured_llm = self._llm.with_structured_output(SingleScoredChunks)
        
        prompt = f"""You are a retrieval scoring system. Analyze the text and score each chunk for relevance.

## TASK
Break the text into chunks and assign a single score:

### relevance_score (0.0-1.0): How relevant is this to the query?
Think: "Does this chunk help answer the query?"
- 1.0: Directly answers the query
- 0.7: Related and useful context
- 0.4: Somewhat related
- 0.0: Completely irrelevant

**QUERY: "{query}"**

## IMPORTANT
Only assign ONE score per chunk (relevance_score maps to retention_score).
Set erasure_score to 0.0 for all chunks (not used in this mode).

## TEXT TO ANALYZE:
{text}

Break into meaningful chunks and score each. Use unique IDs like chunk1, chunk2, etc."""

        result: SingleScoredChunks = structured_llm.invoke(prompt)
        
        for chunk in result.chunks:
            if not chunk.id:
                chunk.id = str(uuid.uuid4())[:8]
        
        # Filter by threshold
        return [c for c in result.chunks if c.retention_score >= self._threshold]
    
    def __call__(self, text: str, query: str) -> list[MemoryChunk]:
        return self.score_and_retrieve(text, query)
