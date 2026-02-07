from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class MemoryChunk(BaseModel):
    """A scored memory unit with binary retention and erasure flags."""
    id: str = Field(description='Unique identifier for this memory chunk')
    content: str = Field(description='The actual content of this memory')
    is_relevant: bool = Field(
        description='True if this chunk is relevant to the query. False if completely unrelated.'
    )
    should_exclude: bool = Field(
        description='True if this should be EXCLUDED (related but does not answer query). False to KEEP.'
    )
    category: Optional[str] = Field(default=None, description='Optional category label')


class ScoredChunks(BaseModel):
    """LLM output schema for scoring text chunks."""
    chunks: list[MemoryChunk] = Field(description='List of scored memory chunks')
    reasoning: Optional[str] = Field(
        default=None,
        description='Brief explanation of scoring rationale'
    )


class CoMoState(TypedDict):
    """Main state for CoMo workflow."""
    input_text: str
    query: Optional[str]
    chunks: list[MemoryChunk]
    committed_memory: list[MemoryChunk]
    iteration: int


class Message(BaseModel):
    """A conversation message with optional ERASE flags."""
    role: Literal['user', 'assistant', 'system'] = Field(description='Message role')
    content: str = Field(description='Message content')
    timestamp: datetime = Field(default_factory=datetime.now)
    is_relevant: Optional[bool] = Field(default=None)
    should_exclude: Optional[bool] = Field(default=None)
    
    def to_str(self) -> str:
        return f"[{self.role}] {self.content}"
