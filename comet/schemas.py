"""CoMeT Core Schemas: Memory Node, Cognitive Load, State definitions."""
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class MemoryNode(BaseModel):
    """Hierarchical memory node with Summary + Trigger hybrid structure."""
    node_id: str = Field(description='Unique node identifier')
    session_id: Optional[str] = Field(default=None, description='Session that created this node')
    depth_level: int = Field(default=0)
    recall_mode: Literal['passive', 'active', 'both'] = Field(
        default='active',
        description='passive=always in context, active=retrieve on demand, both=always in context + searchable',
    )
    topic_tags: list[str] = Field(default_factory=list)
    summary: str = Field(default='', description='Brief topic description')
    detailed_summary: Optional[str] = Field(
        default=None,
        description='Lazy-generated detailed summary (populated on first request)',
    )
    trigger: str = Field(default='', description='When to retrieve this info')
    content_key: str = Field(description='Pointer key to raw data')
    raw_location: str = Field(description='Path to raw data file')
    links: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    compaction_reason: Optional[str] = Field(
        default=None,
        description='Why compaction triggered: topic_shift | high_load | buffer_overflow | forced | external',
    )

    def get_raw_path(self) -> str:
        return self.raw_location


class CognitiveLoad(BaseModel):
    """SLM cognitive load assessment result."""
    logic_flow: Literal['MAINTAIN', 'BROKEN'] = Field(
        description='MAINTAIN: context continues, BROKEN: topic shift detected'
    )
    load_level: int = Field(
        ge=1, le=5,
        description='1=Low, 5=High - 정보 밀도/복잡도'
    )
    redundancy_detected: bool = Field(
        default=False,
        description='True if session memory summaries contain excessive redundancy',
    )

    @property
    def needs_compacting(self) -> bool:
        """Whether compacting should be triggered."""
        return self.logic_flow == 'BROKEN' or self.load_level >= 4


class L1Memory(BaseModel):
    """Fast Layer memory entry."""
    content: str = Field(description='Extracted content from turn')
    raw_content: str = Field(default='', description='Original unprocessed turn text')
    entities: list[str] = Field(default_factory=list, description='Extracted entities')
    intent: Optional[str] = Field(default=None, description='User intent if detected')
    timestamp: datetime = Field(default_factory=datetime.now)


class RetrievalResult(BaseModel):
    node: MemoryNode
    relevance_score: float = Field(description='Fused RRF relevance score')
    rank: int = Field(default=0)
