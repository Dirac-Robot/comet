"""CoMeT Core Schemas: Memory Node, Cognitive Load, State definitions."""
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class MemoryNode(BaseModel):
    """계층적 메모리 노드 - Summary + Trigger 하이브리드."""
    node_id: str = Field(description='Unique node identifier')
    depth_level: int = Field(default=0)
    topic_tags: list[str] = Field(default_factory=list)
    summary: str = Field(default='', description='Brief topic description')
    trigger: str = Field(default='', description='When to retrieve this info')
    content_key: str = Field(description='Pointer key to raw data')
    raw_location: str = Field(description='Path to raw data file')
    links: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    def get_raw_path(self) -> str:
        return self.raw_location


class CognitiveLoad(BaseModel):
    """SLM 인지 부하 판단 결과."""
    logic_flow: Literal['MAINTAIN', 'BROKEN'] = Field(
        description='MAINTAIN: 맥락 유지, BROKEN: 주제 전환 감지'
    )
    load_level: int = Field(
        ge=1, le=5,
        description='1=Low, 5=High - 정보 밀도/복잡도'
    )
    reasoning: Optional[str] = Field(
        default=None,
        description='Brief explanation of the judgment'
    )

    @property
    def needs_compacting(self) -> bool:
        """Compacting 트리거 여부."""
        return self.logic_flow == 'BROKEN' or self.load_level >= 4


class L1Memory(BaseModel):
    """Fast Layer 메모리 항목."""
    content: str = Field(description='Extracted content from turn')
    entities: list[str] = Field(default_factory=list, description='Extracted entities')
    intent: Optional[str] = Field(default=None, description='User intent if detected')
    timestamp: datetime = Field(default_factory=datetime.now)


class CoMeTState(BaseModel):
    """LangGraph State for CoMeT workflow."""
    current_input: str = Field(default='', description='Current turn input')
    l1_buffer: list[L1Memory] = Field(default_factory=list, description='Fast layer buffer')
    cognitive_load: Optional[CognitiveLoad] = Field(default=None)
    pending_nodes: list[MemoryNode] = Field(default_factory=list, description='Nodes to be stored')
    iteration: int = Field(default=0)
