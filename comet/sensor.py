"""CognitiveSensor: SLM-based cognitive load detection for L1 processing."""
from typing import Optional

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from comet.llm_factory import create_chat_model
from comet.schemas import CognitiveLoad, L1Memory
from comet.templates import load_template


class L1Extraction(BaseModel):
    """Structured output for L1 extraction."""
    core_content: str = Field(description='Core content in 1-2 sentences')
    entities: list[str] = Field(default_factory=list, description='Key entities (names, numbers, dates)')
    intent: Optional[str] = Field(default=None, description='User intent: question, request, statement, etc.')


class CognitiveSensor:
    """
    Fast Layer (L1) processor using nano-level SLM API.
    
    Responsibilities:
    - Extract key info from each turn
    - Detect topic shifts (logic_flow)
    - Measure cognitive load
    """

    def __init__(self, config: ADict):
        self._config = config
        self._llm: BaseChatModel = create_chat_model(config.slm_model, config)
        self._structured_llm = self._llm.with_structured_output(CognitiveLoad)
        self._l1_extractor = self._llm.with_structured_output(L1Extraction)

    def extract_l1(self, content: str) -> L1Memory:
        """Extract L1 memory from a single turn via structured output."""
        prompt = load_template('l1_extraction').format(content=content)
        result: L1Extraction = self._l1_extractor.invoke(prompt)

        return L1Memory(
            content=result.core_content,
            entities=result.entities,
            intent=result.intent,
        )

    def assess_load(
        self,
        current_input: str,
        l1_buffer: list[L1Memory],
    ) -> CognitiveLoad:
        """Assess cognitive load based on L1 buffer and current input."""
        l1_summaries = '\n'.join([
            f"- {mem.content}" for mem in l1_buffer[-5:]  # Last 5 items
        ]) if l1_buffer else "(No previous context)"

        prompt = load_template('cognitive_load').format(
            l1_summaries=l1_summaries,
            current_input=current_input,
        )

        result: CognitiveLoad = self._structured_llm.invoke(prompt)
        return result

    def should_compact(
        self,
        load: CognitiveLoad,
        buffer_size: int,
    ) -> bool:
        """Determine if compacting should be triggered."""
        max_buffer = self._config.compacting.max_l1_buffer
        min_buffer = self._config.compacting.get('min_l1_buffer', 3)  # 최소 3턴
        load_threshold = self._config.compacting.load_threshold

        # 버퍼가 최소 크기 미만이면 절대 compact 안 함
        if buffer_size < min_buffer:
            return False

        # Trigger if: logic broken OR high load OR buffer overflow
        return (
            load.logic_flow == 'BROKEN' or
            load.load_level >= load_threshold or
            buffer_size >= max_buffer
        )
