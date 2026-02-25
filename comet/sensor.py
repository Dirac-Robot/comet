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
        # langchain-openai>=0.3.0 defaults to json_schema which allows extra fields (e.g. reasoning).
        # function_calling enforces exact schema via tool calling.
        self._structured_llm = self._llm.with_structured_output(CognitiveLoad, method='function_calling')
        self._l1_extractor = self._llm.with_structured_output(L1Extraction, method='function_calling')

    def extract_l1(self, content: str) -> L1Memory:
        """Extract L1 memory from a single turn via structured output."""
        prompt = load_template('l1_extraction').format(content=content)
        result: L1Extraction = self._l1_extractor.invoke(prompt)

        if result is None:
            return L1Memory(
                content=content[:200],
                raw_content=content,
                entities=[],
                intent=None,
            )

        return L1Memory(
            content=result.core_content,
            raw_content=content,
            entities=result.entities,
            intent=result.intent,
        )

    def assess_load(
        self,
        current_input: str,
        l1_buffer: list[L1Memory],
        session_summaries: list[str] | None = None,
    ) -> CognitiveLoad:
        """Assess cognitive load based on L1 buffer and current input."""
        l1_summaries = '\n'.join([
            f"- {mem.content}" for mem in l1_buffer[-5:]
        ]) if l1_buffer else "(No previous context)"

        if session_summaries:
            session_summaries_text = '\n'.join(
                f"- {s}" for s in session_summaries
            )
        else:
            session_summaries_text = '(No session memory yet)'

        prompt = load_template('cognitive_load').format(
            l1_summaries=l1_summaries,
            current_input=current_input,
            session_summaries=session_summaries_text,
        )

        result: CognitiveLoad = self._structured_llm.invoke(prompt)
        return result

    def get_compaction_reason(
        self,
        load: CognitiveLoad,
        buffer_size: int,
    ) -> Optional[str]:
        """Determine compaction reason based on cognitive load and buffer state.

        Returns reason string or None if no compaction needed.
        """
        max_buffer = self._config.compacting.max_l1_buffer
        min_buffer = self._config.compacting.get('min_l1_buffer', 3)
        load_threshold = self._config.compacting.load_threshold

        if buffer_size < min_buffer:
            return None

        if load.logic_flow == 'BROKEN':
            return 'topic_shift'
        if load.load_level >= load_threshold:
            return 'high_load'
        if buffer_size >= max_buffer:
            return 'buffer_overflow'
        return None

    def should_compact(
        self,
        load: CognitiveLoad,
        buffer_size: int,
    ) -> bool:
        """Determine if compacting should be triggered (backward compat)."""
        return self.get_compaction_reason(load, buffer_size) is not None
