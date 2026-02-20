"""MemoryCompacter: L1 -> L2+ structuring with summary + key generation."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from comet.llm_factory import create_chat_model
from comet.schemas import L1Memory, MemoryNode
from comet.storage import MemoryStore
from comet.templates import load_template

if TYPE_CHECKING:
    from comet.vector_index import VectorIndex


class CompactedResult(BaseModel):
    """Structured output for compacting."""
    summary: str = Field(description='Brief topic description (1 line, NO specific numbers/dates)')
    trigger: str = Field(description='Detailed description of when to retrieve this info')
    recall_mode: str = Field(
        default='active',
        description='passive=always in context, active=on-demand, both=always + searchable',
    )
    topic_tags: list[str] = Field(description='1-2 topic tags')


# Prompt loaded from templates/compacting.txt


class MemoryCompacter:
    """
    Slow Layer processor for L1 -> L2+ structuring.
    
    Takes accumulated L1 buffer and creates structured MemoryNodes
    with summaries and keys to raw data.
    """

    def __init__(self, config: ADict, store: MemoryStore, vector_index: Optional[VectorIndex] = None):
        self._config = config
        self._store = store
        self._vector_index = vector_index
        self._llm: BaseChatModel = create_chat_model(config.slm_model, config)
        self._structured_llm = self._llm.with_structured_output(CompactedResult)

    def compact(
        self,
        l1_buffer: list[L1Memory],
        depth_level: int = 1,
        session_id: Optional[str] = None,
        template_name: str = 'compacting',
        compaction_reason: Optional[str] = None,
    ) -> MemoryNode:
        """
        Compact L1 buffer into a structured MemoryNode.
        
        1. Concatenate L1 contents as raw data
        2. Generate summary + tags via LLM
        3. Save raw data with content_key
        4. Create and save MemoryNode
        """
        # Build raw data from L1 buffer
        raw_data = '\n\n'.join([
            f"[{mem.timestamp.strftime('%H:%M:%S')}] {mem.raw_content or mem.content}"
            for mem in l1_buffer
        ])
        
        # Generate summary via LLM (with existing topic context)
        turns_text = '\n'.join([f"- {mem.content}" for mem in l1_buffer])
        existing_tags = self._store.get_all_tags()
        existing_tags = {t for t in existing_tags if not t.startswith('ORIGIN:')}
        tags_text = ', '.join(sorted(existing_tags)) if existing_tags else '(없음)'
        prompt = load_template(template_name).format(
            turns=turns_text,
            existing_tags=tags_text,
        )
        
        result: CompactedResult = self._structured_llm.invoke(prompt)
        
        # Generate keys and save raw
        node_id = self._store.generate_node_id()
        content_key = self._store.generate_content_key(prefix='raw')
        raw_location = self._store.save_raw(content_key, raw_data)
        
        # Create memory node
        node = MemoryNode(
            node_id=node_id,
            session_id=session_id,
            depth_level=depth_level,
            recall_mode=result.recall_mode if result.recall_mode in ('passive', 'active', 'both') else 'active',
            topic_tags=[t for t in result.topic_tags if not t.startswith('ORIGIN:')],
            summary=result.summary,
            trigger=result.trigger,
            content_key=content_key,
            raw_location=raw_location,
            compaction_reason=compaction_reason,
        )
        
        # Save node
        self._store.save_node(node)

        # Auto-link: find existing nodes with overlapping topic tags
        self._auto_link(node)

        if self._vector_index:
            self._vector_index.upsert(node, raw_content=raw_data)

        return node

    def _auto_link(self, new_node: MemoryNode):
        """Link new node to existing nodes with overlapping topic tags."""
        if not new_node.topic_tags:
            return

        new_tags = {t.lower() for t in new_node.topic_tags}
        for existing in self._store.list_all():
            existing_id = existing['node_id']
            if existing_id == new_node.node_id:
                continue
            existing_tags = {t.lower() for t in existing.get('topic_tags', [])}
            if new_tags & existing_tags:
                self.link_nodes(new_node.node_id, existing_id)
                self.link_nodes(existing_id, new_node.node_id)

    def link_nodes(self, source_id: str, target_id: str):
        """Link two related memory nodes."""
        source = self._store.get_node(source_id)
        if source and target_id not in source.links:
            source.links.append(target_id)
            self._store.save_node(source)
