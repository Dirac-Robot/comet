"""MemoryCompacter: L1 -> L2+ structuring with summary + key generation."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from datetime import datetime

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
    summary: str = Field(description='Factual index of confirmed facts/decisions from the conversation, semicolon-separated if multiple topics')
    trigger: str = Field(description='Retrieval scenario: when would someone need this? Must differ from summary')
    recall_mode: str = Field(
        default='active',
        description='passive=always in context, active=on-demand, both=always + searchable',
    )
    topic_tags: list[str] = Field(description='1-3 topic tags')
    extracted_rules: list[str] = Field(
        default_factory=list,
        description='Personal directives the user DIRECTLY commanded YOU to follow (2nd person). '
                    'Must be cross-session principles, NOT one-time task instructions or domain knowledge. '
                    'Generalize into clear imperatives. Empty if none found (most conversations have zero).',
    )


class ConsolidatedRules(BaseModel):
    """Output of rule consolidation."""
    rules: list[str] = Field(description='Final consolidated list of user rules')


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
        self._llm: BaseChatModel = create_chat_model(config.main_model, config)
        self._structured_llm = self._llm.with_structured_output(CompactedResult)

    def compact(
        self,
        l1_buffer: list[L1Memory],
        depth_level: int = 1,
        session_id: Optional[str] = None,
        template_name: str = 'compacting',
        compaction_reason: Optional[str] = None,
        policy=None,
    ) -> MemoryNode:
        """
        Compact L1 buffer into a structured MemoryNode.
        
        1. Concatenate L1 contents as raw data
        2. Generate summary + tags via LLM
        3. Save raw data with content_key
        4. Create and save MemoryNode
        
        If policy (MemoryGenerationPolicy) is provided, uses base template
        with policy block injection. Otherwise falls back to template_name.
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
        tags_text = ', '.join(sorted(existing_tags)) if existing_tags else '(none)'

        if policy is not None:
            prompt = self._render_policy_prompt(policy, turns_text, tags_text)
        else:
            language = self._config.get('language', 'the same language as the user')
            prompt = load_template(template_name).format(
                turns=turns_text,
                existing_tags=tags_text,
                language=language,
            )
        
        result: CompactedResult = self._structured_llm.invoke(prompt)
        
        # Generate keys and save raw
        node_id = self._store.generate_node_id()
        content_key = self._store.generate_content_key(prefix='raw')
        raw_location = self._store.save_raw(content_key, raw_data)
        
        recall_mode = result.recall_mode if result.recall_mode in ('passive', 'active', 'both') else 'active'
        if result.extracted_rules:
            recall_mode = 'passive'

        # Create memory node
        tags = [t for t in result.topic_tags if not t.startswith('ORIGIN:')]
        if policy is not None:
            for hint in getattr(policy, 'tag_hints', ()):
                if hint not in tags:
                    tags.append(hint)

        node = MemoryNode(
            node_id=node_id,
            session_id=session_id,
            depth_level=depth_level,
            recall_mode=recall_mode,
            topic_tags=tags,
            summary=result.summary,
            trigger=result.trigger,
            content_key=content_key,
            raw_location=raw_location,
            compaction_reason=compaction_reason,
        )
        
        # Save node
        self._store.save_node(node)

        # Save extracted rules (with consolidation)
        if result.extracted_rules:
            self._consolidate_rules(result.extracted_rules, source_node=node_id)

        # Auto-link: find existing nodes with overlapping topic tags
        self._auto_link(node)

        if self._vector_index:
            self._vector_index.upsert(node, raw_content=raw_data)

        return node

    def _render_policy_prompt(self, policy, turns_text: str, tags_text: str) -> str:
        policy_block = policy.render_policy_block()
        modality = getattr(policy, 'modality', 'dialog')
        extract_rules = getattr(policy, 'extract_rules', False)

        if modality == 'dialog':
            summary_instr = (
                'Factual index of confirmed facts/decisions. '
                'Include brief action context (user request, debugging, implementation, etc.). '
                'Semicolon-separated if multiple topics.'
            )
            trigger_instr = (
                'Describe WHEN/WHY to open detailed_summary/raw. Start with "When I...". '
                'MUST differ from summary. 2-4 anchors only.'
            )
            recall_instr = (
                'active (default), passive (permanent instructions), '
                'both (critical constraints)'
            )
        elif modality == 'artifact_code':
            summary_instr = (
                'Start with language/type (e.g. "Python module"). '
                'Include file name, module role, key exports.'
            )
            trigger_instr = (
                '"When I need to verify [export1], [export2] implementation '
                'from [file context]"'
            )
            recall_instr = 'Always "active" for code.'
        elif modality == 'artifact_image':
            summary_instr = 'Describe visual content, source, dimensions, format.'
            trigger_instr = '"When I need to reference [image context] for visual verification"'
            recall_instr = 'Always "active".'
        elif modality == 'execution_trace':
            summary_instr = (
                'Describe execution outcome concisely — '
                'tool name, success/failure, key output values.'
            )
            trigger_instr = '"When I need to verify [execution context] results and output values"'
            recall_instr = 'Always "active".'
        else:
            summary_instr = (
                'Describe ACTUAL FACTS contained (1-2 lines). '
                'Include specific names, numbers, conclusions.'
            )
            trigger_instr = (
                '"When I need to verify [anchor1], [anchor2] exact values from [context]". '
                'STRICT: 1 sentence only. Max 2-4 anchor keywords. '
                'Do NOT list every entity from the summary. '
                'trigger != summary; trigger describes WHEN to recall, not WHAT is stored.'
            )
            recall_instr = 'Always "active" for external content.'

        if extract_rules:
            rules_instr = (
                'Personal directives the user DIRECTLY commanded. '
                'Cross-session principles only. When in doubt, return empty list.'
            )
        else:
            rules_instr = 'Return empty list [].'

        extra_tag = ''
        tag_hints = getattr(policy, 'tag_hints', ())
        if tag_hints:
            extra_tag = f'- MUST include: {", ".join(tag_hints)}'

        base_template = load_template('compacting_base')
        return base_template.format(
            turns=turns_text,
            policy_block=policy_block,
            summary_instruction=summary_instr,
            trigger_instruction=trigger_instr,
            recall_instruction=recall_instr,
            existing_tags=tags_text,
            extra_tag_instruction=extra_tag,
            rules_instruction=rules_instr,
        )

    def _consolidate_rules(self, new_rules: list[str], source_node: str = ''):
        existing = self._store.load_rules()
        existing_texts = [r['rule'] for r in existing]
        if not existing_texts:
            for rule in new_rules:
                self._store.save_rule(rule, source_node=source_node)
            return
        try:
            prompt = load_template('rule_consolidation').format(
                existing_rules='\n'.join(f'- {r}' for r in existing_texts),
                new_rules='\n'.join(f'- {r}' for r in new_rules),
            )
            consolidated_llm = self._llm.with_structured_output(ConsolidatedRules)
            result: ConsolidatedRules = consolidated_llm.invoke(prompt)
            from comet.storage import _atomic_write_json
            rules_path = self._store._rules_path()
            source_map = {r['rule'].strip().lower(): r.get('source_node', '') for r in existing}
            consolidated = []
            for r in result.rules:
                text = r.strip()
                if not text:
                    continue
                src = source_map.get(text.lower(), source_node)
                consolidated.append({
                    'rule': text,
                    'source_node': src,
                    'created_at': datetime.now().isoformat(),
                })
            _atomic_write_json(rules_path, consolidated)
        except Exception:
            for rule in new_rules:
                self._store.save_rule(rule, source_node=source_node)

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
