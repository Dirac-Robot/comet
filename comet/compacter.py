"""MemoryCompacter: L1 -> L2+ structuring with summary + key generation."""
from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from comet.llm_factory import create_chat_model
from comet.schemas import L1Memory, MemoryNode
from comet.storage import MemoryStore
from comet.templates import load_template
from loguru import logger


# Recognized turn-role prefixes (from _normalize_content + caller-supplied
# strings). Plain langchain types: human/ai/system/tool. Caller conventions
# layered on top: user/assistant/session. Anything else stays a free-form
# bracket tag inside the body — only matches in this set are promoted to a
# role label so the compacter prompt can show "USER vs ASSISTANT" cleanly.
_ROLE_ALIASES = {
    'user': 'USER',
    'human': 'USER',
    'assistant': 'ASSISTANT',
    'ai': 'ASSISTANT',
    'system': 'SYSTEM',
    'session': 'SYSTEM',
    'tool': 'TOOL',
}
_ROLE_PREFIX_RE = re.compile(r'^\[([a-zA-Z][\w-]*)\]\s*', re.DOTALL)


def _split_role(content: str) -> tuple[str | None, str]:
    m = _ROLE_PREFIX_RE.match(content)
    if not m:
        return None, content
    role = _ROLE_ALIASES.get(m.group(1).lower())
    if role is None:
        return None, content
    return role, content[m.end():]


def _format_turns_for_prompt(l1_buffer: list[L1Memory]) -> str:
    """Render L1 buffer as role-labeled chat blocks for the compacter prompt.

    Turns whose content starts with a recognized [role] prefix are emitted
    as ``ROLE:\\n<body>`` blocks; unlabeled turns fall through as bare
    bullet lines so external/non-conversational input still renders.
    """
    blocks: list[str] = []
    for mem in l1_buffer:
        role, body = _split_role(mem.content)
        if role:
            blocks.append(f'{role}:\n{body}')
        else:
            blocks.append(f'- {mem.content}')
    return '\n\n'.join(blocks)

if TYPE_CHECKING:
    from comet.vector_index import VectorIndex


_SESSION_BRIEF_INSTRUCTION = (
    'Rewrite the session brief from scratch — a FULL REWRITE, not an append. '
    'Base it on (a) the prior brief shown in the "Previous Session Brief" '
    'block below (if any), and (b) the new signals in the block below '
    '(compacted turns or brief-relevant nodes, depending on what triggered '
    'this regen). Output the complete new brief; no "unchanged" or "see '
    'above" markers.\n\n'
    'Purpose: the brief rides alongside the hard rules block as the session\'s '
    '*buttered* guidance layer — rule + hint + rationale inlined together, '
    'meant to shape next-turn reasoning directly rather than sit as a '
    'reference list. The hard rules handle the 3 reasoning blind spots '
    '(verification, turn-end, loop-guard); the brief handles everything '
    'else — preferences, work-in-flight context, hints learned from '
    'failures/corrections.\n\n'
    'Fixed skeleton; headers in English, body in the USER\'S language.\n\n'
    '## Active Work Context\n'
    '  - 2-4 bullets. Current goal, what\'s in-flight, binding constraints. '
    'Ephemeral only — omit anything that will be stale within a few turns.\n\n'
    '## Hints\n'
    '  - 0-6 bullets. Each bullet = rule + hint + rationale coexisting in '
    'one sentence via dash-clause: "usually do X — because Y, Z rarely '
    'recovers". Prefer hedged vocabulary (usually / tends to / rarely) '
    'over binary "do not". Sources: user corrections ("don\'t do X"), '
    'stable user preferences, approach patterns that repeatedly failed, '
    'patterns the user explicitly confirmed. One hint per bullet — do not '
    'merge multiple lessons. Do not duplicate hard-rule territory '
    '(verification / turn-end / loop-guard); the brief is butter, not a '
    'second sunscreen.\n\n'
    'Total length: ≤ 1500 characters. Omit a section if it has nothing to '
    'say — do NOT pad. If the session has produced no durable signal yet, '
    'return an empty string ("") to leave the prior brief alone.'
)


class CompactedResult(BaseModel):
    """Structured output for compacting."""
    summary: str = Field(description='Factual index of confirmed facts/decisions from the conversation, semicolon-separated if multiple topics')
    trigger: str = Field(description='Retrieval scenario: when would someone need this? Must differ from summary')
    recall_mode: str = Field(
        default='active',
        description='passive=always in context, active=on-demand, both=always + searchable',
    )
    topic_tags: list[str] = Field(description='1-3 topic tags')
    importance: str = Field(
        default='MED',
        description=(
            'Prior on likelihood raw must be re-opened. HIGH for persistent artifacts '
            '(files/decisions/user corrections/constraints), LOW for transient reasoning or '
            'exploratory tool calls (summary is enough), MED otherwise.'
        ),
    )
    session_brief: str = Field(
        default='',
        description=(
            'Optional full rewrite of this session\'s brief (not an append). '
            'Only produced for DIALOG modality. Empty string means "leave the '
            'existing brief untouched". When non-empty, must follow the fixed '
            'section skeleton enforced by the prompt — writing in the user\'s '
            'language for body content, English for section headers.'
        ),
    )


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
        self._llm: BaseChatModel | None = None
        self._structured_llm = None

    def _ensure_llm(self):
        if self._llm is None:
            self._llm = create_chat_model(self._config.main_model, self._config)
            self._structured_llm = self._llm.with_structured_output(CompactedResult)

    def compact(
        self,
        l1_buffer: list[L1Memory],
        depth_level: int = 1,
        session_id: Optional[str] = None,
        template_name: str = 'compacting',
        compaction_reason: Optional[str] = None,
        policy=None,
        preceding_summaries: list[str] | None = None,
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
        
        # Generate summary via LLM (with existing topic context).
        # Render turns as role-labeled blocks so summary can preserve
        # who-said-what (user request vs assistant action) instead of
        # flattening everything into one assistant-style narrative.
        turns_text = _format_turns_for_prompt(l1_buffer)
        existing_tags = self._store.get_all_tags()
        existing_tags = {t for t in existing_tags if not any(t.startswith(p) for p in self._META_PREFIXES)}
        tags_text = ', '.join(sorted(existing_tags)) if existing_tags else '(none)'

        preceding_context = ''
        if preceding_summaries:
            lines = '\n'.join(f'  - {s}' for s in preceding_summaries)
            preceding_context = (
                '### Preceding User Summaries (already indexed — avoid repeating)\n'
                f'{lines}\n\n'
            )

        if policy is not None:
            existing_brief = ''
            if session_id and getattr(policy, 'extract_rules', False):
                try:
                    existing_brief = self._store.load_session_brief(session_id)
                except Exception:
                    existing_brief = ''
            prompt = self._render_policy_prompt(
                policy, turns_text, tags_text, preceding_context, existing_brief,
            )
        else:
            language = self._config.get('language', 'the same language as the user')
            prompt = load_template(template_name).format(
                turns=turns_text,
                existing_tags=tags_text,
                language=language,
                preceding_context=preceding_context,
            )
        
        self._ensure_llm()
        result: CompactedResult = self._structured_llm.invoke(prompt)
        
        # Generate keys and save raw
        node_id = self._store.generate_node_id()
        content_key = self._store.generate_content_key(prefix='raw')
        raw_location = self._store.save_raw(content_key, raw_data)
        
        recall_mode = result.recall_mode if result.recall_mode in ('passive', 'active', 'both') else 'active'

        # Create memory node
        tags = [t for t in result.topic_tags if not any(t.startswith(p) for p in self._META_PREFIXES)]
        if policy is not None:
            for hint in getattr(policy, 'tag_hints', ()):
                if hint not in tags:
                    tags.append(hint)

        # Importance prior is stored as a meta-prefixed tag (same pattern as
        # ORIGIN:/FLAG:/SESSION:) so the existing meta-prefix filter keeps
        # it out of topic-tag enumeration while the renderer can still
        # pull it into the `(O: A: I:)` short-tag block.
        importance = (result.importance or 'MED').upper()
        if importance not in ('HIGH', 'MED', 'LOW'):
            importance = 'MED'
        tags.append(f'IMPORTANCE:{importance}')

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
        if session_id:
            self._store.link_node_to_session(session_id, node_id)

        # Persist the session brief (full rewrite). Only DIALOG modality
        # produces a non-empty brief — other modalities return ''.
        if session_id and result.session_brief and result.session_brief.strip():
            try:
                self._store.save_session_brief(session_id, result.session_brief.strip())
            except Exception as e:
                logger.warning(f'save_session_brief failed (non-fatal): {e}')

        # Auto-link: find existing nodes with overlapping topic tags
        self._auto_link(node)

        if self._vector_index:
            try:
                self._vector_index.upsert(node, raw_content=raw_data)
            except Exception as e:
                logger.warning(f'VectorIndex upsert failed (non-fatal): {e}')

        return node

    def _render_policy_prompt(self, policy, turns_text: str, tags_text: str,
                              preceding_context: str = '',
                              existing_brief: str = '') -> str:
        policy_block = policy.render_compactor_instructions()
        modality = getattr(policy, 'modality', 'dialog')
        extract_rules = getattr(policy, 'extract_rules', False)

        if modality == 'dialog':
            summary_instr = (
                'Factual index of confirmed facts/decisions that lets a future agent judge node relevance without opening raw. '
                'Include brief action context (user request, debugging, implementation, etc.). '
                'Semicolon-separated if multiple topics. '
                'Convert ALL relative time expressions ("last year", "yesterday", "a few weeks ago", "next month") '
                'to absolute dates/periods using session timestamps as anchor.'
            )
            trigger_instr = (
                'Describe WHEN/WHY raw becomes worth opening. Start with "When I...". '
                'MUST differ from summary. 2-4 anchors only. Avoid broad recall phrases.'
            )
            recall_instr = (
                'active (default), passive (permanent instructions), '
                'both (critical constraints)'
            )
        elif modality == 'artifact_code':
            summary_instr = (
                'Start with language/type (e.g. "Python module"). '
                'Include file name, module role, key exports so a future agent can judge relevance before opening raw.'
            )
            trigger_instr = (
                '"When I need to inspect or modify the exact implementation of [export1], [export2] '
                'in [file context]". Use only 2-4 anchors.'
            )
            recall_instr = 'Always "active" for code.'
        elif modality == 'artifact_image':
            summary_instr = 'Describe visual content, source, dimensions, format.'
            trigger_instr = '"When I need visual verification from [image context]"'
            recall_instr = 'Always "active".'
        elif modality == 'execution_trace':
            summary_instr = (
                'Describe execution outcome concisely so a future agent can judge relevance without raw — '
                'tool name, success/failure, key output values.'
            )
            trigger_instr = '"When I need to verify exact results, errors, or output values from [execution context]"'
            recall_instr = 'Always "active".'
        else:
            summary_instr = (
                'Describe ACTUAL FACTS contained (1-2 lines) so a future agent can judge relevance before opening raw. '
                'Include specific names, numbers, conclusions.'
            )
            trigger_instr = (
                '"When I need to verify [anchor1], [anchor2] exact values or source details from [context]". '
                'STRICT: 1 sentence only. Max 2-4 anchor keywords. '
                'Do NOT list every entity from the summary. '
                'trigger != summary; trigger describes WHEN raw becomes worth opening, not WHAT is stored.'
            )
            recall_instr = 'Always "active" for external content.'

        if extract_rules:
            brief_instr = _SESSION_BRIEF_INSTRUCTION
        else:
            brief_instr = (
                'Return empty string "". Session briefs are only produced '
                'for dialog-modality nodes.'
            )

        extra_tag = ''
        tag_hints = getattr(policy, 'tag_hints', ())
        if tag_hints:
            extra_tag = f'- MUST include: {", ".join(tag_hints)}'

        base_template = load_template('compacting_base')
        language = self._config.get('language', 'the same language as the user')
        brief_block = ''
        if existing_brief and existing_brief.strip():
            brief_block = (
                '### Previous Session Brief (for reference when rewriting)\n'
                f'{existing_brief.strip()}\n\n'
            )
        return base_template.format(
            turns=turns_text,
            policy_block=policy_block,
            summary_instruction=summary_instr,
            trigger_instruction=trigger_instr,
            recall_instruction=recall_instr,
            existing_tags=tags_text,
            extra_tag_instruction=extra_tag,
            brief_instruction=brief_instr,
            preceding_context=preceding_context+brief_block,
            language=language,
        )

    _META_PREFIXES = ('ORIGIN:', 'FLAG:', 'SESSION:', 'IMPORTANCE:')

    @staticmethod
    def _topic_only(tags):
        _prefixes = ('ORIGIN:', 'FLAG:', 'SESSION:', 'IMPORTANCE:')
        return {t.lower() for t in tags if not any(t.startswith(p) for p in _prefixes)}

    def _auto_link(self, new_node: MemoryNode):
        """Link new node to existing cross-session nodes.

        Same-session nodes are skipped — session-internal retrieval already
        guarantees 100% recall, so intra-session links add density without value.

        Cross-session linking uses relaxed thresholds (tag ≥1, sim ≥0.40)
        to maximize cross-session knowledge discovery.
        """
        consolidation = self._config.get('consolidation', {})
        cross_min_overlap = consolidation.get('cross_session_min_tag_overlap', 1)
        cross_sim_threshold = consolidation.get('cross_session_link_threshold', 0.40)

        new_topics = self._topic_only(new_node.topic_tags)
        if not new_topics:
            return

        # Collect cross-session candidates with tag overlap
        new_session = new_node.session_id
        tag_matched: set[str] = set()
        for existing in self._store.list_all():
            existing_id = existing['node_id']
            if existing_id == new_node.node_id:
                continue
            # Skip same-session nodes
            if new_session and existing.get('session_id') == new_session:
                continue
            existing_topics = self._topic_only(existing.get('topic_tags', []))
            if len(new_topics & existing_topics) >= cross_min_overlap:
                tag_matched.add(existing_id)

        if not tag_matched:
            return

        linked_ids: set[str]
        if self._vector_index and new_node.summary:
            vec_matched: set[str] = set()
            hits = self._vector_index.search_by_summary(new_node.summary, top_k=10)
            for hit in hits:
                if hit.node_id != new_node.node_id and 1.0-hit.score >= cross_sim_threshold:
                    vec_matched.add(hit.node_id)
            linked_ids = tag_matched & vec_matched
        else:
            linked_ids = {nid for nid in tag_matched
                          if self._tag_overlap_count(new_topics, nid) >= cross_min_overlap+1}

        for target_id in linked_ids:
            self.link_nodes(new_node.node_id, target_id)
            self.link_nodes(target_id, new_node.node_id)

    def _tag_overlap_count(self, new_topics: set[str], node_id: str) -> int:
        existing = self._store.get_node(node_id)
        if not existing:
            return 0
        return len(new_topics & self._topic_only(existing.topic_tags))

    def link_nodes(self, source_id: str, target_id: str):
        """Link two related memory nodes."""
        source = self._store.get_node(source_id)
        if source and target_id not in source.links:
            source.links.append(target_id)
            self._store.save_node(source)
