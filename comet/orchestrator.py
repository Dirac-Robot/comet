"""CoMeT Orchestrator: Main workflow coordinating Sensor, Compacter, and Store."""
import hashlib
import queue
import threading
import uuid
from datetime import datetime
from typing import Callable, Optional, Union

from ato.adict import ADict
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from comet.llm_factory import create_chat_model
from loguru import logger

from comet.schemas import L1Memory, MemoryNode, CognitiveLoad, RetrievalResult
from comet.sensor import CognitiveSensor
from comet.compacter import MemoryCompacter
from comet.storage import MemoryStore
from comet.vector_index import VectorIndex
from comet.retriever import Retriever, AnalyzedQuery
from comet.consolidator import Consolidator

import re

_PATH_BRACKET_RE = re.compile(r'\[Path:\s*(/[^\]]+?)\s*\]')
_UPLOADED_BRACKET_RE = re.compile(r'\[Uploaded file:\s*([^\]]+?)\s*\]')
_FILEPATH_RE = re.compile(r'(?:^|[\s;|])(/(?:Users|home|tmp|var|opt|etc)/[^\[\]|;\n]+?)\s*(?=[;\|\]\n]|$)')

MAX_DISPLAY_MERGE = 3


# ── Session-memory row tag priorities ──
#
# Mirrored from CoBrA's backend/services/tag_namespace.py (ORIGIN_PRIORITY /
# ACT_PRIORITY / KIND_PRIORITY). CoMeT cannot import CoBrA (dependency goes
# the other way), so these are duplicated. Update both sides together when
# adding a tag. Higher integer = higher priority; missing keys default to 0.
_KNOWN_ORIGINS: tuple[str, ...] = (
    'ORIGIN:USER', 'ORIGIN:WEB_SEARCH', 'ORIGIN:WEB_PAGE',
    'ORIGIN:FILE_READ', 'ORIGIN:FILE_WRITE', 'ORIGIN:FILE_EDIT', 'ORIGIN:FILE_UPLOAD',
    'ORIGIN:IMAGE_READ', 'ORIGIN:IMAGE_DETECT', 'ORIGIN:IMAGE_SEGMENT', 'ORIGIN:IMAGE_DEPTH',
    'ORIGIN:BROWSER_READ', 'ORIGIN:TERMINAL_EXEC',
    'ORIGIN:PROJECT_MAP', 'ORIGIN:GREP_RESULT', 'ORIGIN:FIND_FILES',
    'ORIGIN:CODE', 'ORIGIN:EXTERNAL',
    'ORIGIN:SESSION_SCAN', 'ORIGIN:CROSS_SESSION_MESSAGE',
    'ORIGIN:TOOL_BUNDLE', 'ORIGIN:META_BUNDLE',
    'ORIGIN:SESSION_HANDOFF', 'ORIGIN:SUBAGENT_RESULT', 'ORIGIN:PROJECT_GOAL',
)
_ORIGIN_OVERRIDES: dict[str, int] = {
    'ORIGIN:USER': 100,
    'ORIGIN:SUBAGENT_RESULT': 80,
    'ORIGIN:SESSION_HANDOFF': 75,
    'ORIGIN:PROJECT_GOAL': 70,
    'ORIGIN:TOOL_BUNDLE': 60,
    'ORIGIN:META_BUNDLE': 60,
    'ORIGIN:FILE_EDIT': 55,
    'ORIGIN:FILE_WRITE': 55,
    'ORIGIN:CODE': 55,
    'ORIGIN:TERMINAL_EXEC': 52,
    'ORIGIN:CROSS_SESSION_MESSAGE': 45,
    'ORIGIN:EXTERNAL': 30,
}
_ORIGIN_PRIORITY: dict[str, int] = (
    {o: 50 for o in _KNOWN_ORIGINS} | _ORIGIN_OVERRIDES
)

_ACT_PRIORITY: dict[str, int] = {
    'FLAG:ACT_FAIL': 100,
    'FLAG:ACT_EDIT': 80,
    'FLAG:ACT_EXECUTE': 70,
    'FLAG:ACT_DIAGNOSE': 60,
    'FLAG:ACT_FETCH': 40,
    'FLAG:ACT_PLAN': 30,
    'FLAG:ACT_DECIDE': 20,
    'FLAG:ACT_NONE': 0,
}

_KIND_PRIORITY: dict[str, int] = {
    'FLAG:SKILL': 100,
    'FLAG:USER_REJECT': 80,
    'FLAG:USER_FEEDBACK': 70,
    'FLAG:PASSIVE': 10,
}


def _is_prebuilt_tagged(tags) -> bool:
    """True if tags carry a PREBUILT_V:<n> or PREBUILT_CAT:<bucket> marker.

    Prebuilt nodes are hidden library content surfaced via search /
    read_memory_node, never as session-authored memory. Callers that
    render the live session map use this to keep the prebuilt layer out
    of the displayed/promtped node list.
    """
    for t in tags or ():
        if isinstance(t, str) and (t.startswith('PREBUILT_V:') or t.startswith('PREBUILT_CAT:')):
            return True
    return False


def _pick_highest(tags, priority_map: dict[str, int]) -> str:
    """Return the highest-priority tag from ``tags`` per ``priority_map``.

    Unknown tag strings default to score 0 and lose to any priority-mapped
    tag. Returns '' if no tag has positive priority (e.g. node carries only
    FLAG:ACT_NONE or no tags at all).
    """
    best_tag = ''
    best_score = 0
    for t in tags or ():
        score = priority_map.get(t, 0)
        if score > best_score:
            best_score = score
            best_tag = t
    return best_tag


def _extract_source_links(content: str) -> list[str]:
    paths = []
    for m in _PATH_BRACKET_RE.finditer(content):
        p = m.group(1).strip()
        if p and p not in paths:
            paths.append(p)
    for m in _UPLOADED_BRACKET_RE.finditer(content):
        p = m.group(1).strip()
        if p and p not in paths:
            paths.append(p)
    for m in _FILEPATH_RE.finditer(content):
        p = m.group(1).strip()
        if p and p not in paths:
            paths.append(p)
    return paths


MessageInput = Union[str, dict, list, BaseMessage]

_DETAIL_PROMPT = (
    'Given the following raw content from a memory node, write a detailed summary '
    '(3-8 sentences) that captures the KEY facts, entities, relationships, and '
    'conclusions. Preserve specific numbers, names, and technical details. '
    'Do NOT include filler phrases like "This document discusses...".\n\n'
    '--- RAW CONTENT ---\n{raw}\n--- END ---\n\n'
    'Detailed Summary:'
)


class CoMeT:
    """
    Cognitive Memory OS - Dynamic Resolution Memory System.

    Main orchestrator that:
    1. Receives input through add()
    2. SLM sensor extracts L1 and assesses cognitive load
    3. Triggers compacting when load/shift detected
    4. Provides read_memory(key, depth) for navigation
    """

    def __init__(self, config: ADict, session_id: Optional[str] = None,
                 store: Optional[MemoryStore] = None,
                 vector_index: Optional['VectorIndex'] = None):
        if isinstance(config, dict) and not isinstance(config, ADict):
            config = ADict(config)
        self._config = config
        self._store = store or MemoryStore(config)
        self._vector_index = vector_index if vector_index is not None else (
            VectorIndex(config) if config.get('retrieval') else None
        )

        self._recover_pending_snapshots()

        self._sensor = CognitiveSensor(config)
        self._compacter = MemoryCompacter(config, self._store, self._vector_index)
        self._retriever = Retriever(config, self._store, self._vector_index) if self._vector_index else None
        self._consolidator = Consolidator(config, self._store, self._vector_index) if self._vector_index else None
        self._l1_buffer: list[L1Memory] = []
        self._last_load: Optional[CognitiveLoad] = None
        self._session_id: str = session_id or uuid.uuid4().hex[:12]
        self._session_node_ids: list[str] = []
        self._pinned_node_ids: set[str] = set()
        self._ingest_hashes: set[str] = set()
        self._pending_external_links: list[str] = []
        self._pending_read_links: list[str] = []
        self._buffer_origin: str = 'USER'
        self._buffer_extra_tags: set[str] = set()
        self._lock = threading.Lock()
        self._detail_llm: Optional[BaseChatModel] = None
        self._ingest_queue: queue.Queue = queue.Queue()

        existing_nodes = self._store.list_by_session(self._session_id)
        if existing_nodes:
            self._session_node_ids = [n['node_id'] for n in existing_nodes]
            logger.info(f'Restored {len(self._session_node_ids)} nodes for session {self._session_id}')
        else:
            self._store.save_session_meta(self._session_id, {
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'node_count': 0,
            })

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def l1_buffer(self) -> list[L1Memory]:
        return self._l1_buffer

    @property
    def last_load(self) -> Optional[CognitiveLoad]:
        return self._last_load

    @staticmethod
    def _normalize_content(content: MessageInput) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, BaseMessage):
            return f"[{content.type}] {content.content}"
        if isinstance(content, dict):
            role = content.get('role', 'unknown')
            text = content.get('content', '')
            return f"[{role}] {text}"
        raise TypeError(f'Unsupported content type: {type(content)}')

    def add(
        self,
        content: MessageInput,
        *,
        origin: str | None = None,
        extra_tags: list[str] | None = None,
    ) -> Optional[MemoryNode]:
        """
        Add new content to the memory system.

        Accepts:
          - str: plain text
          - dict: {'role': 'user', 'content': '...'}
          - list[str | dict | BaseMessage]: batch input (calls add per item)
          - BaseMessage: LangChain message object

        Args:
            origin: Origin tag for the content (e.g. 'USER', 'SESSION').
                    If provided, overrides the buffer origin for the resulting node.
                    Defaults to 'USER' if never set.
            extra_tags: Additional meta tags (e.g. 'FLAG:USER_FEEDBACK') to attach
                    to the resulting compacted node. Accumulated across add() calls
                    while the buffer is open and applied once at compaction time.

        Returns MemoryNode if compacting was triggered, else None.
        For list input, returns the last compacted node (if any).
        """
        if isinstance(content, list):
            return self.add_many(content)

        text = self._normalize_content(content)
        l1_mem = self._sensor.extract_l1(text)

        session_summaries = self._get_session_summaries()
        load = self._sensor.assess_load(text, self._l1_buffer, session_summaries)
        self._last_load = load

        logger.info(f"Sensor: load_level={load.load_level}, redundancy={load.redundancy_detected}, logic={load.logic_flow}")

        with self._lock:
            if origin is not None:
                self._buffer_origin = origin
            if extra_tags:
                self._buffer_extra_tags.update(extra_tags)
            reason = self._sensor.get_compaction_reason(load, len(self._l1_buffer))
            if self._l1_buffer and reason:
                node = self._compact_buffer(compaction_reason=reason)
                self._l1_buffer = [l1_mem]
                self._session_node_ids.append(node.node_id)
                logger.info(f"Compacted to node: {node.node_id} (reason={reason})")
                self._on_post_add(load)
                return node

            self._l1_buffer.append(l1_mem)
            self._on_post_add(load)
            return None

    def _get_session_summaries(self) -> list[str]:
        """Extract summaries from current session nodes for redundancy detection."""
        unique_ids = list(dict.fromkeys(self._session_node_ids))
        summaries = []
        for nid in unique_ids:
            node = self._store.get_node(nid)
            if node:
                summaries.append(node.summary)
        return summaries

    def _get_user_preceding_summaries(self, max_count: int = 3) -> list[str]:
        unique_ids = list(dict.fromkeys(self._session_node_ids))
        summaries = []
        for nid in unique_ids:
            node = self._store.get_node(nid)
            if node is None:
                continue
            tags_upper = [t.upper() for t in (node.topic_tags or [])]
            if 'ORIGIN:USER' in tags_upper and node.summary:
                summaries.append(node.summary)
        return summaries[-max_count:] if len(summaries) > max_count else summaries

    def _on_post_add(self, load: CognitiveLoad):
        """Hook called after add(). Subclasses can extend for post-processing."""
        pass

    def add_many(self, messages: list[MessageInput]) -> Optional[MemoryNode]:
        """
        Add multiple messages sequentially.

        Returns the last compacted MemoryNode (if any).
        """
        last_node = None
        for msg in messages:
            result = self.add(msg)
            if result is not None:
                last_node = result
        return last_node

    def add_document(
        self,
        content: str,
        source: str = '',
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        background: bool = False,
        on_complete: Optional[Callable] = None,
    ) -> list[MemoryNode]:
        """
        Ingest a long document or file into memory.

        Chunks the content and feeds each chunk through the standard
        add() pipeline (sensor L1 extraction + auto-compaction).
        Flushes remaining buffer at the end.

        Args:
            content: Full document/file text.
            source: Source identifier (e.g. URL, file path).
            chunk_size: Max characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            background: If True, queue for background processing and return [] immediately.
            on_complete: Callback invoked with list[MemoryNode] when background ingestion finishes.

        Returns list of MemoryNodes created during ingestion (empty if background=True).
        """
        if background:
            self._ingest_queue.put((content, source, chunk_size, chunk_overlap, on_complete))
            return []

        if not content.strip():
            return []

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash in self._ingest_hashes:
            logger.info(f'Document skipped (duplicate hash={content_hash}, source={source!r})')
            return []
        self._ingest_hashes.add(content_hash)

        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        prefix = f'[Source: {source}] ' if source else ''
        nodes = []

        for chunk in chunks:
            result = self.add(f'{prefix}{chunk}')
            if result is not None:
                nodes.append(result)

        remaining = self.force_compact()
        if remaining is not None:
            nodes.append(remaining)

        logger.info(f'Document ingested: {len(chunks)} chunks -> {len(nodes)} nodes (source={source!r})')
        return nodes

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks, breaking at sentence/line boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start+chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            boundary = end
            for sep in ['\n\n', '\n', '. ', ', ', ' ']:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    boundary = pos+len(sep)
                    break

            chunks.append(text[start:boundary])
            start = boundary-overlap if overlap < boundary-start else boundary

        return chunks

    def _ingest_loop(self):
        """Background worker: process queued documents."""
        while True:
            try:
                item = self._ingest_queue.get()
                if item is None:
                    break
                content, source, chunk_size, chunk_overlap, on_complete = item
                nodes = self.add_document(content, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if on_complete is not None:
                    try:
                        on_complete(nodes)
                    except Exception as cb_err:
                        logger.warning(f'on_complete callback error: {cb_err}')
            except Exception as e:
                logger.warning(f'Background ingest error: {e}')
            finally:
                self._ingest_queue.task_done()

    def drain(self, timeout: float = None) -> bool:
        """Block until all queued background ingestions are processed.

        Returns True if drained successfully, False on timeout.
        """
        try:
            self._ingest_queue.join()
            return True
        except Exception:
            return False

    def _compact_buffer(self, compaction_reason: Optional[str] = None) -> MemoryNode:
        """Compact current L1 buffer into a MemoryNode."""
        if not self._l1_buffer:
            raise ValueError("Cannot compact empty buffer")

        preceding = self._get_user_preceding_summaries()
        node = self._compacter.compact(
            self._l1_buffer, session_id=self._session_id,
            compaction_reason=compaction_reason,
            preceding_summaries=preceding or None,
        )

        origin_tag = f'ORIGIN:{self._buffer_origin}'
        mutated = False
        if origin_tag not in node.topic_tags:
            node.topic_tags.append(origin_tag)
            mutated = True
        for extra in self._buffer_extra_tags:
            if extra and extra not in node.topic_tags:
                node.topic_tags.append(extra)
                mutated = True
        if mutated:
            self._store.save_node(node)
        self._buffer_origin = 'USER'  # reset to default after compaction
        self._buffer_extra_tags.clear()

        if self._pending_external_links:
            for ext_id in self._pending_external_links:
                self._compacter.link_nodes(node.node_id, ext_id)
                self._compacter.link_nodes(ext_id, node.node_id)
                logger.info(f'Linked turn node {node.node_id} <-> external {ext_id}')
            self._pending_external_links.clear()

        if self._pending_read_links:
            for read_id in self._pending_read_links:
                if read_id not in node.links:
                    self._compacter.link_nodes(node.node_id, read_id)
                    self._compacter.link_nodes(read_id, node.node_id)
                    logger.info(f'Auto-linked read node {read_id} <-> {node.node_id}')
            self._pending_read_links.clear()

        return node

    def add_external(
        self,
        content: str,
        source_tag: str = 'external',
        template_name: str = 'compacting_external',
        policy=None,
        source_links: list[str] | None = None,
    ) -> MemoryNode:
        """
        Ingest external content (e.g. web search results) as a separate node.

        Bypasses the L1 buffer and directly creates an L2 MemoryNode.
        The node is linked to the next compacted turn node automatically.
        If policy is provided, it takes precedence over template_name.
        If source_links is provided, it overrides regex-based path extraction.
        """
        l1_mem = self._sensor.extract_l1(content)
        l1_mem.raw_content = content

        compact_kwargs = {
            'session_id': self._session_id,
            'compaction_reason': 'external',
        }
        if policy is not None:
            compact_kwargs['policy'] = policy
        else:
            compact_kwargs['template_name'] = template_name

        node = self._compacter.compact(
            [l1_mem],
            **compact_kwargs,
        )

        if source_tag:
            origin_tag = source_tag.upper() if source_tag.upper().startswith('ORIGIN:') else f'ORIGIN:{source_tag.upper()}'
            if origin_tag not in node.topic_tags:
                node.topic_tags.append(origin_tag)

        if source_links is not None:
            node.source_links = source_links
        else:
            paths = _extract_source_links(content)
            if paths:
                node.source_links = paths

        self._store.save_node(node)

        with self._lock:
            self._session_node_ids.append(node.node_id)
            self._pending_external_links.append(node.node_id)

        logger.info(f'External node created: {node.node_id} (source={source_tag})')
        return node

    def force_compact(self) -> Optional[MemoryNode]:
        """Force compacting of current buffer."""
        with self._lock:
            if not self._l1_buffer:
                return None

            node = self._compact_buffer(compaction_reason='forced')
            self._l1_buffer = []
            self._session_node_ids.append(node.node_id)
            return node

    def _recover_pending_snapshots(self):
        """Detect and restore incomplete consolidation/synthesis snapshots from previous run."""
        for label in ('consolidation', 'synthesis'):
            if self._store.has_pending_snapshot(label):
                logger.warning(f'Detected incomplete {label} snapshot, restoring...')
                self._store.restore_snapshot(label)
                if self._vector_index:
                    try:
                        self._vector_index.reset()
                        for entry in self._store.list_all():
                            node = self._store.get_node(entry['node_id'])
                            if node:
                                raw = self._store.get_raw(node.content_key) or ''
                                self._vector_index.upsert(node, raw_content=raw[:8000])
                        logger.info(f'VectorIndex rebuilt after {label} recovery')
                    except Exception as e:
                        logger.warning(f'VectorIndex rebuild after {label} recovery failed (non-fatal): {e}')

    def consolidate(self, node_ids: Optional[list[str]] = None) -> dict:
        """Manually consolidate nodes into the RAG knowledge base.

        Runs dedup, cross-linking, and tag normalization.
        If node_ids is None, consolidates all nodes.
        """
        if not self._consolidator:
            logger.warning('Consolidator not available (retrieval config missing)')
            return {'status': 'skipped', 'reason': 'no_consolidator'}
        return self._consolidator.consolidate(node_ids)

    def synthesize(self, threshold: Optional[float] = None) -> list[MemoryNode]:
        """Create virtual nodes by clustering semantically related memories.

        Cross-session knowledge synthesis:
        1. Embedding-based clustering to find related nodes
        2. SLM validates each cluster is coherent
        3. SLM generates synthesized summary/trigger
        4. Virtual node stored with bidirectional links to sources
        """
        if not self._consolidator:
            logger.warning('Consolidator not available (retrieval config missing)')
            return []
        return self._consolidator.synthesize(threshold)

    def close_session(self) -> dict:
        """End current session: force-compact remaining buffer, then consolidate session nodes."""
        self.force_compact()

        node_count = len(self._session_node_ids)
        if not self._session_node_ids:
            logger.info('No session nodes to consolidate')
            self._store.save_session_meta(self._session_id, {
                'status': 'closed',
                'created_at': self._store.get_session_meta(self._session_id).get('created_at', ''),
                'closed_at': datetime.now().isoformat(),
                'node_count': 0,
            })
            return {'status': 'empty', 'session_id': self._session_id}

        result = self.consolidate(self._session_node_ids)
        self._store.save_session_meta(self._session_id, {
            'status': 'closed',
            'created_at': self._store.get_session_meta(self._session_id).get('created_at', ''),
            'closed_at': datetime.now().isoformat(),
            'node_count': node_count,
            'node_ids': list(self._session_node_ids),
        })
        self._session_node_ids = []
        logger.info(f'Session {self._session_id} closed: {node_count} nodes consolidated')
        return {**result, 'session_id': self._session_id}

    def read_memory(self, node_id: str, depth: int = 0) -> Optional[str]:
        """
        Navigation Tool: Read memory at specified depth.

        depth=0: Summary only
        depth=1: Detailed summary (lazy-generated) + metadata
        depth=2: Full raw data
        """
        if node_id not in self._pending_read_links and node_id not in self._session_node_ids:
            self._pending_read_links.append(node_id)
        if depth == 1:
            detailed = self.get_detailed_summary(node_id)
            return self._store.read_memory(node_id, depth, detailed_summary=detailed)
        return self._store.read_memory(node_id, depth)

    def get_raw_content(self, node_id: str) -> Optional[str]:
        """Retrieve full raw content from vector store by node_id key."""
        return self._vector_index.get_raw(node_id)

    def get_detailed_summary(self, node_id: str) -> Optional[str]:
        """Lazy detailed summary: return cached if exists, else generate from raw."""
        node = self._store.get_node(node_id)
        if node is None:
            return None
        if node.detailed_summary:
            return node.detailed_summary
        raw = self.get_raw_content(node_id)
        if not raw:
            return node.summary
        if self._detail_llm is None:
            self._detail_llm = create_chat_model(self._config.slm_model, self._config)
        prompt = _DETAIL_PROMPT.format(raw=raw[:6000])
        response = self._detail_llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = ' '.join(
                c.get('text', '') if isinstance(c, dict) else str(c)
                for c in content
            )
        detailed = content.strip()
        node.detailed_summary = detailed
        self._store.save_node(node)
        logger.info(f'Generated detailed summary for {node_id} ({len(detailed)} chars)')
        return detailed

    def search(self, tag: str) -> list[str]:
        """Search nodes by topic tag."""
        return self._store.search_by_tag(tag)

    def list_high_importance_nodes(
        self, session_id: Optional[str] = None, limit: int = 20,
    ) -> list[dict]:
        """Return this session's IMPORTANCE:HIGH nodes, most recent first.

        Used by the handoff path to curate which nodes get carried over to
        the successor session. Bounded by `limit` to keep the successor's
        injected block compact.
        """
        sid = session_id or self._session_id
        if not sid:
            return []
        entries = self._store.list_by_session(sid) or []
        high: list[dict] = []
        for entry in entries:
            tags = entry.get('topic_tags') or []
            if any(t == 'IMPORTANCE:HIGH' for t in tags):
                high.append(entry)
        # Most recent first — list_by_session follows insertion order, so
        # reverse to prioritize recency.
        high.reverse()
        return high[:max(0, limit)]

    def get_session_brief(self, session_id: Optional[str] = None) -> str:
        """Return the per-session brief (empty string if none exists)."""
        sid = session_id or self._session_id
        if not sid:
            return ''
        return self._store.load_session_brief(sid)

    def set_session_brief(self, brief: str, session_id: Optional[str] = None) -> None:
        """Overwrite the per-session brief (for manual edits / imports)."""
        sid = session_id or self._session_id
        if not sid:
            return
        self._store.save_session_brief(sid, brief)

    def delete_session_brief(self, session_id: Optional[str] = None) -> bool:
        sid = session_id or self._session_id
        if not sid:
            return False
        return self._store.delete_session_brief(sid)

    def regenerate_brief(
        self,
        session_id: Optional[str] = None,
        reason: str = '',
        relevant_tag_filter: Optional[frozenset] = None,
        signal_limit: int = 15,
    ) -> str:
        """Force a full rewrite of the session brief outside the dialog-compact path.

        Trigger sites (user-feedback detection, action-flag resolver, handoff
        chunker) mark ``comet._brief_dirty``; the render side calls this to
        rebuild before the next LLM invoke. Collects recent nodes carrying
        ``relevant_tag_filter`` tags + the prior brief, asks the SLM for a
        full rewrite using ``_SESSION_BRIEF_INSTRUCTION``, persists the
        result. Returns the new brief (or prior brief on failure / no signal).
        """
        sid = session_id or self._session_id
        if not sid:
            return ''

        prior_brief = self._store.load_session_brief(sid)
        entries = self._store.list_by_session(sid) or []
        signals: list[dict] = []
        for entry in reversed(entries):
            if relevant_tag_filter is not None:
                tags = set(entry.get('topic_tags') or [])
                if not (tags & relevant_tag_filter):
                    continue
            signals.append(entry)
            if len(signals) >= signal_limit:
                break

        if not signals:
            # Nothing brief-relevant on record — do not rewrite.
            return prior_brief

        signal_lines = []
        for s in signals:
            tags = ', '.join(s.get('topic_tags') or [])
            summary = (s.get('summary') or '').strip()
            signal_lines.append(f"- [{s.get('node_id')}] ({tags}) {summary}")

        from comet.compacter import _SESSION_BRIEF_INSTRUCTION
        prompt = (
            f"You are rewriting the session brief — a full rewrite driven by "
            f"an out-of-band event (reason: {reason or 'unspecified'}).\n\n"
            f"## Previous Session Brief\n{prior_brief or '(none)'}\n\n"
            f"## Brief-relevant signals (most recent {len(signals)})\n"
            f"{chr(10).join(signal_lines)}\n\n"
            f"## Rewrite rule\n{_SESSION_BRIEF_INSTRUCTION}\n\n"
            f"Return ONLY the new brief text (no preamble, no commentary)."
        )

        try:
            llm = create_chat_model(self._config.slm_model, self._config)
        except Exception as e:
            logger.warning(f'regenerate_brief: SLM init failed for {sid}: {e}')
            return prior_brief
        try:
            from langchain_core.messages import HumanMessage
            result = llm.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.warning(f'regenerate_brief: SLM invoke failed for {sid}: {e}')
            return prior_brief

        raw = result.content if hasattr(result, 'content') else str(result)
        if isinstance(raw, list):
            raw = '\n'.join(
                part.get('text', '') if isinstance(part, dict) else str(part)
                for part in raw
            )
        new_brief = (raw or '').strip()
        if not new_brief:
            return prior_brief

        self._store.save_session_brief(sid, new_brief)
        logger.info(f'regenerate_brief: {sid} rewritten ({len(new_brief)} chars, reason={reason or "unspecified"})')
        return new_brief

    def list_memories(self) -> list[dict]:
        """List all stored memory nodes."""
        return self._store.list_all()

    def list_session_memories(self, session_id: Optional[str] = None) -> list[dict]:
        """List memory nodes for a specific session."""
        return self._store.list_by_session(session_id or self._session_id)

    def list_sessions(self) -> list[dict]:
        """List all registered sessions with metadata."""
        return self._store.list_sessions()

    def pin_node(self, node_id: str) -> bool:
        """Pin an external node to the current session context."""
        node = self._store.get_node(node_id)
        if node is None:
            return False
        self._pinned_node_ids.add(node_id)
        return True

    def unpin_node(self, node_id: str) -> bool:
        """Unpin a node from the current session context."""
        if node_id in self._pinned_node_ids:
            self._pinned_node_ids.discard(node_id)
            return True
        return False

    def get_session_context(self, session_id: Optional[str] = None, max_nodes: int = 50) -> str:
        """Get context window scoped to a specific session's nodes.

        Renders two sections: [External Nodes] (pinned from other sessions)
        and [Session Memory] (own nodes). External section is omitted if empty.

        Prebuilt nodes (PREBUILT_V:*, PREBUILT_CAT:*) are excluded from the
        rendered map — they are hidden library content, not session-authored
        memory. They remain reachable via search/read_memory_node.
        """
        target_id = session_id or self._session_id
        session_nodes = self._store.list_by_session(target_id)
        # Defense in depth: even if a future code path links a prebuilt
        # node to this session, the render hides it. The tag prefix
        # check matches both PREBUILT_V:<n> (version) and
        # PREBUILT_CAT:<bucket> (category) tags written by
        # ingest_prebuilt_memories.
        session_nodes = [
            n for n in session_nodes
            if not _is_prebuilt_tagged(n.get('topic_tags') or [])
        ]
        seen_ids = {n['node_id'] for n in session_nodes}

        own_entries = [(n.get('created_at', ''), n['node_id'], n) for n in session_nodes]

        pinned_entries = []
        if not session_id or session_id == self._session_id:
            for pid in self._pinned_node_ids:
                if pid in seen_ids:
                    continue
                pnode = self._store.get_node(pid)
                if pnode is None:
                    continue
                pdict = {
                    'node_id': pid,
                    'summary': pnode.summary or '',
                    'trigger': pnode.trigger or '',
                    'recall_mode': getattr(pnode, 'recall_mode', 'active'),
                    'topic_tags': pnode.topic_tags or [],
                    'created_at': getattr(pnode, 'created_at', ''),
                }
                pinned_entries.append((pdict.get('created_at', ''), pid, pdict))

        own_entries.sort(key=lambda x: x[0])
        pinned_entries.sort(key=lambda x: x[0])

        # Budget split: pinned gets first claim (usually few), own fills remainder.
        pinned_entries = pinned_entries[:max_nodes]
        own_entries = own_entries[: max(0, max_nodes - len(pinned_entries))]

        def _to_row(nid: str, n: dict) -> dict:
            summary = n.get('summary', '')
            trigger = n.get('trigger', '')
            recall = n.get('recall_mode', 'active')
            prefix = '(passive) ' if recall in ('passive', 'both') else ''
            tags = n.get('topic_tags', []) or []

            # Cap each axis at one tag so `(O:X A:Y F:Z I:H)` stays bounded:
            # nodes sometimes carry multiple ORIGIN:* (bundle supersede) or
            # drifted FLAG:ACT_* values; rendering all of them turned rows
            # into noise and hid the most informative label. Priority maps
            # pick the single highest-signal value per axis.
            origin_tag = _pick_highest(tags, _ORIGIN_PRIORITY)
            act_tag = _pick_highest(tags, _ACT_PRIORITY)
            kind_tag = _pick_highest(tags, _KIND_PRIORITY)

            importance = None
            for t in tags:
                if isinstance(t, str) and t.startswith('IMPORTANCE:'):
                    importance = t[len('IMPORTANCE:'):].upper()
                    break

            short_tags: list[str] = []
            if origin_tag:
                short_tags.append(f"O:{origin_tag[len('ORIGIN:'):]}")
            # Skip ACT on user-authored nodes — the user turn is self-describing
            # and the action label is redundant noise there.
            if act_tag and origin_tag != 'ORIGIN:USER':
                short_tags.append(f"A:{act_tag[len('FLAG:ACT_'):]}")
            if kind_tag:
                short_tags.append(f"F:{kind_tag[len('FLAG:'):]}")
            # Importance prior (HIGH/MED/LOW → H/M/L). MED is the default and
            # would be pure noise in every row; only surface H and L so the
            # agent's attention lands on the extremes.
            if importance in ('HIGH', 'LOW'):
                short_tags.append(f"I:{importance[0]}")

            tag_str = f"({' '.join(short_tags)}) " if short_tags else ''
            return {
                'nid': nid, 'tag_str': tag_str, 'prefix': prefix,
                'summary': summary, 'trigger': trigger, 'origin': origin_tag or '',
            }

        def _render_plain(rows: list[dict]) -> list[str]:
            return [f"[{r['nid']}] {r['tag_str']}{r['prefix']}{r['summary']} | {r['trigger']}" for r in rows]

        def _render_with_origin_merge(rows: list[dict]) -> list[str]:
            out: list[str] = []
            i = 0
            while i < len(rows):
                row = rows[i]
                if row['origin'] and row['origin'] != 'ORIGIN:USER':
                    group = [row]
                    j = i + 1
                    while j < len(rows) and rows[j]['origin'] == row['origin']:
                        group.append(rows[j])
                        j += 1
                    if len(group) >= 2:
                        for _cs in range(0, len(group), MAX_DISPLAY_MERGE):
                            chunk = group[_cs:_cs + MAX_DISPLAY_MERGE]
                            if len(chunk) == 1:
                                r = chunk[0]
                                out.append(f"[{r['nid']}] {r['tag_str']}{r['prefix']}{r['summary']} | {r['trigger']}")
                            else:
                                merged_nids = '+'.join(r['nid'].split('_')[-1] for r in chunk)
                                first_nid_prefix = '_'.join(chunk[0]['nid'].split('_')[:-1])
                                merged_id = f'{first_nid_prefix}_{merged_nids}'
                                merged_summaries = '; '.join(r['summary'] for r in chunk if r['summary'])
                                out.append(f"[{merged_id}] {chunk[0]['tag_str']}{chunk[0]['prefix']}{merged_summaries} | {chunk[0]['trigger']}")
                        i = j
                        continue
                out.append(f"[{row['nid']}] {row['tag_str']}{row['prefix']}{row['summary']} | {row['trigger']}")
                i += 1
            return out

        sections: list[str] = []
        if pinned_entries:
            pinned_rows = [_to_row(nid, n) for _, nid, n in pinned_entries]
            sections.append('[External Nodes] (pinned from other sessions)\n' + '\n'.join(_render_plain(pinned_rows)))
        if own_entries:
            own_rows = [_to_row(nid, n) for _, nid, n in own_entries]
            sections.append('[Session Memory]\n' + '\n'.join(_render_with_origin_merge(own_rows)))

        if not sections:
            return f'(No nodes for session {target_id})'
        return '\n\n'.join(sections)

    def get_context_window(self, max_nodes: int = 5) -> str:
        """Get context window: passive/both nodes always included, then recent active nodes."""
        parts = []
        all_nodes = self._store.list_all()

        passive_nodes = [
            n for n in all_nodes
            if n.get('recall_mode', 'active') in ('passive', 'both')
        ]
        for n in passive_nodes:
            summary = n.get('summary', '')
            trigger = n.get('trigger', '')
            parts.append(f"[{n['node_id']}] (passive) {summary} | {trigger}")

        remaining_slots = max(0, max_nodes-len(passive_nodes))
        if remaining_slots > 0:
            active_nodes = [
                n for n in all_nodes
                if n.get('recall_mode', 'active') == 'active'
            ]
            recent_active = sorted(
                active_nodes, key=lambda x: x.get('created_at', ''), reverse=True,
            )[:remaining_slots]
            for n in recent_active:
                summary = n.get('summary', '')
                trigger = n.get('trigger', '')
                parts.append(f"[{n['node_id']}] {summary} | {trigger}")

        if self._l1_buffer:
            for mem in self._l1_buffer[-2:]:
                parts.append(f"(recent) {mem.content[:40]}...")

        return '\n'.join(parts)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._retriever:
            logger.warning('Retriever not available (retrieval config missing)')
            return []
        return self._retriever.retrieve(query, top_k)

    def retrieve_dual(
        self,
        summary_query: str,
        trigger_query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not self._retriever:
            logger.warning('Retriever not available (retrieval config missing)')
            return []
        return self._retriever.retrieve_dual(summary_query, trigger_query, top_k)

    def retrieve_with_analysis(
        self, query: str, top_k: int = 5,
    ) -> tuple[list[RetrievalResult], AnalyzedQuery]:
        """Retrieve with full query analysis (including risk_level)."""
        if not self._retriever:
            logger.warning('Retriever not available (retrieval config missing)')
            return [], AnalyzedQuery(semantic_query=query, search_intent=query)
        return self._retriever.retrieve_with_analysis(query, top_k)

    def rebuild_index(self):
        if not self._retriever:
            logger.warning('Retriever not available (retrieval config missing)')
            return
        self._retriever.rebuild_index()

    def get_tools(self) -> list[BaseTool]:
        """
        Get LangChain-compatible tools for memory operations.

        Returns tools that can be used with any LangChain agent:
        - get_memory_index: List all memory nodes with triggers
        - read_memory_node: Read detailed content from a specific node
        - search_memory: Search nodes by topic tag
        - retrieve_memory: Dual-path semantic RAG search (if retrieval enabled)
        """
        memo = self  # Capture self for closure

        @tool
        def get_memory_index() -> str:
            """List stored memory nodes with summaries and triggers. Use when the question may depend on prior session facts, decisions, preferences, or stored artifacts."""
            return memo.get_context_window(max_nodes=50)

        @tool
        def read_memory_node(node_id: str) -> str:
            """Read the raw content of a specific memory node. Use when exact wording, numbers, code, paths, ownership, or sequence matters. node_id starts with 'mem_'."""
            result = memo.read_memory(node_id, depth=2)
            return result if result else f"Node {node_id} not found"

        @tool
        def search_memory(tag: str) -> str:
            """Search memory nodes by topic tag when you already know the topic label."""
            results = memo.search(tag)
            if not results:
                return f"No nodes found with tag: {tag}"
            return '\n'.join(results)

        tools = [get_memory_index, read_memory_node, search_memory]

        if self._retriever:
            @tool
            def retrieve_memory(summary_query: str, trigger_query: str) -> str:
                """Semantic search across memory. Use when the question may depend on prior memory or stored artifacts. Skip for self-contained rewriting, translation, or general knowledge that does not depend on memory.

                Uses dual-path retrieval:
                - summary_query: Core keyword/topic of the information you need.
                - trigger_query: The situation/context that triggered this search.

                Both parameters are required.
                Returns summaries and triggers only.
                Use read_memory_node(node_id) only if exact details matter, summaries conflict, or summaries are insufficient.
                """
                results, analyzed = memo._retriever.retrieve_with_analysis(
                    f'{summary_query} {trigger_query}',
                )
                if not results:
                    return 'No relevant memories found'
                parts = []
                for r in results:
                    linked = ', '.join(r.node.links) if r.node.links else 'none'
                    parts.append(
                        f'[{r.node.node_id}] (score={r.relevance_score:.4f})\n'
                        f'  Summary: {r.node.summary}\n'
                        f'  Trigger: {r.node.trigger}\n'
                        f'  Tags: {", ".join(r.node.topic_tags)}\n'
                        f'  Linked: {linked}'
                    )
                body = '\n\n'.join(parts)
                if analyzed.risk_level == 'high':
                    return (
                        '⚠️ HIGH RISK: Summary may be insufficient for exact values, wording, or sequence. '
                        'If your answer depends on this memory, verify the relevant raw node with read_memory_node.\n\n'
                        f'{body}'
                    )
                if analyzed.risk_level == 'low':
                    return (
                        'INFO: Overview-level query. Summaries may be enough; open raw only if you need exact wording or values.\n\n'
                        f'{body}'
                    )
                return body

            tools.append(retrieve_memory)

        return tools
