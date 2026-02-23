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

    def __init__(self, config: ADict, session_id: Optional[str] = None):
        if isinstance(config, dict) and not isinstance(config, ADict):
            config = ADict(config)
        self._config = config
        self._store = MemoryStore(config)
        self._vector_index = VectorIndex(config) if config.get('retrieval') else None

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

    def add(self, content: MessageInput) -> Optional[MemoryNode]:
        """
        Add new content to the memory system.

        Accepts:
          - str: plain text
          - dict: {'role': 'user', 'content': '...'}
          - list[str | dict | BaseMessage]: batch input (calls add per item)
          - BaseMessage: LangChain message object

        Returns MemoryNode if compacting was triggered, else None.
        For list input, returns the last compacted node (if any).
        """
        if isinstance(content, list):
            return self.add_many(content)

        text = self._normalize_content(content)
        l1_mem = self._sensor.extract_l1(text)

        load = self._sensor.assess_load(text, self._l1_buffer)
        self._last_load = load

        logger.debug(f"CogLoad: {load.logic_flow}, level={load.load_level}")

        with self._lock:
            reason = self._sensor.get_compaction_reason(load, len(self._l1_buffer))
            if self._l1_buffer and reason:
                node = self._compact_buffer(compaction_reason=reason)
                self._l1_buffer = [l1_mem]
                self._session_node_ids.append(node.node_id)
                logger.info(f"Compacted to node: {node.node_id} (reason={reason})")
                return node

            self._l1_buffer.append(l1_mem)
            return None

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

        node = self._compacter.compact(
            self._l1_buffer, session_id=self._session_id,
            compaction_reason=compaction_reason,
        )

        origin_tag = 'ORIGIN:USER'
        if origin_tag not in node.topic_tags:
            node.topic_tags.append(origin_tag)
            self._store.save_node(node)

        if self._pending_external_links:
            for ext_id in self._pending_external_links:
                self._compacter.link_nodes(node.node_id, ext_id)
                self._compacter.link_nodes(ext_id, node.node_id)
                logger.info(f'Linked turn node {node.node_id} <-> external {ext_id}')
            self._pending_external_links.clear()

        return node

    def add_external(
        self,
        content: str,
        source_tag: str = 'external',
        template_name: str = 'compacting_external',
    ) -> MemoryNode:
        """
        Ingest external content (e.g. web search results) as a separate node.

        Bypasses the L1 buffer and directly creates an L2 MemoryNode.
        The node is linked to the next compacted turn node automatically.
        """
        l1_mem = self._sensor.extract_l1(content)
        l1_mem.raw_content = content

        node = self._compacter.compact(
            [l1_mem],
            session_id=self._session_id,
            template_name=template_name,
            compaction_reason='external',
        )

        if source_tag and source_tag not in node.topic_tags:
            node.topic_tags.append(source_tag)
        origin_tag = f'ORIGIN:{source_tag.upper()}'
        if origin_tag not in node.topic_tags:
            node.topic_tags.append(origin_tag)
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
                    self._vector_index.reset()
                    for entry in self._store.list_all():
                        node = self._store.get_node(entry['node_id'])
                        if node:
                            raw = self._store.get_raw(node.content_key) or ''
                            self._vector_index.upsert(node, raw_content=raw[:8000])
                    logger.info(f'VectorIndex rebuilt after {label} recovery')

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

    def re_summarize(self, messages: list[MessageInput], session_id: Optional[str] = None) -> dict:
        """Delete session nodes and replay messages through sensor/compactor.

        Args:
            messages: List of messages to replay (str, dict, or BaseMessage).
            session_id: Target session. Defaults to current session.

        Returns:
            Dict with deleted/created counts.
        """
        target_sid = session_id or self._session_id
        existing_nodes = self._store.list_by_session(target_sid)
        deleted_count = 0
        for node_data in existing_nodes:
            nid = node_data['node_id']
            if self._vector_index:
                self._vector_index.delete(nid)
            self._store.delete_node(nid)
            deleted_count += 1
        logger.info(f'Re-summarize: deleted {deleted_count} nodes for session {target_sid}')

        original_sid = self._session_id
        self._session_id = target_sid

        with self._lock:
            self._l1_buffer = []
        self._session_node_ids = []

        created_nodes = []
        for msg in messages:
            node = self.add(msg)
            if node:
                created_nodes.append(node.node_id)

        if self._l1_buffer:
            node = self._compact_buffer(compaction_reason='re_summarize_flush')
            if node:
                self._session_node_ids.append(node.node_id)
                created_nodes.append(node.node_id)

        for nid in created_nodes:
            self._store.link_node_to_session(target_sid, nid)

        self._session_id = original_sid

        logger.info(f'Re-summarize: created {len(created_nodes)} nodes for session {target_sid}')
        return {
            'deleted': deleted_count,
            'created': len(created_nodes),
            'node_ids': created_nodes,
            'session_id': target_sid,
        }

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
        detailed = response.content.strip()
        node.detailed_summary = detailed
        self._store.save_node(node)
        logger.info(f'Generated detailed summary for {node_id} ({len(detailed)} chars)')
        return detailed

    def search(self, tag: str) -> list[str]:
        """Search nodes by topic tag."""
        return self._store.search_by_tag(tag)

    def get_rules(self) -> list[str]:
        """Get all extracted user rules."""
        return [r['rule'] for r in self._store.load_rules()]

    def delete_rule(self, rule_text: str) -> bool:
        """Delete a user rule."""
        return self._store.delete_rule(rule_text)

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
        """Get context window scoped to a specific session's nodes."""
        target_id = session_id or self._session_id
        session_nodes = self._store.list_by_session(target_id)
        seen_ids = {n['node_id'] for n in session_nodes}

        entries = []
        for n in session_nodes:
            entries.append((n.get('created_at', ''), n['node_id'], n, False))

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
                entries.append((pdict.get('created_at', ''), pid, pdict, True))

        entries.sort(key=lambda x: x[0])

        parts = []
        for _, nid, n, is_pinned in entries[:max_nodes]:
            summary = n.get('summary', '')
            trigger = n.get('trigger', '')
            recall = n.get('recall_mode', 'active')
            prefix = '(PIN) ' if is_pinned else ('(passive) ' if recall in ('passive', 'both') else '')
            tags = n.get('topic_tags', [])
            origin = ''
            for t in tags:
                if t.startswith('ORIGIN:'):
                    origin = f'({t}) '
                    break
            parts.append(f"[{nid}] {origin}{prefix}{summary} | {trigger}")

        if not parts:
            return f'(No nodes for session {target_id})'
        return '\n'.join(parts)

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
            """List all stored memory nodes with their IDs, summaries, and triggers."""
            return memo.get_context_window(max_nodes=50)

        @tool
        def read_memory_node(node_id: str) -> str:
            """Read the full raw content of a specific memory node. node_id starts with 'mem_'."""
            result = memo.read_memory(node_id, depth=2)
            return result if result else f"Node {node_id} not found"

        @tool
        def search_memory(tag: str) -> str:
            """Search memory nodes by topic tag."""
            results = memo.search(tag)
            if not results:
                return f"No nodes found with tag: {tag}"
            return '\n'.join(results)

        tools = [get_memory_index, read_memory_node, search_memory]

        if self._retriever:
            @tool
            def retrieve_memory(summary_query: str, trigger_query: str) -> str:
                """Semantic search across memory. Returns summaries and triggers only.

                Uses dual-path retrieval:
                - summary_query: Core keyword/topic of the information you need.
                - trigger_query: The situation/context that triggered this search.

                Both parameters are required.
                If the returned summaries are insufficient, use read_memory_node(node_id) for raw data.
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
                    return f'⚠️ HIGH RISK: This query involves specific values. Use read_memory_node to verify raw content.\n\n{body}'
                if analyzed.risk_level == 'low':
                    return f'✅ LOW RISK: Summary-level answer is sufficient.\n\n{body}'
                return body

            tools.append(retrieve_memory)

        return tools
