"""CoMeT Orchestrator: Main workflow coordinating Sensor, Compacter, and Store."""
from typing import Optional, Union

from ato.adict import ADict
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from loguru import logger

from comet.schemas import L1Memory, MemoryNode, CognitiveLoad, RetrievalResult
from comet.sensor import CognitiveSensor
from comet.compacter import MemoryCompacter
from comet.storage import MemoryStore
from comet.vector_index import VectorIndex
from comet.retriever import Retriever

MessageInput = Union[str, dict, list, BaseMessage]


class CoMeT:
    """
    Cognitive Memory OS - Dynamic Resolution Memory System.

    Main orchestrator that:
    1. Receives input through add()
    2. SLM sensor extracts L1 and assesses cognitive load
    3. Triggers compacting when load/shift detected
    4. Provides read_memory(key, depth) for navigation
    """

    def __init__(self, config: ADict):
        self._config = config
        self._store = MemoryStore(config)
        self._vector_index = VectorIndex(config) if config.get('retrieval') else None
        self._sensor = CognitiveSensor(config)
        self._compacter = MemoryCompacter(config, self._store, self._vector_index)
        self._retriever = Retriever(config, self._store, self._vector_index) if self._vector_index else None
        self._l1_buffer: list[L1Memory] = []
        self._last_load: Optional[CognitiveLoad] = None

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

        if self._l1_buffer and self._sensor.should_compact(load, len(self._l1_buffer)):
            node = self._compact_buffer()
            self._l1_buffer = [l1_mem]
            logger.info(f"Compacted to node: {node.node_id}")
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

    def _compact_buffer(self) -> MemoryNode:
        """Compact current L1 buffer into a MemoryNode."""
        if not self._l1_buffer:
            raise ValueError("Cannot compact empty buffer")
        
        return self._compacter.compact(self._l1_buffer)

    def force_compact(self) -> Optional[MemoryNode]:
        """Force compacting of current buffer."""
        if not self._l1_buffer:
            return None
        
        node = self._compact_buffer()
        self._l1_buffer = []
        return node

    def read_memory(self, node_id: str, depth: int = 0) -> Optional[str]:
        """
        Navigation Tool: Read memory at specified depth.
        
        depth=0: Summary only
        depth=1: Summary + metadata
        depth=2: Full raw data
        """
        return self._store.read_memory(node_id, depth)

    def search(self, tag: str) -> list[str]:
        """Search nodes by topic tag."""
        return self._store.search_by_tag(tag)

    def list_memories(self) -> list[dict]:
        """List all stored memory nodes."""
        return self._store.list_all()

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
        - retrieve_memory: Semantic RAG search (if retrieval enabled)
        """
        memo = self  # Capture self for closure
        
        @tool
        def get_memory_index() -> str:
            """저장된 메모리 노드의 인덱스를 조회합니다. 각 노드의 ID, 요약, trigger가 포함됩니다."""
            return memo.get_context_window(max_nodes=50)
        
        @tool
        def read_memory_node(node_id: str) -> str:
            """특정 메모리 노드의 상세 내용을 조회합니다. node_id는 mem_으로 시작합니다."""
            result = memo.read_memory(node_id, depth=2)
            return result if result else f"Node {node_id} not found"
        
        @tool
        def search_memory(tag: str) -> str:
            """주제 태그로 메모리 노드를 검색합니다."""
            results = memo.search(tag)
            if not results:
                return f"No nodes found with tag: {tag}"
            return '\n'.join(results)

        tools = [get_memory_index, read_memory_node, search_memory]

        if self._retriever:
            @tool
            def retrieve_memory(query: str) -> str:
                """자연어 쿼리로 관련 메모리를 의미 기반 검색합니다. 가장 관련성 높은 노드들이 반환됩니다."""
                results = memo.retrieve(query)
                if not results:
                    return 'No relevant memories found'
                parts = []
                for r in results:
                    parts.append(
                        f"[{r.node.node_id}] (score={r.relevance_score:.4f})\n"
                        f"  Summary: {r.node.summary}\n"
                        f"  Trigger: {r.node.trigger}"
                    )
                return '\n\n'.join(parts)

            tools.append(retrieve_memory)

        return tools
