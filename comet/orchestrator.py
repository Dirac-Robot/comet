"""CoMeT Orchestrator: Main workflow coordinating Sensor, Compacter, and Store."""
from typing import Optional, Callable

from ato.adict import ADict
from langchain_core.tools import tool, BaseTool
from loguru import logger

from comet.schemas import L1Memory, MemoryNode, CognitiveLoad
from comet.sensor import CognitiveSensor
from comet.compacter import MemoryCompacter
from comet.storage import MemoryStore


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
        self._sensor = CognitiveSensor(config)
        self._compacter = MemoryCompacter(config, self._store)
        self._l1_buffer: list[L1Memory] = []
        self._last_load: Optional[CognitiveLoad] = None

    @property
    def l1_buffer(self) -> list[L1Memory]:
        return self._l1_buffer

    @property
    def last_load(self) -> Optional[CognitiveLoad]:
        return self._last_load

    def add(self, content: str) -> Optional[MemoryNode]:
        """
        Add new content to the memory system.
        
        Returns MemoryNode if compacting was triggered, else None.
        """
        # L1 extraction
        l1_mem = self._sensor.extract_l1(content)
        
        # Assess cognitive load
        load = self._sensor.assess_load(content, self._l1_buffer)
        self._last_load = load
        
        logger.debug(f"CogLoad: {load.logic_flow}, level={load.load_level}")
        
        # Check if compacting needed (only if buffer is not empty)
        if self._l1_buffer and self._sensor.should_compact(load, len(self._l1_buffer)):
            # Compact current buffer before adding new
            node = self._compact_buffer()
            
            # Clear buffer and add new content
            self._l1_buffer = [l1_mem]
            
            logger.info(f"Compacted to node: {node.node_id}")
            return node
        
        # Just add to L1 buffer
        self._l1_buffer.append(l1_mem)
        return None

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
        """Get context window: summary + trigger for each node."""
        parts = []
        
        all_nodes = self._store.list_all()
        recent = sorted(all_nodes, key=lambda x: x.get('created_at', ''), reverse=True)[:max_nodes]
        
        if recent:
            for n in recent:
                summary = n.get('summary', '')
                trigger = n.get('trigger', '')
                parts.append(f"[{n['node_id']}] {summary} | {trigger}")
        
        if self._l1_buffer:
            for mem in self._l1_buffer[-2:]:
                parts.append(f"(recent) {mem.content[:40]}...")
        
        return '\n'.join(parts)

    def get_tools(self) -> list[BaseTool]:
        """
        Get LangChain-compatible tools for memory operations.
        
        Returns tools that can be used with any LangChain agent:
        - get_memory_index: List all memory nodes with triggers
        - read_memory_node: Read detailed content from a specific node
        - search_memory: Search nodes by topic tag
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
        
        return [get_memory_index, read_memory_node, search_memory]
