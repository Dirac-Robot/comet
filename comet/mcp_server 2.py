"""CoMeT MCP Server: Expose memory tools via Model Context Protocol."""
import os
import uuid
from typing import Optional

from fastmcp import FastMCP

from ato.adict import ADict
from comet.config import scope
from comet.orchestrator import CoMeT

mcp = FastMCP(
    'CoMeT',
    instructions=(
        'CoMeT is a hierarchical cognitive memory system.\n\n'
        'READING: Use retrieve_memory to search, then read_memory_node for details.\n\n'
        'WRITING: Call create_session to get a session_id first. '
        'Then use add_message(session_id, content) to ingest content. '
        'Call close_session(session_id) when done to consolidate. '
        'Each session is isolated — you can only write to your own session.'
    ),
)

_reader: Optional[CoMeT] = None
_sessions: dict[str, CoMeT] = {}


def _build_config() -> ADict:
    config = ADict()
    scope._apply_defaults(config)
    store_path = os.environ.get('COMET_STORE_PATH', './memory_store')
    config.storage.base_path = store_path
    config.storage.raw_path = os.path.join(store_path, 'raw')
    config.retrieval.vector_db_path = os.path.join(store_path, 'vectors')
    return config


def _get_reader() -> CoMeT:
    global _reader
    if _reader is not None:
        return _reader
    _reader = CoMeT(_build_config())
    return _reader


def _get_session(session_id: str) -> CoMeT:
    if session_id not in _sessions:
        raise ValueError(
            f'Session {session_id!r} not found. '
            f'Call create_session() first to get a valid session_id.'
        )
    return _sessions[session_id]


# --- Read Tools (shared, no session required) ---

@mcp.tool()
def get_memory_index() -> str:
    """List all stored memory nodes. Returns each node's ID, summary, and trigger."""
    return _get_reader().get_context_window(max_nodes=50)


@mcp.tool()
def read_memory_node(node_id: str) -> str:
    """Read detailed raw content from a specific memory node. node_id starts with 'mem_'."""
    result = _get_reader().read_memory(node_id, depth=2)
    return result if result else f'Node {node_id} not found'


@mcp.tool()
def search_memory(tag: str) -> str:
    """Search memory nodes by topic tag."""
    results = _get_reader().search(tag)
    if not results:
        return f'No nodes found with tag: {tag}'
    return '\n'.join(results)


@mcp.tool()
def retrieve_memory(summary_query: str, trigger_query: str) -> str:
    """Semantic search over memory using dual-path RAG.

    Args:
        summary_query: Keywords or topic describing WHAT information you need
        trigger_query: Context describing WHEN/WHY you need this information

    Returns summaries and triggers only.
    Call read_memory_node(node_id) for full raw content if needed.
    """
    memo = _get_reader()
    if memo._retriever is None:
        return 'Retrieval is not enabled (no vector index configured)'
    results = memo.retrieve_dual(summary_query, trigger_query)
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
    return '\n\n'.join(parts)


# --- Write Tools (session-scoped) ---

@mcp.tool()
def create_session() -> str:
    """Create a new memory session. Returns session_id for use in write tools.

    You MUST call this before using add_message or close_session.
    Each session is isolated — nodes created in a session are only
    consolidated when you call close_session with that session_id.
    """
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = CoMeT(_build_config(), session_id=session_id)
    return f'Session created: {session_id}'


@mcp.tool()
def get_session_context(session_id: str) -> str:
    """Get the context window for a specific session.

    Returns a structured index of all memory nodes accumulated in this session,
    including each node's ID, summary, trigger, and tags. This is CoMeT's core
    capability: providing an accurate key-map of the entire session memory so
    you always know what information is available and can decide whether to
    read the raw content via read_memory_node().

    Args:
        session_id: Session ID from create_session()
    """
    memo = _get_session(session_id)
    context = memo.get_context_window(max_nodes=50)
    node_count = len(memo._session_node_ids)
    buffer_size = len(memo.l1_buffer)
    header = f'Session {session_id}: {node_count} nodes, {buffer_size} buffered'
    if not context:
        return f'{header}\n(no compacted nodes yet)'
    return f'{header}\n{context}'


@mcp.tool()
def add_message(session_id: str, content: str) -> str:
    """Add a message to the memory system within a session.

    Args:
        session_id: Session ID from create_session()
        content: Text content to ingest into memory

    Returns compaction status. Content is buffered and automatically
    compacted into memory nodes when cognitive load triggers are detected.
    """
    memo = _get_session(session_id)
    node = memo.add(content)
    if node:
        return (
            f'Compacted to node: {node.node_id}\n'
            f'  Summary: {node.summary}\n'
            f'  Tags: {", ".join(node.topic_tags)}'
        )
    load = memo.last_load
    buffer_size = len(memo.l1_buffer)
    if load:
        return f'Buffered ({buffer_size} items, flow={load.logic_flow}, load={load.load_level})'
    return f'Buffered ({buffer_size} items)'


@mcp.tool()
def add_document(session_id: str, content: str, source: str = '') -> str:
    """Ingest a long document into memory within a session.

    Chunks the content automatically and processes through the L1 pipeline.

    Args:
        session_id: Session ID from create_session()
        content: Full document text
        source: Optional source identifier (URL, filename, etc.)
    """
    memo = _get_session(session_id)
    nodes = memo.add_document(content, source=source)
    if not nodes:
        return 'No nodes created (document may be empty or duplicate)'
    summaries = '\n'.join(
        f'  [{n.node_id}] {n.summary}' for n in nodes
    )
    return f'{len(nodes)} nodes created:\n{summaries}'


@mcp.tool()
def close_session(session_id: str) -> str:
    """Close a session: compact remaining buffer and consolidate session nodes.

    After closing, the session_id becomes invalid.
    Memories written in this session remain accessible via read tools.

    Args:
        session_id: Session ID to close
    """
    memo = _get_session(session_id)
    result = memo.close_session()
    del _sessions[session_id]

    global _reader
    _reader = None

    status = result.get('status', 'done')
    return f'Session {session_id} closed (status={status}). Reader cache refreshed.'


# --- Resources ---

@mcp.resource('memory://nodes')
def list_all_nodes() -> str:
    """All stored memory nodes with summaries and triggers."""
    memo = _get_reader()
    all_nodes = memo.list_memories()
    if not all_nodes:
        return 'No memory nodes stored.'
    parts = []
    for n in all_nodes:
        summary = n.get('summary', '')
        trigger = n.get('trigger', '')
        tags = ', '.join(n.get('topic_tags', []))
        parts.append(
            f'[{n["node_id"]}] {summary}\n'
            f'  Trigger: {trigger}\n'
            f'  Tags: {tags}'
        )
    return '\n\n'.join(parts)


@mcp.resource('memory://sessions')
def list_active_sessions() -> str:
    """List currently active write sessions."""
    if not _sessions:
        return 'No active sessions.'
    parts = []
    for sid, memo in _sessions.items():
        buffer_size = len(memo.l1_buffer)
        node_count = len(memo._session_node_ids)
        parts.append(f'[{sid}] buffer={buffer_size}, nodes={node_count}')
    return '\n'.join(parts)


def main():
    mcp.run()


if __name__ == '__main__':
    main()
