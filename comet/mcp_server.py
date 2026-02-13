"""CoMeT MCP Server: Read-oriented memory access for external agents."""
import json
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from comet.config import scope
from comet.orchestrator import CoMeT

load_dotenv()

mcp = FastMCP(
    'CoMeT Memory',
    instructions='Cognitive Memory Tree — read-only access to hierarchical agent memory.',
)

_comet_instance: CoMeT | None = None


def _get_comet() -> CoMeT:
    global _comet_instance
    if _comet_instance is None:
        store_path = os.environ.get('COMET_STORE_PATH', './memory_store')
        config = scope.build_config()
        config.storage.base_path = store_path
        config.storage.raw_path = os.path.join(store_path, 'raw')
        config.retrieval.vector_db_path = os.path.join(store_path, 'vectors')
        _comet_instance = CoMeT(config)
    return _comet_instance


@mcp.resource('memory://nodes')
def list_nodes() -> str:
    """List all stored memory nodes with summaries and triggers."""
    comet = _get_comet()
    return comet.get_context_window(max_nodes=100)


@mcp.resource('memory://sessions')
def list_sessions() -> str:
    """List all registered memory sessions with metadata."""
    comet = _get_comet()
    sessions = comet.list_sessions()
    if not sessions:
        return 'No sessions found.'
    return json.dumps(sessions, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
def get_memory_index() -> str:
    """List all memory nodes with their IDs, summaries, and triggers.

    Use this first to discover what memories are available.
    """
    comet = _get_comet()
    return comet.get_context_window(max_nodes=50)


@mcp.tool()
def read_memory_node(node_id: str) -> str:
    """Read the full raw content of a specific memory node.

    Args:
        node_id: The node ID (starts with 'mem_').

    Returns the node's raw data including topics, links, and full content.
    """
    comet = _get_comet()
    result = comet.read_memory(node_id, depth=2)
    return result if result else f'Node {node_id} not found'


@mcp.tool()
def search_memory(tag: str) -> str:
    """Search memory nodes by topic tag.

    Args:
        tag: Topic tag to search for (e.g., 'debugging', 'architecture').
    """
    comet = _get_comet()
    results = comet.search(tag)
    if not results:
        return f'No nodes found with tag: {tag}'
    return '\n'.join(results)


@mcp.tool()
def retrieve_memory(summary_query: str, trigger_query: str) -> str:
    """Semantic search across memory using dual-path retrieval (summary + trigger).

    Uses QueryAnalyzer → triple-path vector search → RRF score fusion.

    Args:
        summary_query: What information you're looking for (topic/keyword).
        trigger_query: The situation that prompted this search (context/intent).

    Returns matching nodes with scores. Use read_memory_node() for full content.
    """
    comet = _get_comet()
    results = comet.retrieve_dual(summary_query, trigger_query)
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


if __name__ == '__main__':
    mcp.run()
