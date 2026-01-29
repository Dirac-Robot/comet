"""Web search utilities using DuckDuckGo (free, no API key required)."""
from ddgs import DDGS


def search_web(query: str, max_results: int = 10) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, href, and body
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return [{'title': 'Search Error', 'href': '', 'body': str(e)}]


def format_search_results(results: list[dict]) -> str:
    """Format search results into a readable string."""
    if not results:
        return 'No results found.'
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] {r.get('title', 'No title')}\n    {r.get('body', '')}")
    return '\n\n'.join(formatted)
