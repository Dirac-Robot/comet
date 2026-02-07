"""Template loader for CoMeT prompts."""
from pathlib import Path
from functools import lru_cache


TEMPLATE_DIR = Path(__file__).parent


@lru_cache(maxsize=10)
def load_template(name: str) -> str:
    """
    Load a prompt template by name.
    
    Args:
        name: Template name without extension (e.g., 'cognitive_load')
    
    Returns:
        Template string with placeholders (e.g., {content}, {turns})
    """
    template_path = TEMPLATE_DIR / f'{name}.txt'
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text(encoding='utf-8')


def get_template(name: str, **kwargs) -> str:
    """
    Load and format a template with provided values.
    
    Args:
        name: Template name without extension
        **kwargs: Format arguments for the template
    
    Returns:
        Formatted prompt string
    """
    template = load_template(name)
    return template.format(**kwargs)
