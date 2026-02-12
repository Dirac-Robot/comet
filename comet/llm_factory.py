"""LLM Factory: Provider-agnostic model creation for CoMeT components."""
from typing import Callable

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from loguru import logger


def create_chat_model(model_name: str, config: ADict) -> BaseChatModel:
    """Create a LangChain chat model based on provider prefix or config.

    Provider resolution order:
    1. Explicit prefix in model_name (e.g. 'ollama/gemma2:9b', 'anthropic/claude-...')
    2. config.llm.provider (e.g. 'openai', 'anthropic', 'ollama', 'vllm')
    3. Default to 'openai'

    Supported providers:
    - openai:    gpt-4o, gpt-4o-mini, etc. (via OPENAI_API_KEY)
    - anthropic: claude-3.5-sonnet, etc. (via ANTHROPIC_API_KEY)
    - google:    gemini-2.0-flash, etc. (via GOOGLE_API_KEY)
    - ollama:    local models via Ollama (http://localhost:11434)
    - vllm:      local vLLM server (OpenAI-compatible endpoint)
    """
    provider, resolved_name = _resolve_provider(model_name, config)
    kwargs = _build_kwargs(provider, resolved_name, config)

    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)

    if provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**kwargs)

    if provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**kwargs)

    if provider in ('ollama', 'vllm'):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)

    raise ValueError(f'Unsupported LLM provider: {provider}')


def create_embeddings(config: ADict) -> Callable[[list[str]], list[list[float]]]:
    """Create an embedding function based on config.

    Returns a callable: list[str] -> list[list[float]]

    Supported providers:
    - openai (default): text-embedding-3-small, etc.
    - ollama: local embedding models
    - vllm: vLLM embedding endpoint
    """
    provider = config.retrieval.get('embedding_provider') or config.get('llm', {}).get('provider', 'openai')
    model = config.retrieval.embedding_model

    if provider == 'openai':
        from openai import OpenAI
        base_url = config.retrieval.get('embedding_base_url')
        client = OpenAI(base_url=base_url) if base_url else OpenAI()

        def embed_openai(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]

        logger.debug(f'Embedding provider: openai ({model})')
        return embed_openai

    if provider == 'ollama':
        from openai import OpenAI
        base_url = config.retrieval.get('embedding_base_url', 'http://localhost:11434/v1')
        client = OpenAI(base_url=base_url, api_key='ollama')

        def embed_ollama(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]

        logger.debug(f'Embedding provider: ollama ({model})')
        return embed_ollama

    if provider == 'vllm':
        from openai import OpenAI
        base_url = config.retrieval.get('embedding_base_url', config.get('llm', {}).get('base_url', 'http://localhost:8000/v1'))
        client = OpenAI(base_url=base_url, api_key='vllm')

        def embed_vllm(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]

        logger.debug(f'Embedding provider: vllm ({model})')
        return embed_vllm

    raise ValueError(f'Unsupported embedding provider: {provider}')


def _resolve_provider(model_name: str, config: ADict) -> tuple[str, str]:
    """Parse 'provider/model' prefix or fall back to config.llm.provider."""
    known_prefixes = {'openai', 'anthropic', 'google', 'ollama', 'vllm'}
    if '/' in model_name:
        prefix, rest = model_name.split('/', 1)
        if prefix.lower() in known_prefixes:
            return prefix.lower(), rest
    provider = config.get('llm', {}).get('provider', 'openai')
    return provider, model_name


def _build_kwargs(provider: str, model_name: str, config: ADict) -> dict:
    """Build constructor kwargs per provider."""
    llm_config = config.get('llm', {})

    if provider == 'openai':
        kwargs = {'model': model_name}
        if llm_config.get('base_url'):
            kwargs['base_url'] = llm_config['base_url']
        if llm_config.get('api_key'):
            kwargs['api_key'] = llm_config['api_key']
        return kwargs

    if provider == 'anthropic':
        kwargs = {'model': model_name}
        if llm_config.get('api_key'):
            kwargs['anthropic_api_key'] = llm_config['api_key']
        return kwargs

    if provider == 'google':
        kwargs = {'model': model_name}
        if llm_config.get('api_key'):
            kwargs['google_api_key'] = llm_config['api_key']
        return kwargs

    if provider == 'ollama':
        base_url = llm_config.get('base_url', 'http://localhost:11434/v1')
        return {
            'model': model_name,
            'base_url': base_url,
            'api_key': 'ollama',
        }

    if provider == 'vllm':
        base_url = llm_config.get('base_url', 'http://localhost:8000/v1')
        return {
            'model': model_name,
            'base_url': base_url,
            'api_key': 'vllm',
        }

    return {'model': model_name}
