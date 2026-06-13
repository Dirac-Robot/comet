"""LLM Factory: Provider-agnostic model creation for CoMeT components."""
from typing import Callable

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from loguru import logger


def create_chat_model(
    model_name: str, config: ADict, llm_config: ADict | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model based on provider prefix or config.

    ``llm_config`` carries the provider kwargs (provider, api_key, base_url,
    headers, …) for THIS model. When omitted it falls back to ``config.llm``
    — the historical single shared block. Callers whose model differs from
    the shared one (e.g. the compacter's ``main_model`` vs the sensor's
    ``slm_model`` on another provider) must pass their own block, otherwise
    the model name gets instantiated against another model's provider.

    Provider resolution order:
    1. Explicit prefix in model_name (e.g. 'ollama/gemma2:9b', 'anthropic/claude-...')
    2. llm_config.provider (or config.llm.provider when llm_config is None)
    3. Default to 'openai'

    Supported providers:
    - openai:        gpt-5.4, gpt-5.4-mini, etc. (via OPENAI_API_KEY)
    - openai_oauth:  same OpenAI models routed through Codex/ChatGPT OAuth's
                     Responses API backend — billed against the caller's
                     ChatGPT plan instead of the per-token API. Caller is
                     expected to populate config.llm with:
                       api_key            : OAuth access token (Bearer)
                       base_url           : Codex Responses base URL
                       default_headers    : ChatGPT-Account-ID + version
                       use_responses_api  : True
                       store              : False (Codex rejects store)
                       chat_class         : optional ChatOpenAI subclass that
                                            rewrites the request payload for
                                            the Codex backend's `instructions`
                                            convention. Falls back to vanilla
                                            ChatOpenAI when absent.
    - anthropic: claude-opus-4-6, claude-sonnet-4-6, etc. (via ANTHROPIC_API_KEY)
    - claude_code_oauth: Claude Code CLI OAuth/subscription login via `claude -p`
    - google:    gemini-3-flash-preview, gemini-3.1-pro-preview, etc. (via GOOGLE_API_KEY)
    - ollama:    local models via Ollama (http://localhost:11434)
    - vllm:      local vLLM server (OpenAI-compatible endpoint)
    """
    if llm_config is None:
        llm_config = config.get('llm', {}) or {}
    provider, resolved_name = _resolve_provider(model_name, llm_config)
    kwargs = _build_kwargs(provider, resolved_name, llm_config)

    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)

    if provider == 'openai_oauth':
        # The OAuth-injected chat class (when supplied by the caller) carries
        # the Codex-specific request payload override (system → instructions).
        # Without it, vanilla ChatOpenAI still works against most Responses
        # API surfaces but the Codex backend will reject calls that don't
        # surface the system prompt as `instructions`.
        chat_class = llm_config.get('chat_class')
        if chat_class is None:
            from langchain_openai import ChatOpenAI
            chat_class = ChatOpenAI
        return chat_class(**kwargs)

    if provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**kwargs)

    if provider == 'claude_code_oauth':
        from comet.claude_code_oauth import ClaudeCodeOAuthChatModel
        return ClaudeCodeOAuthChatModel(**kwargs)

    if provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**kwargs)

    if provider in ('ollama', 'vllm'):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)

    raise ValueError(f'Unsupported LLM provider: {provider}')


def structured_output_kwargs(llm_config: ADict | None) -> dict:
    """kwargs for ``.with_structured_output()`` appropriate to the provider.

    The Codex Responses backend (openai_oauth) requires streaming, and its
    streamed json_schema path never populates the ``parsed`` field — the
    default method then raises at parse time. Function calling parses from
    tool_call args and streams fine (it is how the host agent loop runs on
    the same backend). Other providers keep langchain's default method.
    """
    provider = (llm_config or {}).get('provider')
    if provider == 'openai_oauth':
        return {'method': 'function_calling'}
    return {}


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
        base_url = config.retrieval.get('embedding_base_url')
        _client_cache: list = []  # lazy init — API key may not be set yet at import time

        def _get_client():
            if not _client_cache:
                from openai import OpenAI
                _client_cache.append(OpenAI(base_url=base_url) if base_url else OpenAI())
            return _client_cache[0]

        def embed_openai(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            response = _get_client().embeddings.create(model=model, input=texts)
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


MODEL_ALIASES: dict[str, str] = {}


def _resolve_alias(model_name: str) -> str:
    return model_name


def _resolve_provider(model_name: str, llm_config: ADict) -> tuple[str, str]:
    """Parse 'provider/model' prefix or fall back to llm_config.provider."""
    model_name = _resolve_alias(model_name)
    if model_name.startswith('oauth:'):
        bare = model_name[len('oauth:'):]
        provider = 'claude_code_oauth' if bare.startswith('claude-') else 'openai_oauth'
        return provider, bare
    known_prefixes = {
        'openai',
        'openai_oauth',
        'anthropic',
        'claude_code_oauth',
        'google',
        'ollama',
        'vllm',
    }
    if '/' in model_name:
        prefix, rest = model_name.split('/', 1)
        if prefix.lower() in known_prefixes:
            return prefix.lower(), _resolve_alias(rest)
    provider = llm_config.get('provider', 'openai')
    return provider, model_name


def _build_kwargs(provider: str, model_name: str, llm_config: ADict) -> dict:
    """Build constructor kwargs per provider."""
    if provider == 'openai':
        kwargs = {'model': model_name}
        if llm_config.get('base_url'):
            kwargs['base_url'] = llm_config['base_url']
        if llm_config.get('api_key'):
            kwargs['api_key'] = llm_config['api_key']
        return kwargs

    if provider == 'openai_oauth':
        # ChatOpenAI accepts these flags for the OAuth/Responses backend.
        # use_responses_api flips the wire format from Chat Completions to
        # /v1/responses; store=False keeps Codex from rejecting the call;
        # default_headers carries the ChatGPT account id + CLI version that
        # the Codex backend uses for plan-side bookkeeping.
        kwargs = {'model': model_name}
        if llm_config.get('base_url'):
            kwargs['base_url'] = llm_config['base_url']
        if llm_config.get('api_key'):
            kwargs['api_key'] = llm_config['api_key']
        if llm_config.get('default_headers'):
            kwargs['default_headers'] = llm_config['default_headers']
        if llm_config.get('use_responses_api') is not None:
            kwargs['use_responses_api'] = llm_config['use_responses_api']
        else:
            kwargs['use_responses_api'] = True
        if llm_config.get('store') is not None:
            kwargs['store'] = llm_config['store']
        else:
            kwargs['store'] = False
        if llm_config.get('streaming') is not None:
            kwargs['streaming'] = llm_config['streaming']
        else:
            # The Codex Responses backend rejects non-streamed calls
            # ("Stream must be set to true") — stream-and-aggregate even
            # for plain .invoke() callers like the compacter.
            kwargs['streaming'] = True
        return kwargs

    if provider == 'anthropic':
        kwargs = {'model': model_name}
        if llm_config.get('api_key'):
            kwargs['anthropic_api_key'] = llm_config['api_key']
        return kwargs

    if provider == 'claude_code_oauth':
        kwargs = {'model': model_name}
        if llm_config.get('claude_bin'):
            kwargs['claude_bin'] = llm_config['claude_bin']
        if llm_config.get('timeout'):
            kwargs['timeout'] = llm_config['timeout']
        if llm_config.get('cwd'):
            kwargs['cwd'] = llm_config['cwd']
        if llm_config.get('effort'):
            kwargs['effort'] = llm_config['effort']
        return kwargs

    if provider == 'google':
        kwargs = {'model': model_name}
        if llm_config.get('api_key'):
            kwargs['google_api_key'] = llm_config['api_key']
        if llm_config.get('safety_settings'):
            kwargs['safety_settings'] = llm_config['safety_settings']
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
