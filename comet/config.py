"""CoMeT Configuration using ato scope."""
from ato.scope import Scope
from ato.adict import ADict

scope = Scope(name='comet')


@scope.observe(default=True)
def default(comet: ADict):
    comet.slm_model = 'gpt-4o-mini'
    comet.main_model = 'gpt-4o'

    comet.llm = ADict(
        provider='openai',
    )

    comet.compacting = ADict(
        load_threshold=4,
        max_l1_buffer=10,
    )

    comet.storage = ADict(
        type='json',
        base_path='./memory_store',
        raw_path='./memory_store/raw',
    )

    comet.retrieval = ADict(
        embedding_model='text-embedding-3-small',
        vector_backend='chroma',
        vector_db_path='./memory_store/vectors',
        fusion_alpha=0.5,
        rrf_k=5,
        raw_search_weight=0.2,
        top_k=5,
        rerank=False,
    )


@scope.observe()
def local_slm(comet: ADict):
    """Use local SLM via Ollama."""
    comet.llm.provider = 'ollama'
    comet.llm.base_url = 'http://localhost:11434/v1'
    comet.slm_model = 'gemma2:9b'
    comet.retrieval.embedding_provider = 'ollama'
    comet.retrieval.embedding_model = 'nomic-embed-text'


@scope.observe()
def vllm(comet: ADict):
    """Use local vLLM server."""
    comet.llm.provider = 'vllm'
    comet.llm.base_url = 'http://localhost:8000/v1'


@scope.observe()
def anthropic(comet: ADict):
    """Use Anthropic Claude models."""
    comet.llm.provider = 'anthropic'
    comet.slm_model = 'claude-3-5-haiku-latest'
    comet.main_model = 'claude-3-5-sonnet-latest'


@scope.observe()
def google(comet: ADict):
    """Use Google Gemini models."""
    comet.llm.provider = 'google'
    comet.slm_model = 'gemini-2.0-flash'
    comet.main_model = 'gemini-2.0-flash'


@scope.observe()
def aggressive(comet: ADict):
    """More aggressive compacting."""
    comet.compacting.load_threshold = 3
    comet.compacting.max_l1_buffer = 5

