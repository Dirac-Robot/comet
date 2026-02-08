"""CoMeT Configuration using ato scope."""
from ato.scope import Scope
from ato.adict import ADict

scope = Scope()


@scope.observe(default=True)
def default(config: ADict):
    # SLM for Fast Layer (L1)
    config.slm_model = 'gpt-4o-mini'
    
    # Main LLM for Slow Layer (L2+)
    config.main_model = 'gpt-4o'
    
    # Compacting thresholds
    config.compacting = ADict(
        load_threshold=4,  # load_level >= 4 triggers compacting
        max_l1_buffer=10,  # Max items before forced compacting
    )
    
    # Storage settings
    config.storage = ADict(
        type='json',  # 'json' or 'redis'
        base_path='./memory_store',
        raw_path='./memory_store/raw',
    )

    # RAG retrieval settings
    config.retrieval = ADict(
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
def local_slm(config: ADict):
    """Use local SLM via Ollama."""
    config.slm_model = 'ollama/gemma2:9b'
    config.slm_base_url = 'http://localhost:11434/v1'


@scope.observe()
def aggressive(config: ADict):
    """More aggressive compacting."""
    config.compacting.load_threshold = 3
    config.compacting.max_l1_buffer = 5
