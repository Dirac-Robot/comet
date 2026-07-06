"""Compaction post-save atomicity — the duplicate-node regression.

2026-07-06 incident: with a quota-dead embedding key, ``_auto_link`` raised
AFTER ``save_node`` — the node existed on disk but the exception escaped
``compact()``/``add()``, so the L1 buffer was never reset and every later
``add()`` re-compacted the same window into a near-duplicate node (dozens
per session), while callers logged the compaction as failed.

Contract pinned here: once the node is persisted, ``compact()`` /
``_compact_buffer()`` / ``add()`` complete — return the node, reset the
buffer — regardless of enrichment failures (auto-link embeddings, vector
upsert, tag decoration, pending-link stitching).

Run: pytest tests/test_compaction_atomicity.py -q
"""
import threading
from types import SimpleNamespace

import pytest
from ato.adict import ADict

from comet.compacter import MemoryCompacter
from comet.orchestrator import CoMeT
from comet.schemas import L1Memory
from comet.storage import MemoryStore


def _config(tmp_path):
    return ADict(
        language='Korean',
        main_model='stub-model',
        slm_model='stub-model',
        llm=ADict(provider='openai'),
        compacting=ADict(load_threshold=4, max_l1_buffer=10, min_l1_buffer=3),
        storage=ADict(
            type='json',
            base_path=str(tmp_path / 'store'),
            raw_path=str(tmp_path / 'store' / 'raw'),
        ),
        consolidation=ADict(
            min_tag_overlap=2,
            cross_link_threshold=0.45,
            cross_session_min_tag_overlap=1,
            cross_session_link_threshold=0.40,
        ),
    )


def _stub_result():
    return SimpleNamespace(
        summary='요약',
        trigger='다시 필요할 때',
        topic_tags=['TRPG', 'character-creation'],
        recall_mode='active',
        importance='MED',
        flags=[],
        session_brief='',
    )


class _StubStructuredLLM:
    def invoke(self, prompt):
        return _stub_result()


class _QuotaDeadVectorIndex:
    """Embedding provider is down: every call raises (the 429 shape)."""

    def search_by_summary(self, query, top_k=10):
        raise RuntimeError('Error code: 429 - insufficient_quota')

    def upsert(self, node, raw_content=None):
        raise RuntimeError('Error code: 429 - insufficient_quota')


def _make_compacter(tmp_path):
    config = _config(tmp_path)
    store = MemoryStore(config)
    compacter = MemoryCompacter(config, store, vector_index=_QuotaDeadVectorIndex())
    compacter._llm = object()                     # _ensure_llm() no-op gate
    compacter._structured_llm = _StubStructuredLLM()
    return compacter, store


def _l1(text):
    return L1Memory(content=text, raw_content=text)


def test_compact_survives_dead_embeddings_and_persists(tmp_path):
    compacter, store = _make_compacter(tmp_path)
    # A pre-existing cross-session node with overlapping tags forces
    # _auto_link past the tag-overlap gate into the embedding call.
    seed = compacter.compact([_l1('[user] 시드')], session_id='other')
    node = compacter.compact(
        [_l1('[user] 캐릭터 만들어줘'), _l1('[assistant] 엘리안 모르크')],
        session_id='sess-a',
    )
    assert node is not None
    assert store.get_node(node.node_id) is not None
    assert node.node_id != seed.node_id


def _make_comet(tmp_path, buffer_texts):
    """Bare CoMeT wired for add(): stub sensor forces a high_load compaction."""
    config = _config(tmp_path)
    store = MemoryStore(config)
    compacter = MemoryCompacter(config, store, vector_index=_QuotaDeadVectorIndex())
    compacter._llm = object()
    compacter._structured_llm = _StubStructuredLLM()

    load = SimpleNamespace(load_level=5, redundancy_detected=False,
                           logic_flow='MAINTAIN', reasoning='')
    sensor = SimpleNamespace(
        extract_l1=lambda text: _l1(text),
        assess_load=lambda text, buf, summaries: load,
        get_compaction_reason=lambda ld, size: 'high_load',
    )

    comet = object.__new__(CoMeT)
    comet._config = config
    comet._store = store
    comet._vector_index = None
    comet._compacter = compacter
    comet._sensor = sensor
    comet._session_id = 'sess-a'
    comet._session_node_ids = []
    comet._l1_buffer = [_l1(t) for t in buffer_texts]
    comet._buffer_origin = 'USER'
    comet._buffer_extra_tags = set()
    comet._buffer_active_skills = set()
    comet._pending_external_links = []
    comet._pending_read_links = []
    comet._pinned_node_ids = set()
    comet._last_load = None
    comet._lock = threading.Lock()
    return comet, store


def test_add_resets_buffer_when_enrichment_fails(tmp_path):
    """The incident shape: compaction triggered while embeddings are dead.
    The node must be created AND the buffer reset — a second add() must not
    re-compact the same window."""
    comet, store = _make_comet(
        tmp_path, ['[user] 오프닝 브리프', '[assistant] 준비 완료'],
    )
    node = comet.add('[user] 캐릭터 콘셉트 제안해줘')
    assert node is not None
    assert store.get_node(node.node_id) is not None
    # Buffer reset to just the triggering message — the old window is gone.
    assert len(comet._l1_buffer) == 1
    assert '캐릭터 콘셉트' in comet._l1_buffer[0].content

    # Second add compacts ONLY the new window (no re-compaction of the old
    # items): its raw must not contain the opening brief again.
    node2 = comet.add('[user] 다음 장면으로 가자')
    assert node2 is not None
    raw2 = open(node2.raw_location, encoding='utf-8').read()
    assert '오프닝 브리프' not in raw2


def test_pending_link_stitching_failure_does_not_escape(tmp_path):
    comet, store = _make_comet(tmp_path, ['[user] a', '[assistant] b'])
    comet._pending_external_links = ['mem_missing_node']

    def boom(a, b):
        raise RuntimeError('link target gone')

    comet._compacter.link_nodes = boom
    node = comet.add('[user] c')
    assert node is not None
    assert store.get_node(node.node_id) is not None
    assert len(comet._l1_buffer) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
