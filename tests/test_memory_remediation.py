"""P0-B language adherence + P1-C embedding rehydration + FLAG:REVISES.

Covers the three structural fixes from the 2026-07-16 memory remediation
plan: the compaction {language} directive must never carry the
self-referential phrase into the prompt, embedding-upsert failures must be
durably queued and self-heal on rehydrate, and the compactor-judged
REVISES flag must exist on both enum surfaces.
"""
import shutil
import tempfile
from types import SimpleNamespace

from ato.adict import ADict

from comet.compacter import MemoryCompacter, _resolve_compaction_language
from comet.flags import CompactorJudgedFlag, KindFlag
from comet.orchestrator import CoMeT
from comet.schemas import MemoryNode
from comet.storage import MemoryStore


def _store(tmpdir: str) -> MemoryStore:
    return MemoryStore(ADict(storage=ADict(
        base_path=f'{tmpdir}/store', raw_path=f'{tmpdir}/raw',
    )))


def _node(nid: str) -> MemoryNode:
    return MemoryNode(
        node_id=nid, summary=f'summary {nid}',
        content_key=f'raw_{nid}', raw_location=f'raw_{nid}.txt',
    )


# ── P0-B: language directive resolution ──

def test_latin_content_never_gets_self_referential_phrase():
    resolved = _resolve_compaction_language(
        'the same language as the user', 'plain english planning discussion',
    )
    assert 'same language' not in resolved.lower()
    assert 'conversation content' in resolved


def test_concrete_operator_language_is_honored():
    assert _resolve_compaction_language('Korean', 'whatever content') == 'Korean'


def test_cjk_content_still_anchors_to_script():
    korean = '스키마 개정 결정 롤아웃 시작 예외 만료 단계 검토 승인 완료'
    assert _resolve_compaction_language('the same language as the user', korean) == 'Korean'


# ── P1-C: embedding-pending ledger + rehydrate ──

def test_pending_ledger_roundtrip_and_persistence():
    tmpdir = tempfile.mkdtemp()
    try:
        store = _store(tmpdir)
        store.mark_embedding_pending('n1')
        store.mark_embedding_pending('n2')
        store.mark_embedding_pending('n1')  # idempotent
        assert store.list_embedding_pending() == ['n1', 'n2']
        store.clear_embedding_pending('n1')
        assert store.list_embedding_pending() == ['n2']

        # Survives a store reopen (durable, not in-memory).
        store.close()
        store2 = _store(tmpdir)
        assert store2.list_embedding_pending() == ['n2']
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_compacter_mark_helper_never_raises():
    boom = SimpleNamespace(mark_embedding_pending=lambda nid: (_ for _ in ()).throw(RuntimeError('x')))
    fake = SimpleNamespace(_store=boom)
    MemoryCompacter._mark_embedding_pending(fake, 'n1')  # must not raise


def test_rehydrate_reembeds_and_clears():
    tmpdir = tempfile.mkdtemp()
    try:
        store = _store(tmpdir)
        node = _node('n1')
        store.save_raw('raw_n1', 'raw content for n1')
        store.save_node(node)
        store.mark_embedding_pending('n1')
        store.mark_embedding_pending('ghost')  # node file gone → dropped

        upserted = []
        vi = SimpleNamespace(upsert=lambda n, raw_content='': upserted.append((n.node_id, raw_content)))
        fake = SimpleNamespace(_vector_index=vi, _store=store)

        out = CoMeT.rehydrate_pending_embeddings(fake)
        assert out['rehydrated'] == 1 and out['dropped'] == 1 and out['failed'] == 0
        assert upserted and upserted[0][0] == 'n1' and 'raw content' in upserted[0][1]
        assert store.list_embedding_pending() == []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rehydrate_stops_on_provider_error_and_keeps_pending():
    tmpdir = tempfile.mkdtemp()
    try:
        store = _store(tmpdir)
        for nid in ('n1', 'n2'):
            store.save_node(_node(nid))
            store.mark_embedding_pending(nid)

        def _fail(n, raw_content=''):
            raise RuntimeError('429 quota')

        fake = SimpleNamespace(_vector_index=SimpleNamespace(upsert=_fail), _store=store)
        out = CoMeT.rehydrate_pending_embeddings(fake)
        assert out['failed'] == 1 and out['rehydrated'] == 0
        # Early stop: second node was never attempted, both stay pending.
        assert store.list_embedding_pending() == ['n1', 'n2']
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── FLAG:REVISES on both enum surfaces ──

def test_revises_flag_exists_on_both_enums():
    assert KindFlag.REVISES.value == 'FLAG:REVISES'
    assert CompactorJudgedFlag.REVISES.value == 'FLAG:REVISES'


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
