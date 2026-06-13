"""Persistent recall journal — measurable recall frequency + trigger-channel share.

node.recall_count was near-zero and unusable for analysis: record_recall_hits
only bumps an in-memory buffer that the dream reinforced-decay pass drains, and
a restart drops it. retrieve_dual now also emits one structured `memory.recall`
log line per retrieval (independent of that lossy buffer), carrying the recalled
node ids and which of them matched via the trigger channel — so a question like
"did the instruction-form trigger rewrite change recall frequency?" can be
answered from the logs.
"""
from types import SimpleNamespace

from ato.adict import ADict
from loguru import logger

from comet.retriever import Retriever
from comet.vector_index import ScoredResult
from comet.schemas import MemoryNode


def _node(nid: str) -> MemoryNode:
    return MemoryNode(node_id=nid, content_key=f'ck_{nid}', raw_location=f'raw/{nid}', links=[])


def _retriever(summary_ids, trigger_ids):
    cfg = ADict(retrieval=ADict(top_k=5, fusion_alpha=0.5, rrf_k=5, raw_search_weight=0.2))
    nodes = {nid: _node(nid) for nid in set(summary_ids) | set(trigger_ids)}
    store = SimpleNamespace(
        get_node=lambda nid: nodes.get(nid),
        record_recall_hits=lambda ids: None,
    )
    vi = SimpleNamespace(
        count=50,
        search_by_summary=lambda q, k: [
            ScoredResult(node_id=n, score=0.1 + 0.01 * i, rank=i)
            for i, n in enumerate(summary_ids)
        ],
        search_by_trigger=lambda q, k: [
            ScoredResult(node_id=n, score=0.1 + 0.01 * i, rank=i)
            for i, n in enumerate(trigger_ids)
        ],
        search_by_raw=lambda q, k: [],
    )
    return Retriever(cfg, store, vi)


def _capture_recall_events(fn):
    events = []
    sink = logger.add(
        lambda m: events.append(dict(m.record['extra'])),
        filter=lambda r: r['extra'].get('event') == 'memory.recall',
        level='DEBUG',
    )
    try:
        fn()
    finally:
        logger.remove(sink)
    return events


def test_recall_emits_journal_with_trigger_share():
    """A retrieval emits one memory.recall event carrying recalled ids and the
    subset that matched via the trigger channel."""
    r = _retriever(summary_ids=['n1', 'n2'], trigger_ids=['n2', 'n3'])
    events = _capture_recall_events(lambda: r.retrieve_dual('summary q', 'trigger q', top_k=5))

    assert len(events) == 1
    e = events[0]
    recalled = set(e['node_ids'])
    assert {'n1', 'n2', 'n3'} <= recalled          # all primary matches journaled
    assert e['n_recalled'] == len(e['node_ids'])
    # n2 and n3 came through the trigger channel; n1 did not.
    assert set(e['trigger_matched']) == {'n2', 'n3'}
    assert e['n_trigger_matched'] == 2


def test_no_recall_no_event():
    """Empty result set emits nothing (no journal noise on a miss)."""
    r = _retriever(summary_ids=[], trigger_ids=[])
    events = _capture_recall_events(lambda: r.retrieve_dual('q', 'q', top_k=5))
    assert events == []


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
