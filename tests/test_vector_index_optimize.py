"""VectorIndex LanceDB compaction — regression for unbounded fragment/version
growth (every merge_insert appends a new fragment+version; without periodic
optimize the store grows one-per-write → GBs of disk + inflated RAM)."""
import tempfile
from datetime import timedelta

from ato.adict import ADict

import comet.vector_index as vi


def _make_index(monkeypatch):
    # Mock the embedding provider so construction needs no API/model.
    monkeypatch.setattr(vi, 'create_embeddings', lambda config: (lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]))
    tmp = tempfile.mkdtemp()
    cfg = ADict(retrieval=ADict(vector_db_path=tmp))
    return vi.VectorIndex(cfg)


def test_optimize_tables_collapses_versions_and_preserves_rows(monkeypatch):
    idx = _make_index(monkeypatch)
    table = idx._summary_table
    for i in range(25):
        vi.VectorIndex._upsert_rows(table, [idx._row(f'n{i}', [0.1, 0.2, 0.3, 0.4], f'doc {i}',
                                                      {'recall_mode': 'active', 'topic_tags': '', 'created_at': '2026-01-01T00:00:00'})])
    assert len(table.list_versions()) > 1, 'precondition: many versions from per-write fragments'
    rows_before = table.count_rows()

    idx.optimize_tables(cleanup_older_than=timedelta(0))

    assert len(table.list_versions()) == 1, 'optimize should collapse to a single version'
    assert table.count_rows() == rows_before == 25, 'no rows lost in compaction'


def test_note_adds_triggers_optimize_every_n(monkeypatch):
    idx = _make_index(monkeypatch)
    calls = {'n': 0}
    monkeypatch.setattr(idx, '_optimize_async', lambda *a, **k: calls.__setitem__('n', calls['n'] + 1))

    for _ in range(vi._OPTIMIZE_EVERY_N_ADDS - 1):
        idx._note_adds(1)
    assert calls['n'] == 0  # not yet at the threshold
    idx._note_adds(1)
    assert calls['n'] == 1  # fired at the threshold
    assert idx._adds_since_optimize == 0  # counter reset
    # a batch add that crosses the threshold also fires
    idx._note_adds(vi._OPTIMIZE_EVERY_N_ADDS + 5)
    assert calls['n'] == 2


def test_optimize_tables_is_noop_when_closed(monkeypatch):
    idx = _make_index(monkeypatch)
    idx._db = None  # simulate close()
    idx.optimize_tables(cleanup_older_than=timedelta(0))  # must not raise


if __name__ == '__main__':
    import pytest, sys
    sys.exit(pytest.main([__file__, '-v']))
