"""VectorIndex LanceDB compaction — regression for unbounded fragment/version
growth (every merge_insert appends a new fragment+version; without periodic
optimize the store grows one-per-write → GBs of disk + inflated RAM)."""
import tempfile
from datetime import timedelta

import pytest
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


class _FlakyTable:
    """Table whose optimize() raises a retryable commit conflict the first
    ``fail_times`` calls, then succeeds — models a concurrent Rewrite winning the
    commit race (a second daemon sharing the store, or boot reclaim overlapping)."""

    def __init__(self, name, fail_times, exc_msg):
        self.name = name
        self.fail_times = fail_times
        self.exc_msg = exc_msg
        self.calls = 0

    def optimize(self, cleanup_older_than=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.exc_msg)


def _index_no_boot(monkeypatch):
    # Disable the __init__ boot-reclaim thread so the only optimize() caller is the
    # synchronous test call (otherwise the daemon-thread reclaim holds _optimize_lock
    # and our call returns early, or races extra calls onto the fake table).
    monkeypatch.setattr(vi, 'create_embeddings', lambda config: (lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]))
    monkeypatch.setattr(vi.VectorIndex, '_optimize_async', lambda self, **k: None)
    monkeypatch.setattr(vi, '_OPTIMIZE_RETRY_BASE_S', 0.0)  # no real backoff in tests
    tmp = tempfile.mkdtemp()
    cfg = ADict(retrieval=ADict(vector_db_path=tmp))
    return vi.VectorIndex(cfg)


def _run_single_table(idx, table):
    idx._summary_table = table
    idx._trigger_table = None
    idx._raw_table = None
    idx.optimize_tables(cleanup_older_than=timedelta(0))


def test_optimize_retries_retryable_commit_conflict_until_success(monkeypatch):
    idx = _index_no_boot(monkeypatch)
    msg = ('lance error: Retryable commit conflict for version 207: This Rewrite '
           'transaction was preempted by concurrent transaction Rewrite at version 207.')
    table = _FlakyTable('comet_raw', fail_times=2, exc_msg=msg)
    _run_single_table(idx, table)
    assert table.calls == 3, 'should retry the retryable conflict, then succeed'


def test_optimize_gives_up_after_retry_budget(monkeypatch):
    idx = _index_no_boot(monkeypatch)
    msg = 'Retryable commit conflict: preempted by concurrent transaction Rewrite'
    table = _FlakyTable('comet_raw', fail_times=99, exc_msg=msg)  # never succeeds
    _run_single_table(idx, table)  # must NOT raise (non-fatal)
    assert table.calls == vi._OPTIMIZE_MAX_RETRIES + 1, 'bounded: 1 try + N retries'


def test_optimize_does_not_retry_non_retryable_error(monkeypatch):
    idx = _index_no_boot(monkeypatch)
    # a manifest/data-not-found is NOT a retryable conflict — fail fast, log once.
    table = _FlakyTable('comet_summaries', fail_times=99,
                        exc_msg='LanceError(IO): Object at location ... manifest not found')
    _run_single_table(idx, table)  # must NOT raise
    assert table.calls == 1, 'non-retryable error is not retried'


# ── Fix C: stale-handle self-heal (a compaction pruned files a cached handle
#    still pointed at — reopen + retry once instead of failing every op until a
#    restart). Same hazard as the live ['data/*.lance not found'] daemon errors. ─

def test_is_stale_handle_classifier():
    f = vi.VectorIndex._is_stale_handle
    assert f(RuntimeError('lance error: Not found: x/comet_raw.lance/data/abc.lance'))
    assert f(RuntimeError('manifest not found: x/comet_summaries.lance/_versions/9.manifest'))
    assert f(RuntimeError('No such file or directory: x/comet_triggers.lance/data/y.lance'))
    # a retryable commit conflict is a DIFFERENT class (handled by the optimize
    # retry, not a reopen) — and a plain error is neither.
    assert not f(RuntimeError('Retryable commit conflict for version 5'))
    assert not f(ValueError('invalid argument: top_k must be positive'))


class _Stale:
    name = 'comet_summaries'

    def count_rows(self):
        return 1

    def search(self, _vec):
        raise RuntimeError('lance error: Not found: x/comet_summaries.lance/data/y.lance')


class _Good:
    name = 'comet_summaries'

    def count_rows(self):
        return 1

    def search(self, _vec):
        return self

    def distance_type(self, *_a):
        return self

    def limit(self, *_a):
        return self

    def to_list(self):
        return [{'id': 'n1', '_distance': 0.2}]


def test_search_reopens_on_stale_handle_and_retries(monkeypatch):
    idx = _index_no_boot(monkeypatch)
    idx._summary_table = _Stale()
    monkeypatch.setattr(idx, '_refresh_all_tables', lambda: setattr(idx, '_summary_table', _Good()))
    out = idx._search_table(idx._summary_table, [0.1, 0.2, 0.3, 0.4], 3)
    assert len(out) == 1 and out[0].node_id == 'n1'  # reopened + retried successfully


def test_search_does_not_reopen_on_unrelated_error(monkeypatch):
    idx = _index_no_boot(monkeypatch)
    refreshed = {'n': 0}
    monkeypatch.setattr(idx, '_refresh_all_tables', lambda: refreshed.__setitem__('n', refreshed['n'] + 1))

    class _Boom:
        name = 'comet_raw'

        def count_rows(self):
            return 1

        def search(self, _vec):
            raise RuntimeError('some unrelated failure')

    assert idx._search_table(_Boom(), [0.1, 0.2, 0.3, 0.4], 3) == []
    assert refreshed['n'] == 0  # no reopen on a non-stale error


def test_resilient_write_reopens_on_stale_handle_and_retries(monkeypatch):
    idx = _index_no_boot(monkeypatch)
    calls = {'n': 0}
    refreshed = {'n': 0}
    monkeypatch.setattr(idx, '_refresh_all_tables', lambda: refreshed.__setitem__('n', refreshed['n'] + 1))

    def write_fn():
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('lance error: Not found: x/comet_summaries.lance/data/z.lance')

    idx._resilient_write(write_fn)
    assert calls['n'] == 2 and refreshed['n'] == 1  # reopened once, retried once


def test_resilient_write_reraises_unrelated_error(monkeypatch):
    idx = _index_no_boot(monkeypatch)

    def write_fn():
        raise ValueError('boom')

    with pytest.raises(ValueError):
        idx._resilient_write(write_fn)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
