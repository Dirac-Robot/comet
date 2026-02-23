"""Test: MemoryStore snapshot create/restore/discard."""
import json
import shutil
import tempfile
from pathlib import Path

from ato.adict import ADict

from comet.schemas import MemoryNode
from comet.storage import MemoryStore


def make_config(base_dir: str) -> ADict:
    return ADict(
        storage=ADict(
            base_path=f'{base_dir}/store',
            raw_path=f'{base_dir}/raw',
        )
    )


def make_node(node_id: str, summary: str) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        depth_level=0,
        recall_mode='active',
        topic_tags=['test'],
        summary=summary,
        trigger='test trigger',
        content_key=f'raw_{node_id}',
        raw_location=f'raw_{node_id}.txt',
    )


def test_snapshot_create_and_restore():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)

        node_a = make_node('node_a', 'Original summary A')
        node_b = make_node('node_b', 'Original summary B')
        store.save_node(node_a)
        store.save_node(node_b)
        store.save_raw('raw_node_a', 'Raw A')
        store.save_raw('raw_node_b', 'Raw B')
        assert len(store.list_all()) == 2

        store.create_snapshot('test_op')
        assert store.has_pending_snapshot('test_op')

        node_a.summary = 'Modified summary A'
        store.save_node(node_a)
        store.delete_node('node_b')
        assert len(store.list_all()) == 1

        restored_a = store.get_node('node_a')
        assert restored_a.summary == 'Modified summary A'

        store.restore_snapshot('test_op')

        assert len(store.list_all()) == 2
        restored_a = store.get_node('node_a')
        assert restored_a.summary == 'Original summary A'
        restored_b = store.get_node('node_b')
        assert restored_b is not None
        assert restored_b.summary == 'Original summary B'

        assert not store.has_pending_snapshot('test_op')
        print('PASS: test_snapshot_create_and_restore')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_snapshot_discard():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)

        node = make_node('node_x', 'Summary X')
        store.save_node(node)
        store.create_snapshot('discard_test')
        assert store.has_pending_snapshot('discard_test')

        node.summary = 'New summary X'
        store.save_node(node)

        store.discard_snapshot('discard_test')
        assert not store.has_pending_snapshot('discard_test')

        stored = store.get_node('node_x')
        assert stored.summary == 'New summary X'
        print('PASS: test_snapshot_discard')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_snapshot_pending_detection():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)
        store.save_node(make_node('n1', 'S1'))
        store.create_snapshot('consolidation')

        store.save_node(make_node('n2', 'S2'))
        assert len(store.list_all()) == 2

        store2 = MemoryStore(config)
        assert store2.has_pending_snapshot('consolidation')
        store2.restore_snapshot('consolidation')

        assert len(store2.list_all()) == 1
        assert store2.get_node('n1') is not None
        assert store2.get_node('n2') is None
        print('PASS: test_snapshot_pending_detection')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    test_snapshot_create_and_restore()
    test_snapshot_discard()
    test_snapshot_pending_detection()
    print('\nAll snapshot tests passed!')
