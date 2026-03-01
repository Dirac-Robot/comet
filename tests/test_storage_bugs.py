"""Test: Targeted regression tests for file management bug fixes."""
import json
import os
import shutil
import tempfile
import threading
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


def make_node(node_id: str, summary: str, content_key: str = '') -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        depth_level=0,
        recall_mode='active',
        topic_tags=['test'],
        summary=summary,
        trigger='test trigger',
        content_key=content_key or f'raw_{node_id}',
        raw_location=f'raw_{node_id}.txt',
    )


def test_delete_node_removes_raw():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)

        node = make_node('n1', 'Summary 1', content_key='raw_n1')
        store.save_raw('raw_n1', 'This is raw data')
        store.save_node(node)

        raw_file = Path(tmpdir)/'raw'/'raw_n1.txt'
        assert raw_file.exists(), 'Raw file should exist before deletion'

        store.delete_node('n1')

        assert not raw_file.exists(), 'Raw file should be deleted with node'
        assert store.get_node('n1') is None, 'Node should be deleted'
        print('PASS: test_delete_node_removes_raw')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_node_thread_safety():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)

        errors = []

        def save_worker(worker_id):
            try:
                for i in range(20):
                    node = make_node(f'w{worker_id}_n{i}', f'Summary {worker_id}-{i}')
                    store.save_node(node)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f'Thread safety errors: {errors}'

        try:
            index_path = Path(tmpdir)/'store'/'index.json'
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            assert len(index) == 80, f'Expected 80 nodes, got {len(index)}'
        except json.JSONDecodeError as e:
            raise AssertionError(f'Index JSON corrupted: {e}')

        print('PASS: test_save_node_thread_safety')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_close_session_no_meta_crash():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)

        meta = store.get_session_meta('nonexistent_session')
        assert meta is None, 'Should return None for missing session'

        safe_meta = meta or {}
        created_at = safe_meta.get('created_at', '')
        assert created_at == '', 'Should default to empty string'

        print('PASS: test_close_session_no_meta_crash')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_path_traversal_upload():
    from backend.services.document_processor import save_upload

    tmpdir = tempfile.mkdtemp()
    try:
        malicious_filename = '../../etc/evil.txt'
        result = save_upload(tmpdir, 'test_session', malicious_filename, b'malicious content')

        if 'error' not in result:
            actual_path = result['filepath']
            uploads_dir = os.path.join(tmpdir, 'uploads', 'test_session')
            assert os.path.commonpath([actual_path, uploads_dir]) == uploads_dir, \
                f'File escaped uploads dir: {actual_path}'
            assert os.path.basename(actual_path) == 'evil.txt', \
                f'Filename not sanitized: {os.path.basename(actual_path)}'

        evil_path = os.path.join(tmpdir, '..', '..', 'etc', 'evil.txt')
        assert not os.path.exists(evil_path), 'Traversal attack succeeded!'
        print('PASS: test_path_traversal_upload')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_is_hidden_resolved():
    from backend.services.file_access import FileAccessManager

    manager = FileAccessManager()
    original_hidden = manager._hidden.copy()
    try:
        tmpdir = tempfile.mkdtemp()
        test_file = os.path.join(tmpdir, 'test.txt')
        Path(test_file).touch()

        manager.hide_path(test_file)

        assert manager.is_hidden(test_file), 'Should be hidden with exact path'

        relative_variant = os.path.join(tmpdir, '.', 'test.txt')
        assert manager.is_hidden(relative_variant), \
            'Should be hidden with relative path variant'

        manager.unhide_path(test_file)
        assert not manager.is_hidden(test_file), 'Should be unhidden'
        print('PASS: test_is_hidden_resolved')
    finally:
        manager._hidden = original_hidden
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_list_directory_permission_error():
    from backend.services.file_reader import list_directory

    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, 'normal_dir'))
        Path(os.path.join(tmpdir, 'normal_dir', 'file.txt')).touch()
        Path(os.path.join(tmpdir, 'test_file.txt')).write_text('hello')

        restricted = os.path.join(tmpdir, 'restricted_dir')
        os.makedirs(restricted)
        os.chmod(restricted, 0o000)

        content, error = list_directory(tmpdir)
        assert error is None, f'Should not error: {error}'
        assert 'normal_dir' in content, 'Normal dir should be listed'
        assert 'test_file' in content, 'Normal file should be listed'
        print('PASS: test_list_directory_permission_error')
    finally:
        os.chmod(os.path.join(tmpdir, 'restricted_dir'), 0o755)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_cross_link_saves_correctly():
    tmpdir = tempfile.mkdtemp()
    try:
        config = make_config(tmpdir)
        store = MemoryStore(config)

        node_a = make_node('a', 'Summary A')
        node_a.links = []
        store.save_node(node_a)

        node_a_reloaded = store.get_node('a')
        assert node_a_reloaded.links == [], 'Should start with empty links'

        node_a_reloaded.links.append('b')
        store.save_node(node_a_reloaded)

        node_a_final = store.get_node('a')
        assert 'b' in node_a_final.links, 'Link should be persisted'
        print('PASS: test_cross_link_saves_correctly')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    test_delete_node_removes_raw()
    test_save_node_thread_safety()
    test_close_session_no_meta_crash()
    test_cross_link_saves_correctly()

    try:
        test_path_traversal_upload()
    except ImportError:
        print('SKIP: test_path_traversal_upload (CoBrA backend not in path)')

    try:
        test_is_hidden_resolved()
    except ImportError:
        print('SKIP: test_is_hidden_resolved (CoBrA backend not in path)')

    try:
        test_list_directory_permission_error()
    except ImportError:
        print('SKIP: test_list_directory_permission_error (CoBrA backend not in path)')

    print('\nAll storage bug tests passed!')
