"""MemoryStore: Key-Value storage abstraction for CoMeT nodes."""
import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ato.adict import ADict
from loguru import logger

from comet.schemas import MemoryNode


def _atomic_write_json(path, data, **kwargs):
    """Atomic JSON write. Ensures parent dir exists so the writer is
    self-sufficient against external dir wipes (e.g. host reset paths
    that recreate STORE_BASE under us)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault('ensure_ascii', False)
    kwargs.setdefault('indent', 2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, **kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path, text):
    """Atomic text write. Same self-sufficient parent-dir contract as
    :func:`_atomic_write_json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class MemoryStore:
    """
    Simple JSON-based Key-Value store for memory nodes.
    
    Structure:
      base_path/
        index.json          # node_id -> metadata mapping
        nodes/
          {node_id}.json    # Full node data
      raw_path/
        {content_key}.txt   # Raw data files
    """

    def __init__(self, config: ADict):
        self._config = config
        self._base_path = Path(config.storage.base_path)
        self._raw_path = Path(config.storage.raw_path)
        self._nodes_path = self._base_path/'nodes'
        self._index_path = self._base_path/'index.json'
        self._sessions_path = self._base_path/'sessions.json'

        self._ensure_dirs()
        self._lock = threading.RLock()
        # Closed stores reject all writes (no-op return). The host (e.g. CoBrA's
        # reset path) flips this to neutralise stale references held by background
        # threads — without it, a single save_node() after a disk wipe would
        # restore the entire pre-wipe in-memory _index via _save_index. Set BEFORE
        # _load_index so the reconcile below can _save_index().
        self._closed = False
        self._index: dict = self._load_index()
        # Reconcile the index against the actual node files BEFORE anything reads
        # it: a node file is the source of truth, so an index entry without one is
        # a dead 'ghost' (get_node → None, never recallable) that still inflates
        # list_all()/the memory-panel count and bloats index.json. Node files can
        # vanish in bulk (a reset/wipe tombstoning nodes/, a crash mid-op) on a
        # path that bypasses delete_node's file+index symmetry, leaving the index
        # un-pruned. Self-healing: re-converges on every store open.
        self._reconcile_index_with_files()
        self._sessions: dict = self._load_sessions()
        # Retrieval-hit buffer for usage-driven salience. The retriever bumps
        # counters here on every recall (hot-path cheap — no node I/O); the
        # dream reinforced-decay pass drains it and folds counts into
        # node.strength / last_recall_at. In-memory only: an approximate
        # signal, so losing it on restart is graceful (under-counts recency,
        # never corrupts a node).
        self._recall_hits: dict[str, int] = {}

    def _reconcile_index_with_files(self) -> int:
        """Drop index entries whose backing ``nodes/<id>.json`` is gone.

        The node FILE is the source of truth; ``save_node`` writes it atomically
        BEFORE adding the index entry, so an index entry without a file is always
        a removal that bypassed ``delete_node`` (file+index symmetry) — a dead
        ghost: ``get_node`` returns None (never recallable) yet ``list_all`` still
        counts it (the memory panel showed 19k ghosts vs ~85 real nodes). Runs
        once at open; idempotent + self-healing across the host's frequent
        restarts. Returns the number of orphans dropped."""
        if not self._index:
            return 0
        orphans = [nid for nid in list(self._index)
                   if not (self._nodes_path/f'{nid}.json').exists()]
        if not orphans:
            return 0
        for nid in orphans:
            self._index.pop(nid, None)
        self._save_index()
        logger.warning(
            f'index reconcile: dropped {len(orphans)} orphan index entries '
            f'(no backing node file); {len(self._index)} remain'
        )
        return len(orphans)

    def close(self):
        """Mark the store as closed and drop in-memory state.

        After close(), all writers (save_node, save_raw, _save_index,
        _save_sessions, delete_node, link/unlink, brief writers,
        inherited-memory writers) become no-ops. Reads still work for
        any in-flight callers, but they observe the cleared dicts.
        """
        with self._lock:
            self._closed = True
            self._index.clear()
            self._sessions.clear()

    def _ensure_dirs(self):
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        self._raw_path.mkdir(parents=True, exist_ok=True)

    def _snapshot_dir(self, label: str) -> Path:
        return self._base_path/'.snapshot'/label

    def has_pending_snapshot(self, label: str) -> bool:
        return self._snapshot_dir(label).exists()

    def create_snapshot(self, label: str):
        snap = self._snapshot_dir(label)
        if snap.exists():
            shutil.rmtree(snap)
        snap.mkdir(parents=True)

        snap_nodes = snap/'nodes'
        snap_nodes.mkdir()

        if self._index_path.exists():
            shutil.copy2(self._index_path, snap/'index.json')
        if self._sessions_path.exists():
            shutil.copy2(self._sessions_path, snap/'sessions.json')
        for node_file in self._nodes_path.iterdir():
            if node_file.suffix == '.json':
                shutil.copy2(node_file, snap_nodes/node_file.name)

        logger.info(f'Snapshot created: {label}')

    def restore_snapshot(self, label: str):
        snap = self._snapshot_dir(label)
        if not snap.exists():
            logger.warning(f'No snapshot to restore: {label}')
            return False

        with self._lock:
            snap_index = snap/'index.json'
            if snap_index.exists():
                shutil.copy2(snap_index, self._index_path)
            snap_sessions = snap/'sessions.json'
            if snap_sessions.exists():
                shutil.copy2(snap_sessions, self._sessions_path)

            snap_nodes = snap/'nodes'
            if snap_nodes.exists():
                for existing in self._nodes_path.iterdir():
                    if existing.suffix == '.json':
                        existing.unlink()
                for node_file in snap_nodes.iterdir():
                    if node_file.suffix == '.json':
                        shutil.copy2(node_file, self._nodes_path/node_file.name)

            self._index = self._load_index()
            self._sessions = self._load_sessions()

        self.discard_snapshot(label)
        logger.info(f'Snapshot restored: {label}')
        return True

    def discard_snapshot(self, label: str):
        snap = self._snapshot_dir(label)
        if snap.exists():
            shutil.rmtree(snap)
            logger.debug(f'Snapshot discarded: {label}')

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                with open(self._index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f'Corrupt index.json at {self._index_path}, starting fresh')
        return {}

    def _save_index(self):
        if self._closed:
            return
        _atomic_write_json(self._index_path, self._index)

    def _load_sessions(self) -> dict:
        if self._sessions_path.exists():
            try:
                with open(self._sessions_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f'Corrupt sessions.json at {self._sessions_path}, starting fresh')
        return {}

    def _save_sessions(self):
        if self._closed:
            return
        _atomic_write_json(self._sessions_path, self._sessions)

    def reload_index(self):
        """Reload index and sessions from disk, discarding in-memory state."""
        with self._lock:
            self._index = self._load_index()
            self._sessions = self._load_sessions()

    def generate_node_id(self) -> str:
        """Generate unique node ID with timestamp."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        short_uuid = uuid.uuid4().hex[:6]
        return f"mem_{ts}_{short_uuid}"

    def generate_content_key(self, prefix: str = 'raw') -> str:
        """Generate unique content key for raw data."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        short_uuid = uuid.uuid4().hex[:6]
        return f"{prefix}_{ts}_{short_uuid}"

    def save_raw(self, content_key: str, raw_data: str) -> str:
        if self._closed:
            return ''
        raw_file = self._raw_path/f"{content_key}.txt"
        _atomic_write_text(raw_file, raw_data)
        return str(raw_file)

    def save_node(self, node: MemoryNode) -> str:
        if self._closed:
            return node.node_id
        node_file = self._nodes_path/f"{node.node_id}.json"
        _atomic_write_json(node_file, node.model_dump(mode='json'))

        with self._lock:
            if self._closed:
                return node.node_id
            self._index[node.node_id] = {
                'summary': node.summary,
                'trigger': node.trigger,
                'recall_mode': node.recall_mode,
                'topic_tags': node.topic_tags,
                'depth_level': node.depth_level,
                'session_id': node.session_id,
                'links': node.links,
                'source_links': node.source_links,
                'capsule': node.capsule,
                'created_at': node.created_at.isoformat(),
            }
            self._save_index()

        return node.node_id

    def record_recall_hits(self, node_ids: list[str]) -> None:
        """Record retrieval hits for usage-driven salience reinforcement.

        Hot-path-cheap: bumps an in-memory counter under the store lock —
        no node deserialize/save. The dream reinforced-decay pass drains
        this buffer (``drain_recall_hits``) and folds the counts into
        node.strength / last_recall_at off the retrieval hot path. No-op on
        a closed store; hits for nodes that vanish before the drain are
        harmlessly dropped (reinforcement is an approximate signal).
        """
        if not node_ids or self._closed:
            return
        with self._lock:
            if self._closed:
                return
            for nid in node_ids:
                if nid:
                    self._recall_hits[nid] = self._recall_hits.get(nid, 0) + 1

    def drain_recall_hits(self) -> dict:
        """Return and atomically clear the accumulated recall-hit buffer."""
        with self._lock:
            hits = self._recall_hits
            self._recall_hits = {}
            return hits

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory node by ID.

        Merge-display ids (``mem_..._A+B``) are stored under their base node —
        the first segment — while the ``+suffix`` is a render-time marker for
        nodes consolidated into it, not a stored key. On a miss with a ``+`` in
        the id, fall back to the base so a displayed merge id handed back by a
        caller (search_linked, pin/detach) resolves instead of 404-ing.
        """
        node_file = self._nodes_path/f"{node_id}.json"
        if not node_file.exists():
            if '+' in node_id:
                base = node_id.split('+', 1)[0]
                node_file = self._nodes_path/f"{base}.json"
                if not node_file.exists():
                    return None
            else:
                return None

        with open(node_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return MemoryNode(**data)

    def get_raw(self, content_key: str) -> Optional[str]:
        """Retrieve raw data by content key."""
        raw_file = self._raw_path/f"{content_key}.txt"
        if not raw_file.exists():
            return None
        
        with open(raw_file, 'r', encoding='utf-8') as f:
            return f.read()

    def read_memory(self, node_id: str, depth: int = 0, detailed_summary: str = None) -> Optional[str]:
        """
        Navigation Tool: Read memory at specified depth.

        depth=0: Summary only
        depth=1: Detailed summary (lazy-generated) + metadata
        depth=2: Full raw data
        """
        node = self.get_node(node_id)
        if not node:
            return None

        if depth == 0:
            return f"{node.summary} | {node.trigger}" if node.trigger else node.summary

        if depth == 1:
            detail = detailed_summary or node.detailed_summary or node.summary
            capsule = getattr(node, 'capsule', '')
            source = getattr(node, 'source_links', [])
            parts = [
                f"[{node.node_id}]",
                f"Detailed: {detail}",
                f"Topics: {', '.join(node.topic_tags)}",
                f"Trigger: {node.trigger}",
                f"Links: {', '.join(node.links) if node.links else 'None'}",
            ]
            if capsule:
                parts.append(f"Capsule: {capsule}")
            if source:
                parts.append(f"Sources: {', '.join(source)}")
            return '\n'.join(parts)

        # depth >= 2: Full raw data + links for navigation
        raw = self.get_raw(node.content_key)
        links_text = ', '.join(node.links) if node.links else 'None'
        return (
            f"[{node.node_id}] - FULL RAW DATA\n"
            f"Topics: {', '.join(node.topic_tags)}\n"
            f"Linked nodes: {links_text}\n"
            f"{'='*40}\n"
            f"{raw or '(Raw data not found)'}"
        )

    def search_by_tag(self, tag: str) -> list[str]:
        """Find node IDs by topic tag."""
        results = []
        for node_id, meta in self._index.items():
            if tag.lower() in [t.lower() for t in meta.get('topic_tags', [])]:
                results.append(node_id)
        return results

    def list_all(self) -> list[dict]:
        """List all nodes with summaries."""
        with self._lock:
            return [
                {'node_id': k, **v}
                for k, v in self._index.items()
            ]

    def list_by_session(self, session_id: str) -> list[dict]:
        """List nodes belonging to a specific session via direct lookup."""
        with self._lock:
            meta = self._sessions.get(session_id, {})
            node_ids = list(meta.get('node_ids', []))
            if not node_ids:
                node_ids = [
                    nid for nid, info in self._index.items()
                    if info.get('session_id') == session_id
                ]
                if node_ids:
                    if session_id not in self._sessions:
                        self._sessions[session_id] = {}
                    self._sessions[session_id]['node_ids'] = node_ids
                    self._save_sessions()
                    logger.info(
                        f'Recovered {len(node_ids)} node_ids for session '
                        f'{session_id} from index'
                    )
            result = []
            for nid in node_ids:
                if nid in self._index:
                    result.append({'node_id': nid, **self._index[nid]})
            return result

    def get_all_tags(self) -> set[str]:
        """Get all unique topic tags across all nodes."""
        with self._lock:
            tags = set()
            for meta in self._index.values():
                for tag in meta.get('topic_tags', []):
                    tags.add(tag)
            return tags

    def save_session_meta(self, session_id: str, meta: dict):
        """Save or update session metadata in the registry."""
        with self._lock:
            existing = self._sessions.get(session_id, {})
            existing.update(meta)
            if 'node_ids' not in existing:
                existing['node_ids'] = []
            self._sessions[session_id] = existing
            self._save_sessions()

    def link_node_to_session(self, session_id: str, node_id: str):
        """Add node_id to a session's node_ids list."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {'node_ids': []}
            node_ids = self._sessions[session_id].setdefault('node_ids', [])
            if node_id not in node_ids:
                node_ids.append(node_id)
                self._save_sessions()

    def unlink_node_from_sessions(self, node_id: str):
        """Remove node_id from all sessions' node_ids lists."""
        with self._lock:
            changed = False
            # Iterate over a snapshot of the values so a concurrent
            # save_session_meta that adds a new key cannot trip the
            # iterator with "dictionary changed size during iteration".
            # The list mutation below targets the original meta dicts
            # (shared references), so the unlink still takes effect.
            for meta in list(self._sessions.values()):
                nids = meta.get('node_ids', [])
                if node_id in nids:
                    nids.remove(node_id)
                    changed = True
            if changed:
                self._save_sessions()

    def unlink_node_from_session(self, session_id: str, node_id: str) -> bool:
        """Remove node_id from a specific session's node_ids list."""
        with self._lock:
            meta = self._sessions.get(session_id)
            if not meta:
                return False
            nids = meta.get('node_ids', [])
            if node_id in nids:
                nids.remove(node_id)
                self._save_sessions()
                return True
            return False

    def get_session_meta(self, session_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific session."""
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """List all registered sessions with metadata."""
        with self._lock:
            return [
                {'session_id': k, **v}
                for k, v in self._sessions.items()
            ]

    def delete_session_meta(self, session_id: str) -> bool:
        """Remove session metadata from the registry entirely."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._save_sessions()
                return True
            return False

    def delete_node(self, node_id: str) -> bool:
        """Delete a memory node and remove from index and sessions."""
        if self._closed:
            return False
        with self._lock:
            node_file = self._nodes_path/f"{node_id}.json"
            if node_file.exists():
                node_file.unlink()
            self.unlink_node_from_sessions(node_id)
            self._remove_links_to(node_id)
            if node_id in self._index:
                del self._index[node_id]
                self._save_index()
                return True
            return False

    def _remove_links_to(self, target_id: str):
        """Remove references to target_id from other nodes' links arrays."""
        for other_id in list(self._index.keys()):
            if other_id == target_id:
                continue
            other_file = self._nodes_path/f"{other_id}.json"
            if not other_file.exists():
                continue
            try:
                with open(other_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                links = data.get('links', [])
                if target_id in links:
                    data['links'] = [l for l in links if l != target_id]
                    _atomic_write_json(other_file, data)
            except Exception:
                continue

    # ── Session briefs ──
    #
    # A session brief is the per-session replacement for the old cross-session
    # rules index. Each session has at most one brief, stored as a plain .md
    # file under {base}/session_briefs/<session_id>.md and fully rewritten
    # (never appended) by the compacter on each DIALOG node creation. Bounded
    # length is enforced by the LLM prompt, not by this layer.

    def _session_briefs_dir(self) -> Path:
        return self._base_path/'session_briefs'

    def _session_brief_path(self, session_id: str) -> Path:
        safe = session_id.replace('/', '_').replace('..', '_')
        return self._session_briefs_dir()/f'{safe}.md'

    def load_session_brief(self, session_id: str) -> str:
        path = self._session_brief_path(session_id)
        if not path.exists():
            return ''
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ''

    def save_session_brief(self, session_id: str, brief: str) -> None:
        if self._closed:
            return
        with self._lock:
            brief_dir = self._session_briefs_dir()
            brief_dir.mkdir(parents=True, exist_ok=True)
            path = self._session_brief_path(session_id)
            tmp = path.with_suffix(path.suffix+'.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(brief)
            os.replace(tmp, path)

    def delete_session_brief(self, session_id: str) -> bool:
        with self._lock:
            path = self._session_brief_path(session_id)
            if not path.exists():
                return False
            try:
                path.unlink()
                return True
            except Exception:
                return False

    def list_session_briefs(self) -> list[str]:
        brief_dir = self._session_briefs_dir()
        if not brief_dir.exists():
            return []
        return sorted(p.stem for p in brief_dir.glob('*.md'))

    # ── Inherited memory (handoff carry-over) ──
    #
    # On handoff, the new session inherits a curated slice of the source
    # session's high-importance nodes + the per-chunk synthesis nodes the
    # handoff compactor produced. The node_ids list holds curated HIGH
    # carry-overs; synthesis_node_ids holds the topical chunk summaries.
    # The harness renderer surfaces each as its own block so the successor
    # can tell at a glance: "these are carry-overs, not my own map."

    def _inherited_memory_dir(self) -> Path:
        return self._base_path/'inherited_memory'

    def _inherited_memory_path(self, session_id: str) -> Path:
        safe = session_id.replace('/', '_').replace('..', '_')
        return self._inherited_memory_dir()/f'{safe}.json'

    def load_inherited_memory(self, session_id: str) -> dict:
        path = self._inherited_memory_path(session_id)
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                # On-disk migration: pre-chunking payloads stored a single
                # `synthesis_node_id` string. Translate it to a 1-element
                # list so the rest of the stack only ever sees the list form.
                if 'synthesis_node_ids' not in data and data.get('synthesis_node_id'):
                    data['synthesis_node_ids'] = [data['synthesis_node_id']]
                data.setdefault('synthesis_node_ids', [])
                return data
        except Exception:
            pass
        return {}

    def save_inherited_memory(
        self, session_id: str, source_session_id: str, node_ids: list[str],
        synthesis_node_ids: list[str] | None = None,
    ) -> None:
        if self._closed:
            return
        with self._lock:
            target_dir = self._inherited_memory_dir()
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                'source_session_id': source_session_id,
                'node_ids': list(node_ids),
                'synthesis_node_ids': list(synthesis_node_ids or []),
                'created_at': datetime.now().isoformat(),
            }
            _atomic_write_json(self._inherited_memory_path(session_id), payload)

    def delete_inherited_memory(self, session_id: str) -> bool:
        with self._lock:
            path = self._inherited_memory_path(session_id)
            if not path.exists():
                return False
            try:
                path.unlink()
                return True
            except Exception:
                return False
