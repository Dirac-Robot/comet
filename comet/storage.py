"""MemoryStore: Key-Value storage abstraction for CoMeT nodes."""
import json
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ato.adict import ADict
from loguru import logger

from comet.schemas import MemoryNode


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
        self._index: dict = self._load_index()
        self._sessions: dict = self._load_sessions()

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
            with open(self._index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self._index_path, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def _load_sessions(self) -> dict:
        if self._sessions_path.exists():
            with open(self._sessions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_sessions(self):
        with open(self._sessions_path, 'w', encoding='utf-8') as f:
            json.dump(self._sessions, f, ensure_ascii=False, indent=2)

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
        """Save raw data and return file path."""
        raw_file = self._raw_path/f"{content_key}.txt"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(raw_data)
        return str(raw_file)

    def save_node(self, node: MemoryNode) -> str:
        """Save a memory node and update index."""
        node_file = self._nodes_path/f"{node.node_id}.json"

        with open(node_file, 'w', encoding='utf-8') as f:
            json.dump(node.model_dump(mode='json'), f, ensure_ascii=False, indent=2)

        with self._lock:
            self._index[node.node_id] = {
                'summary': node.summary,
                'trigger': node.trigger,
                'recall_mode': node.recall_mode,
                'topic_tags': node.topic_tags,
                'depth_level': node.depth_level,
                'session_id': node.session_id,
                'created_at': node.created_at.isoformat(),
            }
            self._save_index()

        if node.session_id:
            self.link_node_to_session(node.session_id, node.node_id)

        return node.node_id

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory node by ID."""
        node_file = self._nodes_path/f"{node_id}.json"
        if not node_file.exists():
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
            return (
                f"[{node.node_id}]\n"
                f"Detailed: {detail}\n"
                f"Topics: {', '.join(node.topic_tags)}\n"
                f"Trigger: {node.trigger}\n"
                f"Links: {', '.join(node.links) if node.links else 'None'}"
            )

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
        meta = self._sessions.get(session_id, {})
        node_ids = meta.get('node_ids', [])
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
        existing = self._sessions.get(session_id, {})
        existing.update(meta)
        if 'node_ids' not in existing:
            existing['node_ids'] = []
        self._sessions[session_id] = existing
        self._save_sessions()

    def link_node_to_session(self, session_id: str, node_id: str):
        """Add node_id to a session's node_ids list."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {'node_ids': []}
        node_ids = self._sessions[session_id].setdefault('node_ids', [])
        if node_id not in node_ids:
            node_ids.append(node_id)
            self._save_sessions()

    def unlink_node_from_sessions(self, node_id: str):
        """Remove node_id from all sessions' node_ids lists."""
        changed = False
        for meta in self._sessions.values():
            nids = meta.get('node_ids', [])
            if node_id in nids:
                nids.remove(node_id)
                changed = True
        if changed:
            self._save_sessions()

    def unlink_node_from_session(self, session_id: str, node_id: str) -> bool:
        """Remove node_id from a specific session's node_ids list."""
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
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """List all registered sessions with metadata."""
        return [
            {'session_id': k, **v}
            for k, v in self._sessions.items()
        ]

    def delete_node(self, node_id: str) -> bool:
        """Delete a memory node and remove from index and sessions."""
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
                    with open(other_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                continue

    def _rules_path(self) -> Path:
        return self._base_path/'rules.json'

    def load_rules(self) -> list[dict]:
        path = self._rules_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def save_rule(self, rule: str, source_node: str = ''):
        with self._lock:
            rules = self.load_rules()
            normalized = rule.strip().lower()
            if any(r['rule'].strip().lower() == normalized for r in rules):
                return
            rules.append({
                'rule': rule.strip(),
                'source_node': source_node,
                'created_at': datetime.now().isoformat(),
            })
            with open(self._rules_path(), 'w', encoding='utf-8') as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)

    def delete_rule(self, rule_text: str) -> bool:
        with self._lock:
            rules = self.load_rules()
            normalized = rule_text.strip().lower()
            filtered = [r for r in rules if r['rule'].strip().lower() != normalized]
            if len(filtered) == len(rules):
                return False
            with open(self._rules_path(), 'w', encoding='utf-8') as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
            return True
