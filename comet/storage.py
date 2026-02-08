"""MemoryStore: Key-Value storage abstraction for CoMeT nodes."""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ato.adict import ADict
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
        
        self._ensure_dirs()
        self._index: dict = self._load_index()

    def _ensure_dirs(self):
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        self._raw_path.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self._index_path, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

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
        
        self._index[node.node_id] = {
            'summary': node.summary,
            'trigger': node.trigger,
            'recall_mode': node.recall_mode,
            'topic_tags': node.topic_tags,
            'depth_level': node.depth_level,
            'created_at': node.created_at.isoformat(),
        }
        self._save_index()
        
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

    def read_memory(self, node_id: str, depth: int = 0) -> Optional[str]:
        """
        Navigation Tool: Read memory at specified depth.
        
        depth=0: Summary only
        depth=1: Summary + topic tags + links
        depth=2: Full raw data
        """
        node = self.get_node(node_id)
        if not node:
            return None
        
        if depth == 0:
            return f"{node.summary} | {node.trigger}" if node.trigger else node.summary
        
        if depth == 1:
            return (
                f"[{node.node_id}]\n"
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
        return [
            {'node_id': k, **v}
            for k, v in self._index.items()
        ]

    def get_all_tags(self) -> set[str]:
        """Get all unique topic tags across all nodes."""
        tags = set()
        for meta in self._index.values():
            for tag in meta.get('topic_tags', []):
                tags.add(tag)
        return tags

    def delete_node(self, node_id: str) -> bool:
        """Delete a memory node and remove from index."""
        node_file = self._nodes_path/f"{node_id}.json"
        if node_file.exists():
            node_file.unlink()
        if node_id in self._index:
            del self._index[node_id]
            self._save_index()
            return True
        return False

