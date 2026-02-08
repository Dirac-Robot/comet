"""Consolidator: Session-to-RAG migration with dedup, cross-linking, and tag normalization."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ato.adict import ADict
from loguru import logger

from comet.schemas import MemoryNode
from comet.storage import MemoryStore

if TYPE_CHECKING:
    from comet.vector_index import VectorIndex


MERGE_THRESHOLD = 0.32
CROSS_LINK_THRESHOLD = 0.15


class Consolidator:
    """Consolidates session memory nodes into a unified RAG knowledge base.

    Three-phase process:
    1. Dedup: detect semantically similar nodes via VectorIndex, merge if above threshold
    2. Cross-link: create bidirectional links between similar (but non-duplicate) nodes
    3. Tag normalization: unify variant tags that refer to the same concept
    """

    def __init__(
        self,
        config: ADict,
        store: MemoryStore,
        vector_index: Optional[VectorIndex] = None,
    ):
        self._config = config
        self._store = store
        self._vector_index = vector_index
        self._merge_threshold = config.get(
            'consolidation', {},
        ).get('merge_threshold', MERGE_THRESHOLD)
        self._cross_link_threshold = config.get(
            'consolidation', {},
        ).get('cross_link_threshold', CROSS_LINK_THRESHOLD)

    def consolidate(self, node_ids: Optional[list[str]] = None) -> dict:
        """Run full consolidation pipeline on given nodes (or all nodes if None).

        Returns a summary dict with counts of actions taken.
        """
        if not self._vector_index:
            logger.warning('VectorIndex not available, skipping consolidation')
            return {'status': 'skipped', 'reason': 'no_vector_index'}

        if node_ids is None:
            all_entries = self._store.list_all()
            node_ids = [e['node_id'] for e in all_entries]

        if not node_ids:
            return {'status': 'empty', 'merged': 0, 'linked': 0, 'tags_normalized': 0}

        merged = self._dedup(node_ids)
        linked = self._cross_link(node_ids)
        normalized = self._normalize_tags()

        summary = {
            'status': 'done',
            'merged': merged,
            'linked': linked,
            'tags_normalized': normalized,
        }
        logger.info(f'Consolidation complete: {summary}')
        return summary

    def _dedup(self, node_ids: list[str]) -> int:
        """Detect and merge semantically duplicate nodes.

        For each node, search VectorIndex for similar nodes.
        If similarity > threshold, merge the newer node into the older one
        (keep older node_id, append raw data reference, update summary).
        """
        merged_count = 0
        merged_into: dict[str, str] = {}

        for node_id in node_ids:
            if node_id in merged_into:
                continue

            node = self._store.get_node(node_id)
            if node is None:
                continue

            hits = self._vector_index.search_by_summary(node.summary, top_k=5)

            for hit in hits:
                if hit.node_id == node_id:
                    continue
                if hit.node_id in merged_into:
                    continue

                similarity = 1.0-hit.score
                if similarity < self._merge_threshold:
                    continue

                other = self._store.get_node(hit.node_id)
                if other is None:
                    continue

                if node.created_at <= other.created_at:
                    keeper, absorbed = node, other
                else:
                    keeper, absorbed = other, node

                self._merge_nodes(keeper, absorbed)
                merged_into[absorbed.node_id] = keeper.node_id
                merged_count += 1
                logger.info(
                    f'Merged {absorbed.node_id} into {keeper.node_id} '
                    f'(similarity={similarity:.3f})'
                )

                if node_id == absorbed.node_id:
                    break

        return merged_count

    def _merge_nodes(self, keeper: MemoryNode, absorbed: MemoryNode):
        """Merge absorbed node into keeper node."""
        combined_tags = list(set(keeper.topic_tags + absorbed.topic_tags))
        keeper.topic_tags = combined_tags

        combined_links = list(set(keeper.links + absorbed.links))
        if absorbed.node_id in combined_links:
            combined_links.remove(absorbed.node_id)
        if keeper.node_id in combined_links:
            combined_links.remove(keeper.node_id)
        keeper.links = combined_links

        if absorbed.recall_mode in ('passive', 'both') and keeper.recall_mode == 'active':
            keeper.recall_mode = absorbed.recall_mode

        self._store.save_node(keeper)

        for entry in self._store.list_all():
            if absorbed.node_id in entry.get('links', []):
                linked_node = self._store.get_node(entry['node_id'])
                if linked_node:
                    linked_node.links = [
                        keeper.node_id if lid == absorbed.node_id else lid
                        for lid in linked_node.links
                    ]
                    linked_node.links = list(set(linked_node.links))
                    if linked_node.node_id in linked_node.links:
                        linked_node.links.remove(linked_node.node_id)
                    self._store.save_node(linked_node)

        self._vector_index.delete(absorbed.node_id)
        self._store.delete_node(absorbed.node_id)

        self._vector_index.upsert(keeper)

    def _cross_link(self, node_ids: list[str]) -> int:
        """Create bidirectional links between similar (but non-duplicate) nodes."""
        linked_count = 0

        for node_id in node_ids:
            node = self._store.get_node(node_id)
            if node is None:
                continue

            hits = self._vector_index.search_by_summary(node.summary, top_k=10)

            for hit in hits:
                if hit.node_id == node_id:
                    continue

                similarity = 1.0-hit.score
                if similarity < self._cross_link_threshold:
                    continue

                if hit.node_id in node.links:
                    continue

                other = self._store.get_node(hit.node_id)
                if other is None:
                    continue

                node.links.append(hit.node_id)
                if node_id not in other.links:
                    other.links.append(node_id)
                    self._store.save_node(other)
                linked_count += 1

            if node.links:
                self._store.save_node(node)

        return linked_count

    def _normalize_tags(self) -> int:
        """Unify semantically equivalent tags across all nodes.

        Uses simple substring/case-insensitive matching to find variant tags.
        """
        all_tags = self._store.get_all_tags()
        if not all_tags:
            return 0

        tag_map: dict[str, str] = {}
        sorted_tags = sorted(all_tags, key=len)

        for i, tag_a in enumerate(sorted_tags):
            if tag_a in tag_map:
                continue
            for tag_b in sorted_tags[i+1:]:
                if tag_b in tag_map:
                    continue
                if tag_a.lower() == tag_b.lower() and tag_a != tag_b:
                    tag_map[tag_b] = tag_a
                elif tag_a.lower() in tag_b.lower() and len(tag_a) >= 2:
                    tag_map[tag_b] = tag_a

        if not tag_map:
            return 0

        normalized_count = 0
        for entry in self._store.list_all():
            node = self._store.get_node(entry['node_id'])
            if node is None:
                continue

            new_tags = []
            changed = False
            for tag in node.topic_tags:
                if tag in tag_map:
                    new_tags.append(tag_map[tag])
                    changed = True
                else:
                    new_tags.append(tag)

            if changed:
                node.topic_tags = list(set(new_tags))
                self._store.save_node(node)
                normalized_count += 1

        if tag_map:
            logger.info(f'Tag normalization: {tag_map}')

        return normalized_count
