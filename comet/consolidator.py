"""Consolidator: Session-to-RAG migration with dedup, cross-linking, and tag normalization."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ato.adict import ADict
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field

from comet.llm_factory import create_chat_model
from comet.schemas import MemoryNode
from comet.storage import MemoryStore
from comet.templates import load_template

if TYPE_CHECKING:
    from comet.vector_index import VectorIndex


MERGE_THRESHOLD = 0.32
CROSS_LINK_THRESHOLD = 0.15
CLUSTER_THRESHOLD = 0.22
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 8


class ClusterValidation(BaseModel):
    """SLM output for cluster validation."""
    should_synthesize: bool = Field(description='Whether this cluster should be synthesized')
    reasoning: str = Field(description='Brief explanation')


class SynthesizedResult(BaseModel):
    """SLM output for virtual node creation."""
    summary: str = Field(description='Broader topic description covering all source nodes')
    trigger: str = Field(description='When to retrieve this synthesized knowledge')
    recall_mode: str = Field(default='active')
    topic_tags: list[str] = Field(description='1-3 topic tags for the unified topic')


class MergedSummaryTrigger(BaseModel):
    """SLM output for summary + trigger regeneration after node merge."""
    summary: str = Field(description='Merged summary covering both nodes')
    trigger: str = Field(description='Combined trigger covering both merged nodes')


class Consolidator:
    """Consolidates session memory nodes into a unified RAG knowledge base.

    Three-phase process:
    1. Dedup: detect semantically similar nodes via VectorIndex, merge if above threshold
    2. Cross-link: create bidirectional links between similar (but non-duplicate) nodes
    3. Tag normalization: unify variant tags that refer to the same concept

    Optional synthesis:
    4. Synthesize: cluster related nodes and create virtual parent nodes
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
        self._cluster_threshold = config.get(
            'consolidation', {},
        ).get('cluster_threshold', CLUSTER_THRESHOLD)
        self._llm: Optional[BaseChatModel] = None

    def _ensure_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = create_chat_model(self._config.slm_model, self._config)
        return self._llm

    def consolidate(self, node_ids: Optional[list[str]] = None) -> dict:
        """Run full consolidation pipeline on given nodes (or all nodes if None).

        Returns a summary dict with counts of actions taken.
        Snapshot-protected: creates a backup before mutation and restores
        on failure, so interrupted consolidation never corrupts state.
        """
        if not self._vector_index:
            logger.warning('VectorIndex not available, skipping consolidation')
            return {'status': 'skipped', 'reason': 'no_vector_index'}

        if node_ids is None:
            all_entries = self._store.list_all()
            node_ids = [e['node_id'] for e in all_entries]

        if not node_ids:
            return {'status': 'empty', 'merged': 0, 'linked': 0, 'tags_normalized': 0}

        self._store.create_snapshot('consolidation')
        try:
            merged_count, absorbed_ids = self._dedup(node_ids)
            live_ids = [nid for nid in node_ids if nid not in absorbed_ids]
            linked = self._cross_link(live_ids)
            normalized = self._normalize_tags()
            pruned = self._prune_dangling_links()
        except Exception:
            logger.error('Consolidation failed, restoring snapshot')
            self._restore_and_rebuild('consolidation')
            raise

        self._store.discard_snapshot('consolidation')

        summary = {
            'status': 'done',
            'merged': merged_count,
            'linked': linked,
            'tags_normalized': normalized,
            'pruned_links': pruned,
        }
        logger.info(f'Consolidation complete: {summary}')
        return summary

    # ------------------------------------------------------------------
    # Synthesis: cross-session virtual node creation
    # ------------------------------------------------------------------

    def synthesize(self, threshold: Optional[float] = None) -> list[MemoryNode]:
        """Create virtual nodes by clustering semantically related memories.

        Pipeline:
        1. Embedding-based clustering (Union-Find on pairwise similarity)
        2. SLM validates each cluster
        3. SLM generates synthesized summary/trigger
        4. Virtual node stored with bidirectional links to source nodes

        Snapshot-protected: restores on failure.
        Returns list of newly created virtual MemoryNodes.
        """
        if not self._vector_index:
            logger.warning('VectorIndex not available, skipping synthesis')
            return []

        effective_threshold = threshold or self._cluster_threshold
        clusters = self._find_clusters(effective_threshold)
        if not clusters:
            logger.info('No clusters found for synthesis')
            return []

        validated = self._validate_clusters(clusters)
        if not validated:
            logger.info('No clusters passed SLM validation')
            return []

        self._store.create_snapshot('synthesis')
        try:
            virtual_nodes = []
            for cluster_node_ids in validated:
                node = self._create_virtual_node(cluster_node_ids)
                if node:
                    virtual_nodes.append(node)
        except Exception:
            logger.error('Synthesis failed, restoring snapshot')
            self._restore_and_rebuild('synthesis')
            raise

        self._store.discard_snapshot('synthesis')
        logger.info(f'Synthesis complete: {len(virtual_nodes)} virtual nodes created')
        return virtual_nodes

    def _restore_and_rebuild(self, label: str):
        """Restore snapshot and rebuild VectorIndex from store."""
        self._store.restore_snapshot(label)
        if self._vector_index:
            self._rebuild_vector_index()

    def _rebuild_vector_index(self):
        """Rebuild VectorIndex from all nodes in MemoryStore."""
        if not self._vector_index:
            return
        self._vector_index.reset()
        for entry in self._store.list_all():
            node = self._store.get_node(entry['node_id'])
            if node:
                raw = self._store.get_raw(node.content_key) or ''
                self._vector_index.upsert(node, raw_content=raw[:8000])
        logger.info(f'VectorIndex rebuilt with {len(self._store.list_all())} nodes')

    def _find_clusters(self, threshold: float) -> list[list[str]]:
        """Find clusters of related nodes via embedding similarity + Union-Find."""
        all_entries = self._store.list_all()
        node_ids = [e['node_id'] for e in all_entries if e.get('depth_level', 0) < 2]
        if len(node_ids) < MIN_CLUSTER_SIZE:
            return []

        parent: dict[str, str] = {nid: nid for nid in node_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        seen_pairs: set[tuple[str, str]] = set()
        for node_id in node_ids:
            node = self._store.get_node(node_id)
            if node is None:
                continue
            hits = self._vector_index.search_by_summary(node.summary, top_k=10)
            for hit in hits:
                if hit.node_id == node_id or hit.node_id not in parent:
                    continue
                pair = tuple(sorted([node_id, hit.node_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                similarity = 1.0-hit.score
                if similarity >= threshold:
                    union(node_id, hit.node_id)

        groups: dict[str, list[str]] = {}
        for nid in node_ids:
            root = find(nid)
            groups.setdefault(root, []).append(nid)

        clusters = [
            members for members in groups.values()
            if MIN_CLUSTER_SIZE <= len(members) <= MAX_CLUSTER_SIZE
        ]

        logger.info(f'Found {len(clusters)} candidate clusters (threshold={threshold:.3f})')
        return clusters

    def _validate_clusters(self, clusters: list[list[str]]) -> list[list[str]]:
        """Use SLM to validate each cluster is a coherent knowledge unit."""
        llm = self._ensure_llm()
        structured_llm = llm.with_structured_output(ClusterValidation)
        template = load_template('synthesis_validate')
        validated = []

        for cluster_ids in clusters:
            nodes_text_parts = []
            for nid in cluster_ids:
                node = self._store.get_node(nid)
                if node:
                    nodes_text_parts.append(
                        f'- [{nid}] {node.summary}\n  Trigger: {node.trigger}\n  Tags: {", ".join(node.topic_tags)}'
                    )
            if len(nodes_text_parts) < MIN_CLUSTER_SIZE:
                continue

            prompt = template.format(
                count=len(nodes_text_parts),
                nodes_text='\n'.join(nodes_text_parts),
            )
            try:
                result: ClusterValidation = structured_llm.invoke(prompt)
                if result.should_synthesize:
                    validated.append(cluster_ids)
                    logger.info(f'Cluster validated ({len(cluster_ids)} nodes): {result.reasoning}')
                else:
                    logger.debug(f'Cluster rejected ({len(cluster_ids)} nodes): {result.reasoning}')
            except Exception as e:
                logger.warning(f'Cluster validation failed: {e}')
                continue

        return validated

    def _create_virtual_node(self, cluster_ids: list[str]) -> Optional[MemoryNode]:
        """Synthesize a virtual node from a validated cluster."""
        llm = self._ensure_llm()
        structured_llm = llm.with_structured_output(SynthesizedResult)
        template = load_template('synthesis_create')

        sources_parts = []
        raw_parts = []
        source_session_ids = set()
        for nid in cluster_ids:
            node = self._store.get_node(nid)
            if node is None:
                continue
            if node.session_id:
                source_session_ids.add(node.session_id)
            idx_entry = self._store._index.get(nid, {})
            for sid in idx_entry.get('session_ids', []):
                source_session_ids.add(sid)
            raw = self._store.get_raw(node.content_key) or ''
            sources_parts.append(
                f'### [{nid}]\nSummary: {node.summary}\nTrigger: {node.trigger}\n'
                f'Tags: {", ".join(node.topic_tags)}\n'
                f'Raw excerpt: {raw[:500]}' + ('...' if len(raw) > 500 else '')
            )
            raw_parts.append(f'[{nid}]\n{raw}')

        if len(sources_parts) < MIN_CLUSTER_SIZE:
            return None

        existing_tags = self._store.get_all_tags()
        tags_text = ', '.join(sorted(existing_tags)) if existing_tags else '(없음)'
        prompt = template.format(
            sources='\n\n'.join(sources_parts),
            existing_tags=tags_text,
        )

        try:
            result: SynthesizedResult = structured_llm.invoke(prompt)
        except Exception as e:
            logger.warning(f'Virtual node synthesis failed: {e}')
            return None

        combined_raw = '\n\n---\n\n'.join(raw_parts)
        node_id = self._store.generate_node_id()
        content_key = self._store.generate_content_key(prefix='synth')
        raw_location = self._store.save_raw(content_key, combined_raw)

        session_ids_list = sorted(source_session_ids)
        virtual_node = MemoryNode(
            node_id=node_id,
            depth_level=2,
            recall_mode='active',
            topic_tags=result.topic_tags,
            summary=result.summary,
            trigger=result.trigger,
            content_key=content_key,
            raw_location=raw_location,
            links=list(cluster_ids),
            session_id=session_ids_list[0] if session_ids_list else None,
        )
        self._store.save_node(virtual_node)

        for sid in session_ids_list:
            self._store.link_node_to_session(sid, node_id)

        for source_id in cluster_ids:
            source_node = self._store.get_node(source_id)
            if source_node and node_id not in source_node.links:
                source_node.links.append(node_id)
                self._store.save_node(source_node)

        if self._vector_index:
            self._vector_index.upsert(virtual_node, raw_content=combined_raw[:8000])

        logger.info(
            f'Virtual node {node_id} created from {len(cluster_ids)} sources: '
            f'{result.summary}'
        )
        return virtual_node

    # ------------------------------------------------------------------
    # Dedup / Cross-link / Tag normalization
    # ------------------------------------------------------------------

    def _dedup(self, node_ids: list[str]) -> tuple[int, set[str]]:
        """Detect and merge semantically duplicate nodes.

        For each node, search VectorIndex for similar nodes.
        If similarity > threshold, merge the newer node into the older one
        (keep older node_id, append raw data reference, update summary).

        Returns (merged_count, set of absorbed node IDs).
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

        return merged_count, set(merged_into.keys())

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
            nid = entry['node_id']
            if nid in (keeper.node_id, absorbed.node_id):
                continue
            linked_node = self._store.get_node(nid)
            if not linked_node or absorbed.node_id not in linked_node.links:
                continue
            linked_node.links = [
                keeper.node_id if lid == absorbed.node_id else lid
                for lid in linked_node.links
            ]
            linked_node.links = list(set(linked_node.links))
            if linked_node.node_id in linked_node.links:
                linked_node.links.remove(linked_node.node_id)
            self._store.save_node(linked_node)

        absorbed_sessions = []
        for sid, meta in self._store._sessions.items():
            if absorbed.node_id in meta.get('node_ids', []):
                absorbed_sessions.append(sid)
        for sid in absorbed_sessions:
            self._store.link_node_to_session(sid, keeper.node_id)

        self._vector_index.delete(absorbed.node_id)
        self._store.delete_node(absorbed.node_id)

        raw = self._store.get_raw(keeper.content_key) or ''
        self._vector_index.upsert(keeper, raw_content=raw)

        self._regenerate_summary_trigger(keeper, absorbed)

    def _regenerate_summary_trigger(self, keeper: MemoryNode, absorbed: MemoryNode):
        """Regenerate summary and trigger for keeper after merging absorbed node."""
        try:
            llm = self._ensure_llm()
            structured_llm = llm.with_structured_output(MergedSummaryTrigger)
            template = load_template('merge_summary')
            prompt = template.format(
                keeper_summary=keeper.summary,
                keeper_trigger=keeper.trigger,
                absorbed_summary=absorbed.summary,
                absorbed_trigger=absorbed.trigger,
            )
            result: MergedSummaryTrigger = structured_llm.invoke(prompt)
            old_summary = keeper.summary
            old_trigger = keeper.trigger
            keeper.summary = result.summary
            keeper.trigger = result.trigger
            self._store.save_node(keeper)
            if self._vector_index:
                raw = self._store.get_raw(keeper.content_key) or ''
                self._vector_index.upsert(keeper, raw_content=raw)
            logger.info(
                f'Merged node {keeper.node_id} regenerated: '
                f'summary "{old_summary[:40]}..." -> "{keeper.summary[:40]}...", '
                f'trigger "{old_trigger[:40]}..." -> "{keeper.trigger[:40]}..."'
            )
        except Exception as e:
            logger.warning(f'Summary/trigger regeneration failed for {keeper.node_id}, keeping original: {e}')

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

    def _prune_dangling_links(self) -> int:
        """Remove links pointing to non-existent nodes."""
        pruned = 0
        existing_ids = {e['node_id'] for e in self._store.list_all()}
        for entry in self._store.list_all():
            node = self._store.get_node(entry['node_id'])
            if node is None:
                continue
            clean_links = [lid for lid in node.links if lid in existing_ids]
            if len(clean_links) != len(node.links):
                removed = len(node.links)-len(clean_links)
                pruned += removed
                logger.debug(f'Pruned {removed} dangling link(s) from {node.node_id}')
                node.links = clean_links
                self._store.save_node(node)
        if pruned:
            logger.info(f'Pruned {pruned} total dangling links')
        return pruned

    # ------------------------------------------------------------------
    # GCRI External Memory Ingestion
    # ------------------------------------------------------------------

    def ingest_gcri_memory(self, gcri_memory_path: str, session_id: Optional[str] = None) -> dict:
        """Ingest GCRI external memory (rules + knowledge) as CoMeT MemoryNodes.

        Converts GCRI's JSON-based external memory into CoMeT nodes so they
        can participate in consolidation, cross-linking, and retrieval.

        Args:
            gcri_memory_path: Path to GCRI external_memory.json
            session_id: Optional session tag for ingested nodes

        Returns:
            Summary dict with counts of ingested items
        """
        import hashlib
        import json as json_mod
        from pathlib import Path

        gcri_path = Path(gcri_memory_path)
        if not gcri_path.exists():
            logger.warning(f'GCRI memory not found: {gcri_path}')
            return {'status': 'not_found', 'ingested': 0}

        try:
            with open(gcri_path, 'r', encoding='utf-8') as f:
                data = json_mod.load(f)
        except Exception as e:
            logger.error(f'Failed to read GCRI memory: {e}')
            return {'status': 'error', 'error': str(e), 'ingested': 0}

        existing_hashes = self._get_gcri_hashes()
        ingested = 0

        for rule in data.get('global_rules', []):
            rule_hash = hashlib.md5(rule.encode()).hexdigest()[:12]
            if rule_hash in existing_hashes:
                continue
            node = self._create_gcri_node(
                summary=f'[GCRI Rule] {rule[:120]}',
                trigger='When solving tasks that require learned constraints or best practices',
                raw_content=rule,
                tags=['gcri', 'rule', 'global'],
                source_hash=rule_hash,
                session_id=session_id,
            )
            if node:
                ingested += 1

        for domain, rules in data.get('domain_rules', {}).items():
            for rule in rules:
                rule_hash = hashlib.md5(f'{domain}:{rule}'.encode()).hexdigest()[:12]
                if rule_hash in existing_hashes:
                    continue
                node = self._create_gcri_node(
                    summary=f'[GCRI Rule/{domain}] {rule[:120]}',
                    trigger=f'When solving {domain}-related tasks',
                    raw_content=rule,
                    tags=['gcri', 'rule', domain],
                    source_hash=rule_hash,
                    session_id=session_id,
                )
                if node:
                    ingested += 1

        for domain, entries in data.get('knowledge', {}).items():
            for entry in entries:
                title = entry.get('title', '')
                content = entry.get('content', '')
                code = entry.get('code', '')
                knowledge_hash = hashlib.md5(f'{domain}:{title}'.encode()).hexdigest()[:12]
                if knowledge_hash in existing_hashes:
                    continue
                raw_parts = [f'# {title}\n\n{content}']
                if code:
                    raw_parts.append(f'\n```\n{code}\n```')
                entry_tags = ['gcri', 'knowledge', domain]
                entry_tags.extend(entry.get('tags', []))
                node = self._create_gcri_node(
                    summary=f'[GCRI Knowledge/{domain}] {title}: {content[:100]}',
                    trigger=f'When needing {entry.get("type", "knowledge")} about {title}',
                    raw_content='\n'.join(raw_parts),
                    tags=entry_tags,
                    source_hash=knowledge_hash,
                    session_id=session_id,
                )
                if node:
                    ingested += 1

        summary = {'status': 'done', 'ingested': ingested}
        if ingested > 0 and self._vector_index:
            deduped, _ = self._dedup([
                e['node_id'] for e in self._store.list_all()
                if 'gcri' in e.get('topic_tags', [])
            ])
            summary['deduped'] = deduped

        logger.info(f'GCRI memory ingestion: {summary}')
        return summary

    def _create_gcri_node(
        self, summary: str, trigger: str, raw_content: str,
        tags: list[str], source_hash: str, session_id: Optional[str] = None,
    ) -> Optional[MemoryNode]:
        """Create a single CoMeT node from GCRI memory entry."""
        try:
            node_id = self._store.generate_node_id()
            content_key = self._store.generate_content_key(prefix='gcri')
            raw_location = self._store.save_raw(content_key, raw_content)
            node = MemoryNode(
                node_id=node_id,
                session_id=session_id,
                depth_level=0,
                recall_mode='active',
                topic_tags=tags + [f'gcri_hash:{source_hash}'],
                summary=summary,
                trigger=trigger,
                content_key=content_key,
                raw_location=raw_location,
            )
            self._store.save_node(node)
            if self._vector_index:
                self._vector_index.upsert(node, raw_content=raw_content[:8000])
            return node
        except Exception as e:
            logger.warning(f'Failed to create GCRI node: {e}')
            return None

    def _get_gcri_hashes(self) -> set[str]:
        """Get set of already-ingested GCRI source hashes."""
        hashes = set()
        for entry in self._store.list_all():
            for tag in entry.get('topic_tags', []):
                if tag.startswith('gcri_hash:'):
                    hashes.add(tag[10:])
        return hashes
