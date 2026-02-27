"""CoMeTPro: CoMeT with automatic consolidation support."""
import threading
from typing import Optional

from loguru import logger

from comet.orchestrator import CoMeT
from comet.schemas import CognitiveLoad


class CoMeTPro(CoMeT):
    """Extended CoMeT with auto-consolidation.

    When config.consolidation.auto_enabled is True:
    1. After each add(), sensor evaluates session summary redundancy
    2. If consolidation needed, runs consolidate() on session nodes
    3. Session node sequence is updated: absorbed IDs → keeper IDs
    4. get_session_context displays merged nodes at original positions
       with time-slot markers for duplicate appearances
    """

    def __init__(self, config, session_id=None, store=None, vector_index=None):
        super().__init__(config, session_id=session_id, store=store, vector_index=vector_index)
        self._node_sequence: list[str] = list(self._session_node_ids)

    @property
    def auto_enabled(self) -> bool:
        consolidation = getattr(self._config, 'consolidation', None)
        if consolidation is None:
            return False
        return bool(getattr(consolidation, 'auto_enabled', False))

    @auto_enabled.setter
    def auto_enabled(self, value: bool):
        from ato.adict import ADict
        if not hasattr(self._config, 'consolidation'):
            self._config.consolidation = ADict()
        self._config.consolidation.auto_enabled = bool(value)

    def _on_post_add(self, load: CognitiveLoad):
        if self._session_node_ids and self._session_node_ids[-1] not in self._node_sequence:
            self._node_sequence.append(self._session_node_ids[-1])

        if not self.auto_enabled:
            return
        if not self._consolidator:
            return

        summaries = self._get_session_summaries()
        if not summaries:
            return

        try:
            should = self._sensor.assess_consolidation_need(summaries)
        except Exception as e:
            logger.warning(f'Consolidation assessment failed: {e}')
            return

        if not should:
            return

        try:
            result = self.consolidate(self._session_node_ids)
            merge_map = result.get('merge_map', {})
            if merge_map:
                self._apply_merge_map(merge_map)
                logger.info(f'Auto-consolidation complete: merged {len(merge_map)} nodes')
        except Exception as e:
            logger.warning(f'Auto-consolidation failed: {e}')

    def _apply_merge_map(self, merge_map: dict[str, str]):
        """Update node sequence by replacing absorbed IDs with keeper IDs."""
        self._node_sequence = [
            merge_map.get(nid, nid) for nid in self._node_sequence
        ]
        self._session_node_ids = [
            nid for nid in self._session_node_ids
            if nid not in merge_map
        ]

    def get_session_context(self, session_id: Optional[str] = None, max_nodes: int = 50) -> str:
        """Get context with consolidated node ordering.

        After consolidation, merged nodes appear at every position
        their source nodes originally occupied. First appearance shows
        full summary + trigger; subsequent appearances show only the
        node_id as a time-slot marker.
        """
        if session_id and session_id != self._session_id:
            return super().get_session_context(session_id, max_nodes)

        sequence = self._node_sequence if self._node_sequence else self._session_node_ids
        if not sequence:
            return f'(No nodes for session {self._session_id})'

        seen = set()
        parts = []

        for pid in self._pinned_node_ids:
            if pid in seen:
                continue
            pnode = self._store.get_node(pid)
            if pnode is None:
                continue
            seen.add(pid)
            origin = ''
            for t in (pnode.topic_tags or []):
                if t.startswith('ORIGIN:'):
                    origin = f'({t}) '
                    break
            parts.append(f'[{pid}] {origin}(PIN) {pnode.summary} | {pnode.trigger}')

        for nid in sequence[:max_nodes]:
            if nid in seen:
                parts.append(f'[{nid}]')
                continue

            seen.add(nid)
            node = self._store.get_node(nid)
            if node is None:
                continue

            tags = node.topic_tags or []
            origin = ''
            for t in tags:
                if t.startswith('ORIGIN:'):
                    origin = f'({t}) '
                    break
            recall = getattr(node, 'recall_mode', 'active')
            prefix = '(passive) ' if recall in ('passive', 'both') else ''
            parts.append(f'[{nid}] {origin}{prefix}{node.summary} | {node.trigger}')

        return '\n'.join(parts)
