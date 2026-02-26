"""CoMeTPro: CoMeT with automatic consolidation support."""
import threading
from typing import Optional

from loguru import logger

from comet.orchestrator import CoMeT
from comet.schemas import CognitiveLoad


class CoMeTPro(CoMeT):
    """Extended CoMeT with auto-consolidation.

    When config.consolidation.auto_enabled is True, consolidation
    is triggered automatically after a configurable number of
    compacted nodes accumulate in the session.
    """

    _AUTO_CONSOLIDATE_THRESHOLD = 10

    def __init__(self, config, session_id=None):
        super().__init__(config, session_id=session_id)
        self._nodes_since_consolidation = 0
        self._auto_consolidate_lock = threading.Lock()

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
        if not self.auto_enabled:
            return
        if not self._consolidator:
            return

        with self._auto_consolidate_lock:
            self._nodes_since_consolidation += 1
            if self._nodes_since_consolidation >= self._AUTO_CONSOLIDATE_THRESHOLD:
                self._nodes_since_consolidation = 0

        if self._nodes_since_consolidation == 0:
            try:
                result = self._consolidator.consolidate(self._session_node_ids)
                logger.info(f'Auto-consolidation triggered ({len(self._session_node_ids)} nodes): {result}')
            except Exception as e:
                logger.warning(f'Auto-consolidation failed: {e}')
