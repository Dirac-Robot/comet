"""VectorIndex: LanceDB-backed triple-table embedding store.

Three Lance tables (summary / trigger / raw) hold parallel embeddings for
each MemoryNode. Disk-resident via mmap so RAM usage stays bounded even
with hundreds of thousands of nodes.

The public surface (method names, ScoredResult shape, distance semantics)
matches the previous Chroma-backed implementation so that consumers
(Retriever, Consolidator, Compacter, CoMeT orchestrator, and CoBrA tools)
require no changes when this module is replaced.
"""
import os
import threading
from typing import Callable, Optional

import lancedb
import pyarrow as pa
from ato.adict import ADict
from loguru import logger
from pydantic import BaseModel, Field

from comet.llm_factory import create_embeddings
from comet.schemas import MemoryNode


class ScoredResult(BaseModel):
    node_id: str
    score: float = Field(description='Cosine distance score (lower = closer; 0=identical, 2=opposite)')
    rank: int = 0


_SUMMARY_TABLE = 'comet_summaries'
_TRIGGER_TABLE = 'comet_triggers'
_RAW_TABLE = 'comet_raw'


def _build_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field('id', pa.string()),
        pa.field('vector', pa.list_(pa.float32(), dim)),
        pa.field('document', pa.string()),
        pa.field('recall_mode', pa.string()),
        pa.field('topic_tags', pa.string()),
        pa.field('created_at', pa.string()),
    ])


def _escape_id(node_id: str) -> str:
    """SQL string escape — doubles single quotes."""
    return node_id.replace("'", "''")


def _ids_in_clause(ids: list[str]) -> str:
    quoted = ', '.join(f"'{_escape_id(i)}'" for i in ids)
    return f'id IN ({quoted})'


class VectorIndex:
    """Triple-table vector store using LanceDB.

    Tables:
    - comet_summaries: embeds MemoryNode.summary
    - comet_triggers:  embeds MemoryNode.trigger
    - comet_raw:       embeds raw content (fallback / fusion path)
    """

    def __init__(self, config: ADict):
        self._config = config
        self._write_lock = threading.Lock()
        retrieval = config.retrieval
        self._db_path = retrieval.vector_db_path
        os.makedirs(self._db_path, exist_ok=True)

        self._embed_fn: Callable[[list[str]], list[list[float]]] = create_embeddings(config)
        self._embedding_dim: Optional[int] = None

        self._db = lancedb.connect(self._db_path)
        self._summary_table = None
        self._trigger_table = None
        self._raw_table = None
        self._init_tables()

    # ── Table lifecycle ──

    def _detect_dim(self) -> int:
        """One-shot probe of the embedding model to fix the table schema."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        probe = self._embed_fn(['__dim_probe__'])[0]
        self._embedding_dim = len(probe)
        return self._embedding_dim

    def _open_or_create(self, name: str) -> 'lancedb.table.Table':
        existing = set(self._db.table_names())
        if name in existing:
            return self._db.open_table(name)
        dim = self._detect_dim()
        schema = _build_schema(dim)
        return self._db.create_table(name, schema=schema, mode='create')

    def _init_tables(self):
        self._summary_table = self._open_or_create(_SUMMARY_TABLE)
        self._trigger_table = self._open_or_create(_TRIGGER_TABLE)
        self._raw_table = self._open_or_create(_RAW_TABLE)

    def _embed(self, text: str) -> list[float]:
        vec = self._embed_fn([text])[0]
        if self._embedding_dim is None:
            self._embedding_dim = len(vec)
        return vec

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._embed_fn(texts)
        if self._embedding_dim is None and vecs:
            self._embedding_dim = len(vecs[0])
        return vecs

    @staticmethod
    def _build_metadata(node: MemoryNode) -> dict:
        return {
            'recall_mode': node.recall_mode,
            'topic_tags': ','.join(node.topic_tags),
            'created_at': node.created_at.isoformat(),
        }

    @staticmethod
    def _row(node_id: str, vec: list[float], document: str, meta: dict) -> dict:
        return {
            'id': node_id,
            'vector': vec,
            'document': document or '',
            'recall_mode': meta['recall_mode'],
            'topic_tags': meta['topic_tags'],
            'created_at': meta['created_at'],
        }

    @staticmethod
    def _upsert_rows(table, rows: list[dict]):
        if not rows:
            return
        (
            table
            .merge_insert('id')
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    # ── Write API ──

    def upsert(self, node: MemoryNode, raw_content: str = ''):
        if self._db is None:
            # close() was called — table refs are None and the LanceDB
            # connection is gone. Stale callers (e.g. dream / brief regen
            # that captured this VectorIndex before reset) must observe
            # a no-op rather than crash with NoneType.merge_insert.
            return
        embed_texts = [node.summary, node.trigger]
        if raw_content:
            embed_texts.append(raw_content[:8000])
        vecs = self._embed_batch(embed_texts)
        summary_vec, trigger_vec = vecs[0], vecs[1]
        raw_vec = vecs[2] if raw_content else None

        meta = self._build_metadata(node)
        with self._write_lock:
            self._upsert_rows(
                self._summary_table,
                [self._row(node.node_id, summary_vec, node.summary, meta)],
            )
            self._upsert_rows(
                self._trigger_table,
                [self._row(node.node_id, trigger_vec, node.trigger, meta)],
            )
            if raw_vec is not None:
                self._upsert_rows(
                    self._raw_table,
                    [self._row(node.node_id, raw_vec, raw_content, meta)],
                )
        logger.debug(f'VectorIndex: upserted {node.node_id}')

    def upsert_batch(self, nodes: list[MemoryNode], raw_contents: Optional[list[str]] = None):
        if not nodes:
            return
        if self._db is None:
            return

        ids = [n.node_id for n in nodes]
        summaries = [n.summary for n in nodes]
        triggers = [n.trigger for n in nodes]
        metas = [self._build_metadata(n) for n in nodes]

        summary_vecs = self._embed_batch(summaries)
        trigger_vecs = self._embed_batch(triggers)

        with self._write_lock:
            self._upsert_rows(
                self._summary_table,
                [self._row(nid, v, doc, m) for nid, v, doc, m in zip(ids, summary_vecs, summaries, metas)],
            )
            self._upsert_rows(
                self._trigger_table,
                [self._row(nid, v, doc, m) for nid, v, doc, m in zip(ids, trigger_vecs, triggers, metas)],
            )

            if raw_contents:
                truncated = [(r or '')[:8000] for r in raw_contents]
                raw_vecs = self._embed_batch(truncated)
                self._upsert_rows(
                    self._raw_table,
                    [self._row(nid, v, doc, m) for nid, v, doc, m in zip(ids, raw_vecs, truncated, metas)],
                )

        logger.info(f'VectorIndex: batch upserted {len(nodes)} nodes')

    # ── Read API ──

    @staticmethod
    def _search_table(table, query_vec: list[float], top_k: int) -> list['ScoredResult']:
        if table is None:
            return []
        try:
            total = table.count_rows()
        except Exception as e:
            logger.warning(f'VectorIndex: count_rows failed: {e}')
            return []
        if total == 0:
            return []
        try:
            rows = (
                table
                .search(query_vec)
                .distance_type('cosine')
                .limit(min(top_k, total))
                .to_list()
            )
        except Exception as e:
            logger.warning(f'VectorIndex: search failed on {table.name}: {e}')
            return []
        return [
            ScoredResult(
                node_id=r['id'],
                score=float(r.get('_distance', 0.0)),
                rank=rank,
            )
            for rank, r in enumerate(rows)
        ]

    def search_by_summary(self, query: str, top_k: int = 10) -> list[ScoredResult]:
        query_vec = self._embed(query)
        return self._search_table(self._summary_table, query_vec, top_k)

    def search_by_trigger(self, query: str, top_k: int = 10) -> list[ScoredResult]:
        query_vec = self._embed(query)
        return self._search_table(self._trigger_table, query_vec, top_k)

    def search_by_raw(self, query: str, top_k: int = 10) -> list[ScoredResult]:
        query_vec = self._embed(query)
        return self._search_table(self._raw_table, query_vec, top_k)

    def get_raw(self, node_id: str) -> Optional[str]:
        """Retrieve raw content document by node_id from the raw table."""
        if self._raw_table is None:
            return None
        try:
            if self._raw_table.count_rows() == 0:
                return None
            rows = (
                self._raw_table
                .search()
                .where(f"id = '{_escape_id(node_id)}'")
                .select(['id', 'document'])
                .limit(1)
                .to_list()
            )
            if rows:
                return rows[0].get('document') or None
        except Exception as e:
            logger.debug(f'VectorIndex.get_raw failed for {node_id}: {e}')
        return None

    def has_similar(self, text: str, threshold: float = 0.08) -> Optional[str]:
        """Cosine-distance duplicate check against the raw table.

        threshold: distance ceiling. 0.08 ≈ 96% similar.
        Returns matching node_id or None.
        """
        if self._raw_table is None:
            return None
        try:
            if self._raw_table.count_rows() == 0:
                return None
            query_vec = self._embed(text[:8000])
            rows = (
                self._raw_table
                .search(query_vec)
                .distance_type('cosine')
                .limit(1)
                .to_list()
            )
        except Exception as e:
            logger.debug(f'VectorIndex.has_similar failed: {e}')
            return None
        if not rows:
            return None
        top = rows[0]
        if float(top.get('_distance', 1.0)) <= threshold:
            return top.get('id')
        return None

    @property
    def count(self) -> int:
        if self._summary_table is None:
            return 0
        try:
            return self._summary_table.count_rows()
        except Exception as e:
            logger.warning(f'VectorIndex: count failed: {e}')
            return 0

    def delete(self, node_id: str):
        clause = f"id = '{_escape_id(node_id)}'"
        with self._write_lock:
            for table in (self._summary_table, self._trigger_table, self._raw_table):
                if table is None:
                    continue
                try:
                    table.delete(clause)
                except Exception as e:
                    logger.debug(f'VectorIndex.delete on {table.name}: {e}')

    def close(self):
        """Release LanceDB references for clean shutdown.

        LanceDB has no explicit close() — clearing refs lets the GC release
        any held file descriptors.
        """
        with self._write_lock:
            self._summary_table = None
            self._trigger_table = None
            self._raw_table = None
            self._db = None

    def reset(self):
        """Drop and recreate all three tables. In-process state remains valid."""
        if self._db is None:
            logger.warning('VectorIndex.reset() called after close(), skipping')
            return
        with self._write_lock:
            existing = set(self._db.table_names())
            for name in (_SUMMARY_TABLE, _TRIGGER_TABLE, _RAW_TABLE):
                if name in existing:
                    try:
                        self._db.drop_table(name)
                    except Exception as e:
                        logger.warning(f'VectorIndex.reset: drop_table({name}) failed: {e}')
            self._init_tables()

    # ── Curator helpers (consumed by CoBrA) ──

    def rerank_by_query(
        self,
        query: str,
        candidate_ids: list[str],
        weights: tuple[float, float] = (0.6, 0.4),
    ) -> list[tuple[str, float]]:
        """Rank candidate node_ids by similarity to query.

        Combines weighted similarity over summary and trigger embeddings.
        Returns list of (node_id, similarity_score) sorted descending.
        similarity is in [0.0, 1.0] (1.0 = identical).

        Used by CoBrA's _rank_candidates to keep query-time reranking
        out of the consumer side.
        """
        if not candidate_ids:
            return []

        try:
            query_vec = self._embed(query)
        except Exception as e:
            logger.warning(f'VectorIndex.rerank_by_query embed failed: {e}')
            return [(cid, 0.0) for cid in candidate_ids]

        sum_map = self._fetch_vectors(self._summary_table, candidate_ids)
        trg_map = self._fetch_vectors(self._trigger_table, candidate_ids)

        w_sum, w_trg = weights
        scored: list[tuple[str, float]] = []
        for cid in candidate_ids:
            s_emb = sum_map.get(cid)
            t_emb = trg_map.get(cid)
            s_sim = _cosine_sim(query_vec, s_emb) if s_emb is not None else 0.0
            t_sim = _cosine_sim(query_vec, t_emb) if t_emb is not None else 0.0
            scored.append((cid, w_sum * s_sim + w_trg * t_sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _fetch_vectors(table, ids: list[str]) -> dict[str, list[float]]:
        if table is None or not ids:
            return {}
        try:
            if table.count_rows() == 0:
                return {}
            rows = (
                table
                .search()
                .where(_ids_in_clause(ids))
                .select(['id', 'vector'])
                .limit(len(ids))
                .to_list()
            )
        except Exception as e:
            logger.warning(f'VectorIndex._fetch_vectors on {table.name}: {e}')
            return {}
        out: dict[str, list[float]] = {}
        for r in rows:
            v = r.get('vector')
            if v is None:
                continue
            try:
                out[r['id']] = list(v)
            except Exception:
                continue
        return out

    def healthcheck(self) -> bool:
        """Verify the vector store is reachable. Returns False on failure."""
        if self._db is None:
            return False
        try:
            _ = self._db.table_names()
            return True
        except Exception as e:
            logger.warning(f'VectorIndex healthcheck failed: {e}')
            return False


def _cosine_sim(a, b) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 on degenerate input."""
    if a is None or b is None:
        return 0.0
    try:
        if len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            xf = float(x)
            yf = float(y)
            dot += xf * yf
            na += xf * xf
            nb += yf * yf
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))
    except Exception:
        return 0.0
