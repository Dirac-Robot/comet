"""VectorIndex: Dual-collection embedding store using ChromaDB."""
from typing import Optional

import chromadb
from ato.adict import ADict
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from comet.schemas import MemoryNode


class ScoredResult(BaseModel):
    node_id: str
    score: float = Field(description='Distance score (lower = closer)')
    rank: int = 0


class VectorIndex:
    """Triple-collection vector store for summary, trigger, and raw embeddings.

    Maintains three ChromaDB collections:
    - summary_collection: embeds MemoryNode.summary
    - trigger_collection: embeds MemoryNode.trigger
    - raw_collection: embeds raw content (fallback search path)
    """

    def __init__(self, config: ADict):
        self._config = config
        retrieval = config.retrieval

        self._client = chromadb.PersistentClient(path=retrieval.vector_db_path)
        self._openai = OpenAI()
        self._embed_model = retrieval.embedding_model

        self._summary_col = self._client.get_or_create_collection(
            name='comet_summaries',
            metadata={'hnsw:space': 'cosine'},
        )
        self._trigger_col = self._client.get_or_create_collection(
            name='comet_triggers',
            metadata={'hnsw:space': 'cosine'},
        )
        self._raw_col = self._client.get_or_create_collection(
            name='comet_raw',
            metadata={'hnsw:space': 'cosine'},
        )

    def _embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            model=self._embed_model,
            input=text,
        )
        return response.data[0].embedding

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._openai.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def upsert(self, node: MemoryNode, raw_content: str = ''):
        texts = [node.summary, node.trigger]
        if raw_content:
            texts.append(raw_content)
        vecs = self._embed_batch(texts)
        summary_vec, trigger_vec = vecs[0], vecs[1]
        raw_vec = vecs[2] if raw_content else None

        metadata = {
            'recall_mode': node.recall_mode,
            'topic_tags': ','.join(node.topic_tags),
            'created_at': node.created_at.isoformat(),
        }

        self._summary_col.upsert(
            ids=[node.node_id],
            embeddings=[summary_vec],
            metadatas=[metadata],
            documents=[node.summary],
        )
        self._trigger_col.upsert(
            ids=[node.node_id],
            embeddings=[trigger_vec],
            metadatas=[metadata],
            documents=[node.trigger],
        )
        if raw_vec is not None:
            self._raw_col.upsert(
                ids=[node.node_id],
                embeddings=[raw_vec],
                metadatas=[metadata],
                documents=[raw_content[:1000]],
            )
        logger.debug(f'VectorIndex: upserted {node.node_id}')

    def upsert_batch(self, nodes: list[MemoryNode], raw_contents: Optional[list[str]] = None):
        if not nodes:
            return

        ids = [n.node_id for n in nodes]
        summaries = [n.summary for n in nodes]
        triggers = [n.trigger for n in nodes]
        metadatas = [
            {
                'topic_tags': ','.join(n.topic_tags),
                'created_at': n.created_at.isoformat(),
            }
            for n in nodes
        ]

        summary_vecs = self._embed_batch(summaries)
        trigger_vecs = self._embed_batch(triggers)

        self._summary_col.upsert(
            ids=ids,
            embeddings=summary_vecs,
            metadatas=metadatas,
            documents=summaries,
        )
        self._trigger_col.upsert(
            ids=ids,
            embeddings=trigger_vecs,
            metadatas=metadatas,
            documents=triggers,
        )

        if raw_contents:
            raw_vecs = self._embed_batch(raw_contents)
            self._raw_col.upsert(
                ids=ids,
                embeddings=raw_vecs,
                metadatas=metadatas,
                documents=[r[:1000] for r in raw_contents],
            )

        logger.info(f'VectorIndex: batch upserted {len(nodes)} nodes')

    def search_by_summary(self, query: str, top_k: int = 10) -> list[ScoredResult]:
        query_vec = self._embed(query)
        results = self._summary_col.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._summary_col.count() or 1),
        )
        return self._parse_results(results)

    def search_by_trigger(self, query: str, top_k: int = 10) -> list[ScoredResult]:
        query_vec = self._embed(query)
        results = self._trigger_col.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._trigger_col.count() or 1),
        )
        return self._parse_results(results)

    def search_by_raw(self, query: str, top_k: int = 10) -> list[ScoredResult]:
        if self._raw_col.count() == 0:
            return []
        query_vec = self._embed(query)
        results = self._raw_col.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._raw_col.count()),
        )
        return self._parse_results(results)

    @staticmethod
    def _parse_results(results: dict) -> list[ScoredResult]:
        parsed = []
        if not results['ids'] or not results['ids'][0]:
            return parsed

        ids = results['ids'][0]
        distances = results['distances'][0] if results.get('distances') else [0.0]*len(ids)

        for rank, (node_id, dist) in enumerate(zip(ids, distances)):
            parsed.append(ScoredResult(
                node_id=node_id,
                score=dist,
                rank=rank,
            ))
        return parsed

    @property
    def count(self) -> int:
        return self._summary_col.count()

    def delete(self, node_id: str):
        self._summary_col.delete(ids=[node_id])
        self._trigger_col.delete(ids=[node_id])
        try:
            self._raw_col.delete(ids=[node_id])
        except Exception:
            pass

    def reset(self):
        self._client.delete_collection('comet_summaries')
        self._client.delete_collection('comet_triggers')
        try:
            self._client.delete_collection('comet_raw')
        except Exception:
            pass
        self._summary_col = self._client.get_or_create_collection(
            name='comet_summaries',
            metadata={'hnsw:space': 'cosine'},
        )
        self._trigger_col = self._client.get_or_create_collection(
            name='comet_triggers',
            metadata={'hnsw:space': 'cosine'},
        )
        self._raw_col = self._client.get_or_create_collection(
            name='comet_raw',
            metadata={'hnsw:space': 'cosine'},
        )
