"""Semantic vector search using ChromaDB with query-embedding cache."""

import re
from collections import OrderedDict
from typing import Any

from src.database.chroma_manager import ChromaManager
from src.embeddings.embedding_model import EmbeddingModel
from src.search.query_mapper import QueryMapper
from src.utils.config import QUERY_EMBEDDING_CACHE_SIZE, TOP_K_RESULTS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SemanticSearch:
    """Perform semantic similarity search via ChromaDB embeddings."""

    def __init__(
        self,
        chroma_manager: ChromaManager,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.chroma_manager = chroma_manager
        self.embedding_model = embedding_model
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        section_titles: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for semantically similar document chunks."""
        query_embedding = self._get_query_embedding(query)
        results = self.chroma_manager.query_semantic(
            query_embedding, top_k=top_k, section_titles=section_titles
        )

        for result in results:
            result["search_type"] = "semantic"

        logger.debug("Semantic search returned %d result(s).", len(results))
        return results

    @staticmethod
    def _build_semantic_query(query: str) -> str:
        """Enrich the query with synonym and intent terms for better vector matching."""
        query_lower = query.lower()
        tokens = set(re.findall(r"\b[a-z0-9]{2,}\b", query_lower))
        expanded = QueryMapper.expanded_query_terms(query_lower, tokens)
        extra_terms = sorted(term for term in expanded if term not in tokens and len(term) > 2)
        target_sections = QueryMapper.resolve_target_sections(query_lower, tokens)
        if not extra_terms and not target_sections:
            return query

        parts = [query.strip()]
        if extra_terms:
            parts.append(" ".join(extra_terms[:12]))
        if target_sections:
            parts.append(" ".join(target_sections[:3]))
        return " ".join(parts)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Return a cached query embedding when available."""
        semantic_query = self._build_semantic_query(query)
        normalized = semantic_query.strip().lower()
        if not normalized:
            return self.embedding_model.encode_query(query)

        cached = self._embedding_cache.get(normalized)
        if cached is not None:
            self._embedding_cache.move_to_end(normalized)
            return cached

        embedding = self.embedding_model.encode_query(semantic_query)
        self._embedding_cache[normalized] = embedding

        while len(self._embedding_cache) > QUERY_EMBEDDING_CACHE_SIZE:
            self._embedding_cache.popitem(last=False)

        return embedding
