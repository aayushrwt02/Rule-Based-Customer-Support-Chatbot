"""Hybrid retrieval pipeline with caching and fast-path optimization."""

import keyword
import re
from dataclasses import dataclass
from typing import Any

from src.database.chroma_manager import ChromaManager
from src.embeddings.embedding_model import EmbeddingModel
from src.search.keyword_search import KeywordSearch
from src.search.query_mapper import QueryMapper
from src.search.regex_search import RegexSearchEngine
from src.search.semantic_search import SemanticSearch
from src.utils.config import (
    FAQ_MATCH_BOOST,
    KEYWORD_FAST_PATH_THRESHOLD,
    KEYWORD_WEIGHT,
    MIN_RELEVANCE_SCORE,
    REGEX_WEIGHT,
    SECTION_EXACT_MATCH_BOOST,
    SECTION_TITLE_MATCH_BOOST,
    SEMANTIC_WEIGHT,
    TOP_K_RESULTS,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

QUERY_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9]+\b")
MULTI_TOPIC_RE = re.compile(r"\b(?:and|also|plus)\b", re.IGNORECASE)


@dataclass
class RetrievalResult:
    """A ranked retrieval result with combined scoring."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    combined_score: float
    keyword_score: float
    regex_score: float
    semantic_score: float
    search_methods: list[str]


class HybridRetriever:
    """
    Hybrid retrieval: keyword → regex → semantic.

    Caches indexed documents in memory for faster repeated queries.
    Skips semantic search when keyword or section-title match is very strong.
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.chroma_manager = chroma_manager
        self.embedding_model = embedding_model
        self.keyword_search = KeywordSearch()
        self.regex_search = RegexSearchEngine()
        self.semantic_search = SemanticSearch(chroma_manager, embedding_model)
        self._cached_documents: list[dict[str, Any]] | None = None

    def _get_documents(self) -> list[dict[str, Any]]:
        """Load and cache all documents with precomputed lowercase text."""
        if self._cached_documents is None:
            raw_documents = self.chroma_manager.get_all_documents()
            self._cached_documents = []
            for doc in raw_documents:
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                section_title = str(metadata.get("section_title", ""))
                self._cached_documents.append(
                    {
                        **doc,
                        "text_lower": text.lower(),
                        "section_title_lower": section_title.lower(),
                    }
                )
            logger.info(
                "Cached %d document chunk(s) for retrieval.", len(self._cached_documents)
            )
        return self._cached_documents

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[RetrievalResult]:
        if not query.strip():
            return []

        query_lower = query.lower()
        if QueryMapper.is_out_of_scope_query(query_lower):
            logger.info("Out-of-scope query rejected: %s", query[:60])
            return []

        all_documents = self._get_documents()
        if not all_documents:
            logger.warning("No documents in the database.")
            return []

        query_lower = query.lower()
        query_tokens = self._extract_query_tokens(query_lower)
        target_sections = self._resolve_target_sections(query_lower, query_tokens)
        search_pool = self._narrow_search_pool(all_documents, target_sections)

        keyword_results = self.keyword_search.search(
            query, search_pool, top_k=top_k, query_tokens=query_tokens
        )
        regex_results = self.regex_search.search(query, search_pool, top_k=top_k)

        best_keyword = keyword_results[0]["score"] if keyword_results else 0.0
        section_fast_path = self._section_title_fast_path(
            query_lower, query_tokens, target_sections, all_documents, top_k
        )

        if section_fast_path:
            logger.debug("Fast path: direct section-title match for '%s'.", query[:50])
            return section_fast_path

        if best_keyword >= KEYWORD_FAST_PATH_THRESHOLD:
            logger.debug(
                "Fast path: strong keyword match (%.2f), skipping semantic.", best_keyword
            )
            semantic_results: list[dict[str, Any]] = []
        else:
            semantic_results = self.semantic_search.search(
                query, top_k=top_k, section_titles=target_sections or None
            )

        merged = self._merge_results(keyword_results, regex_results, semantic_results)
        ranked = self._rank_results(merged, query, query_tokens, target_sections)
        filtered = [
            r for r in ranked 
            if (
                r.combined_score >= MIN_RELEVANCE_SCORE
                and (
                    r.keyword_score > 0 
                    or r.regex_score > 0 
                    or r.semantic_score >= 0.55
                )
            )
        ]

        # Reject weak semantic-only matches
        filtered = [
            r for r in filtered
            if not (
                r.semantic_score > 0
                and r.keyword_score == 0 
                and r.regex_score == 0
                and r.semantic_score < 0.55
            )
        ]

        logger.info(
            "Retrieval: %d keyword, %d regex, %d semantic -> %d result(s).",
            len(keyword_results),
            len(regex_results),
            len(semantic_results),
            len(filtered),
        )

        # No valid result found
        if not filtered:
            return []
        
        best = filtered[0]
        
        # Reject weak semantic-only answer
        if (
            best.semantic_score > 0
            and best.keyword_score == 0
            and best.regex_score == 0
            and best.semantic_score < 0.60
        ):
            logger.info("Rejected weak semantic-only retrieval.")
            return []
        
        # Reject overall low-confidence answer
        if best.combined_score < 0.45:
            logger.info("Rejected low-confidence retrieval.")
            return []
        
        return filtered[:top_k]

    @staticmethod
    def _extract_query_tokens(query_lower: str) -> set[str]:
        return {token.lower() for token in QUERY_TOKEN_RE.findall(query_lower)}

    @staticmethod
    def _narrow_search_pool(
        all_documents: list[dict[str, Any]],
        target_sections: list[str],
    ) -> list[dict[str, Any]]:
        """Restrict keyword/regex search to likely sections when intent is clear."""
        if not target_sections or len(target_sections) > 3:
            return all_documents

        target_lowers = {section.lower() for section in target_sections}
        filtered = [
            doc
            for doc in all_documents
            if doc.get("section_title_lower") in target_lowers
        ]
        return filtered if filtered else all_documents

    def _section_title_fast_path(
        self,
        query_lower: str,
        query_tokens: set[str],
        target_sections: list[str],
        all_documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[RetrievalResult] | None:
        """Return results immediately when the query maps to known section titles."""
        if not target_sections:
            return None

        if len(target_sections) == 1 and not MULTI_TOPIC_RE.search(query_lower):
            target = target_sections[0].lower()
            matches = [
                doc
                for doc in all_documents
                if doc.get("section_title_lower") == target
            ]
            if len(matches) == 1:
                doc = matches[0]
                if not self._fast_path_is_valid(doc, query_lower, query_tokens, target_sections):
                    return None
                return [
                    RetrievalResult(
                        chunk_id=doc["chunk_id"],
                        text=doc["text"],
                        metadata=doc.get("metadata", {}),
                        combined_score=1.0,
                        keyword_score=0.95,
                        regex_score=0.0,
                        semantic_score=0.0,
                        search_methods=["section_title"],
                    )
                ]

            if matches:
                best = QueryMapper.pick_best_document(
                    matches, query_lower, query_tokens, target_sections
                )
                if best is None or not self._fast_path_is_valid(
                    best, query_lower, query_tokens, target_sections
                ):
                    return None
                return [
                    RetrievalResult(
                        chunk_id=best["chunk_id"],
                        text=best["text"],
                        metadata=best.get("metadata", {}),
                        combined_score=0.98,
                        keyword_score=0.93,
                        regex_score=0.0,
                        semantic_score=0.0,
                        search_methods=["section_title"],
                    )
                ]

        if len(target_sections) >= 2:
            if not MULTI_TOPIC_RE.search(query_lower):
                best_doc = QueryMapper.pick_best_document(
                    all_documents,
                    query_lower,
                    query_tokens,
                    target_sections,
                )
                if (
                    best_doc
                    and best_doc.get("section_title_lower")
                    in {t.lower() for t in target_sections}
                    and self._fast_path_is_valid(
                        best_doc, query_lower, query_tokens, target_sections
                    )
                ):
                    return [
                        RetrievalResult(
                            chunk_id=best_doc["chunk_id"],
                            text=best_doc["text"],
                            metadata=best_doc.get("metadata", {}),
                            combined_score=0.98,
                            keyword_score=0.93,
                            regex_score=0.0,
                            semantic_score=0.0,
                            search_methods=["section_title"],
                        )
                    ]

            results: list[RetrievalResult] = []
            for target in target_sections[:top_k]:
                target_lower = target.lower()
                section_docs = [
                    doc
                    for doc in all_documents
                    if doc.get("section_title_lower") == target_lower
                ]
                if not section_docs:
                    continue

                doc = QueryMapper.pick_best_document(
                    section_docs,
                    query_lower,
                    query_tokens,
                    target_sections,
                )
                if doc is None:
                    continue
                results.append(
                    RetrievalResult(
                        chunk_id=doc["chunk_id"],
                        text=doc["text"],
                        metadata=doc.get("metadata", {}),
                        combined_score=0.97,
                        keyword_score=0.92,
                        regex_score=0.0,
                        semantic_score=0.0,
                        search_methods=["section_title"],
                    )
                )

            if len(results) >= 2:
                return results[:top_k]

            if results:
                return results

        return None

    @staticmethod
    def _fast_path_is_valid(
        doc: dict[str, Any],
        query_lower: str,
        query_tokens: set[str],
        target_sections: list[str],
    ) -> bool:
        """Ensure fast-path section matches query intent and content relevance."""
        section_title_lower = doc.get("section_title_lower", "")
        if not section_title_lower:
            return False

        priority = QueryMapper.section_priority_score(
            section_title_lower,
            target_sections,
            query_lower,
            query_tokens,
        )
        relevance = QueryMapper.chunk_topic_relevance(
            doc.get("text_lower", ""),
            query_lower,
            query_tokens,
        )

        if QueryMapper._resolve_compound_phrases(query_lower):
            return priority >= 0.35 and relevance >= 0.20

        if len(target_sections) == 1:
            return priority >= 0.30

        return priority >= 0.40 and relevance >= 0.15

    @staticmethod
    def _section_content_score(
        doc: dict[str, Any],
        query_lower: str,
        query_tokens: set[str] | None = None,
    ) -> float:
        if query_tokens is None:
            query_tokens = HybridRetriever._extract_query_tokens(query_lower)
        return QueryMapper.chunk_topic_relevance(
            doc.get("text_lower", ""),
            query_lower,
            query_tokens,
        )

    def _merge_results(
        self,
        keyword_results: list[dict[str, Any]],
        regex_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for result_list, weight_key in [
            (keyword_results, "keyword_score"),
            (regex_results, "regex_score"),
            (semantic_results, "semantic_score"),
        ]:
            for result in result_list:
                chunk_id = result["chunk_id"]
                if chunk_id not in merged:
                    merged[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": result["text"],
                        "metadata": result.get("metadata", {}),
                        "keyword_score": 0.0,
                        "regex_score": 0.0,
                        "semantic_score": 0.0,
                        "search_methods": [],
                    }

                merged[chunk_id][weight_key] = max(
                    merged[chunk_id][weight_key], result.get("score", 0.0)
                )
                search_type = result.get("search_type", weight_key.replace("_score", ""))
                if search_type not in merged[chunk_id]["search_methods"]:
                    merged[chunk_id]["search_methods"].append(search_type)

        return merged

    def _rank_results(
        self,
        merged: dict[str, dict[str, Any]],
        query: str,
        query_tokens: set[str],
        target_sections: list[str] | None = None,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        query_lower = query.lower()
        if target_sections is None:
            target_sections = self._resolve_target_sections(query_lower, query_tokens)

        for data in merged.values():
            combined = (
                data["keyword_score"] * KEYWORD_WEIGHT
                + data["regex_score"] * REGEX_WEIGHT
                + data["semantic_score"] * SEMANTIC_WEIGHT
            )

            # Penalize semantic-only matches
            if (
                data["semantic_score"] > 0
                and data["keyword_score"] == 0
                and data["regex_score"] == 0
            ):
                combined *= 0.75

            metadata = data.get("metadata", {})
            section_type = str(metadata.get("section_type", ""))
            section_title = str(metadata.get("section_title", ""))
            section_title_lower = section_title.lower()

            if section_type == "faq" and any(
                w in query_lower for w in ("how", "what", "when", "who", "many", "apply")
            ):
                combined += FAQ_MATCH_BOOST
                combined += self._faq_match_boost(data["text"], query_lower)

            if section_title_lower and any(
                token in section_title_lower for token in query_tokens if len(token) > 2
            ):
                combined += 0.08

            if target_sections:
                combined += QueryMapper.section_priority_score(
                    section_title_lower,
                    target_sections,
                    query_lower,
                    query_tokens,
                ) * 0.25

                for target in target_sections:
                    if section_title_lower == target.lower():
                        combined += SECTION_TITLE_MATCH_BOOST
                        if len(target_sections) == 1:
                            combined += SECTION_EXACT_MATCH_BOOST

            combined += QueryMapper.chunk_topic_relevance(
                data.get("text", "").lower(),
                query_lower,
                query_tokens,
            ) * 0.12

            if section_type == "setup_step" and not any(
                w in query_tokens for w in ("step", "setup", "create", "add", "company", "warehouse", "product")
            ):
                combined *= 0.90

            results.append(
                RetrievalResult(
                    chunk_id=data["chunk_id"],
                    text=data["text"],
                    metadata=metadata,
                    combined_score=min(combined, 1.0),
                    keyword_score=data["keyword_score"],
                    regex_score=data["regex_score"],
                    semantic_score=data["semantic_score"],
                    search_methods=data["search_methods"],
                )
            )

        results.sort(
            key=lambda r: (
                r.combined_score, 
                r.keyword_score,
                r.regex_score,
                r.semantic_score,
            ),
            reverse=True
        )
        return results

    @classmethod
    def _resolve_target_sections(
        cls,
        query_lower: str,
        query_tokens: set[str],
    ) -> list[str]:
        """Map query tokens and intent phrases to official section titles."""
        return QueryMapper.resolve_target_sections(query_lower, query_tokens)

    @staticmethod
    def _faq_match_boost(text: str, query_lower: str) -> float:
        """Extra boost when FAQ content closely matches the user query."""
        for match in re.finditer(r"Q:\s*(.+?)\s+A:", text, re.IGNORECASE | re.DOTALL):
            faq_q = match.group(1).lower().strip()
            query_words = set(re.findall(r"\b[a-z]{3,}\b", query_lower))
            faq_words = set(re.findall(r"\b[a-z]{3,}\b", faq_q))
            if not query_words or not faq_words:
                continue
            overlap = len(query_words & faq_words) / len(query_words)
            if overlap >= 0.6:
                return 0.18
            if overlap >= 0.45:
                return 0.10

        if "frequently asked questions" in text.lower():
            query_words = set(re.findall(r"\b[a-z]{3,}\b", query_lower))
            text_words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
            if query_words:
                overlap = len(query_words & text_words) / len(query_words)
                if overlap >= 0.4:
                    return 0.12

        return 0.0
