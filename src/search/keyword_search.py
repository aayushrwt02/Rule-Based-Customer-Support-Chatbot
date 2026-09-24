"""Keyword search with synonym expansion and section-title boosting."""

import re
from typing import Any

from src.search.query_mapper import QueryMapper
from src.utils.config import FAQ_TOPIC_KEYWORDS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "what", "when",
        "where", "why", "how", "which", "who", "this", "that", "these",
        "those", "and", "or", "but", "if", "about", "tell", "please", "also",
        "many", "much", "get", "apply", "available",
    }
)


class KeywordSearch:
    """Search document chunks by keyword frequency with synonym support."""

    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        query_tokens: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(query)
        if not keywords or not documents:
            return []

        if query_tokens is None:
            query_tokens = set(keywords)

        expanded = self._expand_with_synonyms(set(keywords))
        results: list[dict[str, Any]] = []

        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue

            metadata = doc.get("metadata", {})
            section_title = str(metadata.get("section_title", "")).lower()
            text_lower = doc.get("text_lower") or text.lower()

            score = self._calculate_score(
                expanded, text_lower, section_title, query, query_tokens
            )

            if score > 0:
                results.append(
                    {
                        "chunk_id": doc["chunk_id"],
                        "text": doc["text"],
                        "metadata": metadata,
                        "score": score,
                        "search_type": "keyword",
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _extract_keywords(self, query: str) -> list[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def _expand_with_synonyms(self, keywords: set[str]) -> set[str]:
        return QueryMapper.expand_query_keywords(keywords)

    def _calculate_score(
        self,
        keywords: set[str],
        text_lower: str,
        section_title: str,
        query: str,
        query_tokens: set[str],
    ) -> float:
        if not keywords:
            return 0.0

        query_lower = query.lower()
        merged_synonyms = QueryMapper.merged_section_synonyms()
        total_matches = 0
        matched_keywords = 0

        for keyword in keywords:
            count = len(
                re.findall(r"\b" + re.escape(keyword) + r"\w*\b", text_lower)
            )
            if count > 0:
                matched_keywords += 1
                total_matches += count

        if matched_keywords == 0:
            return 0.0

        keyword_coverage = matched_keywords / len(keywords)
        frequency_score = min(total_matches / (len(keywords) * 3), 1.0)
        score = (keyword_coverage * 0.55) + (frequency_score * 0.45)

        if section_title:
            title_matches = sum(1 for kw in keywords if kw in section_title)
            if title_matches > 0:
                score += min(title_matches * 0.28, 0.55)

        for keyword, mapped_title in sorted(
            merged_synonyms.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if (keyword in query_lower or keyword in query_tokens) and mapped_title.lower() == section_title:
                score += 0.30

        for topic in QueryMapper.matched_faq_topics(query_lower):
            topic_keywords = FAQ_TOPIC_KEYWORDS.get(topic, [])
            query_has_topic = any(kw in query_lower for kw in topic_keywords)
            text_has_topic = any(kw in text_lower for kw in topic_keywords)
            if query_has_topic and text_has_topic:
                score += 0.22
                break

        if "gst" in query_tokens and any(
            kw in text_lower for kw in ("gst", "cgst", "sgst", "igst", "gstr", "itc")
        ):
            score += 0.18

        if "payment" in query_tokens or "upi" in query_tokens:
            if any(kw in text_lower for kw in ("payment", "upi", "cash", "cheque", "qr")):
                score += 0.18

        if "inventory" in query_tokens or "warehouse" in query_tokens:
            if any(kw in text_lower for kw in ("inventory", "stock", "warehouse", "barcode")):
                score += 0.15

        if any(w in query_lower for w in ("how", "what", "when", "who", "many", "apply")):
            for match in re.finditer(r"Q:\s*(.+?)\s+A:", text_lower, re.IGNORECASE):
                faq_q = match.group(1)
                faq_words = set(re.findall(r"\b[a-z]{3,}\b", faq_q))
                query_words = set(re.findall(r"\b[a-z]{3,}\b", query_lower))
                if query_words and faq_words:
                    overlap = len(query_words & faq_words) / len(query_words)
                    if overlap >= 0.45:
                        score += 0.35
                        break

            if "frequently asked questions" in text_lower:
                query_words = set(re.findall(r"\b[a-z]{3,}\b", query_lower))
                text_words = set(re.findall(r"\b[a-z]{3,}\b", text_lower))
                if query_words:
                    overlap = len(query_words & text_words) / len(query_words)
                    if overlap >= 0.35:
                        score += 0.25

        return min(score, 1.0)
