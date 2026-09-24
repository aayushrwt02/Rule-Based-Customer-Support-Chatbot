"""Unified query-to-document mapping using all configured synonym and intent tables."""

from __future__ import annotations

import re
from typing import Any

from src.preprocessing.section_registry import SectionRegistry
from src.utils.config import (
    FAQ_TOPIC_KEYWORDS,
    OUT_OF_SCOPE_KEYWORDS,
    QUERY_INTENT_SECTIONS,
    QUERY_SYNONYMS,
    SECTION_TITLE_SYNONYMS,
)

FAQ_TOPIC_TO_SECTIONS: dict[str, list[str]] = {
    "orders": ["Order Processing", "Product Catalogue", "Frequently Asked Questions"],
    "warehouse": ["Step 2: Create Warehouses", "Inventory Management", "Getting Started"],
    "gst": ["GST", "Tax module"],
    "accounting": ["Accounting", "Finance module"],
    "payments": ["Payments"],
    "getting_started": [
        "Getting Started",
        "Step 1: Create Your Company",
        "Frequently Asked Questions",
    ],
    "features": ["Key Features", "Frequently Asked Questions"],
    "security": ["Security"],
    "developer": ["Introduction", "About Direkt"],
    "purchase": ["Purchase Orders", "Order Processing"],
}

GENERIC_SECTIONS = frozenset(
    {"introduction", "about direkt", "frequently asked questions"}
)

# Bare tokens that must not alone drive section resolution
AMBIGUOUS_TOKENS = frozenset(
    {
        "management", "module", "modules", "order", "orders", "feature",
        "features", "processing", "about", "tell", "what", "how", "create",
    }
)

MODULE_TOPIC_ALIASES: dict[str, str] = {
    "sale": "sales",
    "sales": "sales",
    "inventory": "inventory",
    "finance": "finance",
    "financial": "finance",
    "tax": "tax",
    "ai": "ai",
    "gst": "tax",
}

MANAGEMENT_TOPIC_SECTIONS: dict[str, list[str]] = {
    "inventory": ["Inventory Management", "Inventory module"],
    "ai": ["AI module"],
    "finance": ["Finance module"],
    "financial": ["Finance module"],
    "sales": ["Sales module"],
    "sale": ["Sales module"],
    "tax": ["Tax module", "GST"],
    "gst": ["GST", "Tax module"],
    "purchase": ["Purchase Orders"],
    "purchasing": ["Purchase Orders"],
}

COMPOUND_MODULE_RE = re.compile(r"\b(\w+)\s+modules?\b", re.IGNORECASE)
COMPOUND_MANAGEMENT_RE = re.compile(r"\b(\w+)\s+management\b", re.IGNORECASE)
CREATE_SALE_RE = re.compile(
    r"\b(?:create|make)\s+(?:a\s+)?sale\b|\b(?:how\s+to\s+)?(?:create|make)\s+(?:a\s+)?sale\b",
    re.IGNORECASE,
)
CREATE_WAREHOUSE_RE = re.compile(
    r"\b(?:create|add)\s+(?:a\s+)?warehouses?\b|\bhow\s+to\s+(?:create|add)\s+(?:a\s+)?warehouses?\b",
    re.IGNORECASE,
)
PURCHASE_ORDER_RE = re.compile(
    r"\b(?:purchase\s+orders?|purchas(?:e|ing)\s+order|order\s+purchas(?:e|ing)|"
    r"how\s+to\s+purchas(?:e|ing)|procurement)\b",
    re.IGNORECASE,
)
POS_FEATURE_RE = re.compile(r"\bpos\s+features?\b", re.IGNORECASE)
OVERVIEW_QUERY_RE = re.compile(
    r"\b(?:introduce|introduction|overview|benefits?\s+of|advantages?\s+of)\b",
    re.IGNORECASE,
)
GENERIC_OVERVIEW_RE = re.compile(
    r"\b(?:what\s+is|tell\s+me\s+about)\s+(?:direkt|direct)\s*$",
    re.IGNORECASE,
)


class QueryMapper:
    """Resolve user queries to section targets and expanded search terms."""

    _merged_synonyms_cache: dict[str, str] | None = None

    @classmethod
    def merged_section_synonyms(cls) -> dict[str, str]:
        if cls._merged_synonyms_cache is None:
            merged = dict(SECTION_TITLE_SYNONYMS)
            merged.update(SectionRegistry.build_title_synonyms())
            cls._merged_synonyms_cache = merged
        return cls._merged_synonyms_cache

    @classmethod
    def clear_cache(cls) -> None:
        cls._merged_synonyms_cache = None

    @classmethod
    def resolve_target_sections(cls, query_lower: str, query_tokens: set[str]) -> list[str]:
        """Map query text to ordered official section titles using all mapping tables."""
        if cls.is_out_of_scope_query(query_lower):
            return []

        compound_targets = cls._resolve_compound_phrases(query_lower)
        if compound_targets:
            return compound_targets

        targets: list[str] = []
        intent_sections: set[str] = set()

        matched_intents: list[tuple[str, list[str]]] = []
        for phrase, sections in QUERY_INTENT_SECTIONS.items():
            if phrase in query_lower:
                matched_intents.append((phrase, sections))

        matched_intents.sort(key=lambda item: len(item[0]), reverse=True)
        for _, sections in matched_intents:
            for section in sections:
                intent_sections.add(section)
                if section not in targets:
                    targets.append(section)

        merged_synonyms = cls.merged_section_synonyms()
        for keyword, section_title in sorted(
            merged_synonyms.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if len(keyword.split()) == 1 and keyword in AMBIGUOUS_TOKENS:
                continue
            if (keyword in query_lower or keyword in query_tokens) and section_title not in targets:
                targets.append(section_title)

        for topic in cls.matched_faq_topics(query_lower):
            for section in FAQ_TOPIC_TO_SECTIONS.get(topic, []):
                if section not in targets:
                    targets.append(section)

        for group_key, synonyms in QUERY_SYNONYMS.items():
            group_hit = group_key in query_tokens or group_key in query_lower
            synonym_hit = any(syn in query_lower for syn in synonyms if len(syn) > 2)
            if not group_hit and not synonym_hit:
                continue

            mapped = merged_synonyms.get(group_key)
            if mapped and mapped not in targets:
                targets.append(mapped)

        targets = cls._prioritize_specific_sections(query_lower, query_tokens, targets)

        if intent_sections:
            intent_first = [s for s in targets if s in intent_sections]
            intent_rest = [s for s in targets if s not in intent_sections]
            targets = intent_first + intent_rest

        return targets

    @classmethod
    def is_out_of_scope_query(cls, query_lower: str) -> bool:
        """Return True when the query asks about topics not covered in the manual."""
        return any(keyword in query_lower for keyword in OUT_OF_SCOPE_KEYWORDS)

    @classmethod
    def answer_covers_query(cls, query_lower: str, answer_lower: str) -> bool:
        """Return False when the answer clearly does not address the query topic."""
        if cls.is_out_of_scope_query(query_lower):
            return False

        if not answer_lower.strip():
            return False

        if cls._is_generic_overview_query(query_lower):
            return "direkt" in answer_lower or "business" in answer_lower

        distinctive = cls._distinctive_query_terms(query_lower)
        if not distinctive:
            return True

        overview_terms = {
            "introduce", "introduction", "overview", "benefit", "benefits",
            "advantage", "advantages", "feature", "features",
        }
        if distinctive <= overview_terms:
            return "direkt" in answer_lower or "business" in answer_lower

        answer_words_3 = set(re.findall(r"\b[a-z]{3,}\b", answer_lower))
        answer_words_2 = set(re.findall(r"\b[a-z]{2,}\b", answer_lower))

        normalized_distinctive = {cls._normalize_word_form(w) for w in distinctive}
        normalized_answer_3 = {cls._normalize_word_form(w) for w in answer_words_3}
        overlap = normalized_distinctive & normalized_answer_3
        if overlap:
            return True

        if distinctive & answer_words_2:
            return True

        merged = cls.merged_section_synonyms()
        for term in distinctive:
            mapped = merged.get(term)
            if mapped and mapped.lower() in answer_lower:
                return True

        for group_key, synonyms in QUERY_SYNONYMS.items():
            group_hit = group_key in distinctive
            synonym_hit = any(
                (term in syn and len(term) >= 3)
                or (syn in term and len(syn) >= 3)
                or cls._normalize_word_form(term) == cls._normalize_word_form(syn.split()[0] if " " in syn else syn)
                for term in distinctive
                for syn in synonyms
            )
            if not group_hit and not synonym_hit:
                continue
            if any(syn in answer_lower for syn in synonyms if len(syn) > 2):
                return True

        module_match = COMPOUND_MODULE_RE.search(query_lower)
        if module_match:
            topic = MODULE_TOPIC_ALIASES.get(
                module_match.group(1).lower(), module_match.group(1).lower()
            )
            topic_synonyms = QUERY_SYNONYMS.get(topic, [])
            if topic_synonyms and any(
                syn in answer_lower for syn in topic_synonyms if len(syn) > 2
            ):
                return True
            if topic in answer_words_2:
                return True

        management_match = COMPOUND_MANAGEMENT_RE.search(query_lower)
        if management_match:
            topic = MODULE_TOPIC_ALIASES.get(
                management_match.group(1).lower(), management_match.group(1).lower()
            )
            topic_synonyms = QUERY_SYNONYMS.get(topic, [])
            if topic_synonyms and any(
                syn in answer_lower for syn in topic_synonyms if len(syn) > 2
            ):
                return True
            if topic in answer_words_2:
                return True

        if any(term in answer_lower for term in distinctive):
            return True

        return False

    @staticmethod
    def _normalize_word_form(word: str) -> str:
        """Normalize singular/plural forms for relaxed matching."""
        w = word.lower()
        if len(w) > 3 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("ses"):
            return w[:-2]
        if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        if len(w) > 2 and not w.endswith("s"):
            return w + "s"
        return w

    @classmethod
    def _distinctive_query_terms(cls, query_lower: str) -> set[str]:
        """Extract meaningful topic terms, excluding generic/question words."""
        skip = {
            "direkt", "direct", "the", "does", "do", "is", "are", "was", "were",
            "what", "which", "who", "how", "when", "where", "why", "can", "could",
            "would", "should", "will", "about", "tell", "me", "provide", "support",
            "use", "using", "has", "have", "had", "its", "it", "this", "that",
            "with", "for", "and", "or", "not", "any", "an", "a", "in", "on", "to",
            "of", "be", "been", "being", "get", "give", "show", "explain",
            "at", "by", "up", "out", "if", "so", "we", "he", "as", "or",
        }
        terms = {
            token
            for token in re.findall(r"\b[a-z]{2,}\b", query_lower)
            if token not in skip
        }
        return terms

    @classmethod
    def _resolve_compound_phrases(cls, query_lower: str) -> list[str]:
        """Resolve high-confidence compound phrases before generic token matching."""
        if CREATE_SALE_RE.search(query_lower):
            return ["Creating a Sale"]

        if CREATE_WAREHOUSE_RE.search(query_lower):
            return cls._dedupe_sections(["Step 2: Create Warehouses", "Getting Started"])

        specific_topic = cls._resolve_specific_topic(query_lower)
        if specific_topic:
            return [specific_topic]

        if PURCHASE_ORDER_RE.search(query_lower):
            return cls._dedupe_sections(["Purchase Orders", "Order Processing"])

        pos_match = POS_FEATURE_RE.search(query_lower)
        if pos_match:
            return cls._dedupe_sections(
                ["Omnichannel Point of Sale(POS)", "Creating a Sale", "Sales module"]
            )

        module_match = COMPOUND_MODULE_RE.search(query_lower)
        if module_match:
            topic = module_match.group(1).lower()
            canonical = MODULE_TOPIC_ALIASES.get(topic, topic)
            section = cls._find_module_section(canonical)
            if section:
                return [section]

        management_match = COMPOUND_MANAGEMENT_RE.search(query_lower)
        if management_match:
            topic = management_match.group(1).lower()
            sections = MANAGEMENT_TOPIC_SECTIONS.get(topic)
            if sections:
                return cls._dedupe_sections(sections)

        if cls._is_generic_overview_query(query_lower):
            if any(w in query_lower for w in ("benefit", "advantage", "feature")):
                return cls._dedupe_sections(
                    ["Key Features", "About Direkt", "Introduction"]
                )
            if any(w in query_lower for w in ("introduce", "introduction", "overview")):
                return cls._dedupe_sections(
                    ["Introduction", "About Direkt", "Key Features"]
                )
            return cls._dedupe_sections(["About Direkt", "Introduction", "Key Features"])

        return []

    @classmethod
    def _resolve_specific_topic(cls, query_lower: str) -> str | None:
        """Map queries like 'tell me about direkt security' to a specific section."""
        merged = cls.merged_section_synonyms()
        best_keyword = ""
        best_section: str | None = None

        for keyword, section_title in sorted(
            merged.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if len(keyword) <= 3:
                continue
            if keyword in {"direkt", "direct", "about", "tell", "what", "how"}:
                continue
            if keyword not in query_lower:
                continue
            if len(keyword) > len(best_keyword):
                best_keyword = keyword
                best_section = section_title

        if best_section and best_section.lower() not in GENERIC_SECTIONS:
            return best_section

        return None

    @classmethod
    def _is_generic_overview_query(cls, query_lower: str) -> bool:
        """True only for broad product-overview questions, not topic-specific ones."""
        if "direkt" not in query_lower and "direct" not in query_lower:
            return False

        if cls._resolve_specific_topic(query_lower):
            return False

        if OVERVIEW_QUERY_RE.search(query_lower):
            return True

        if GENERIC_OVERVIEW_RE.search(query_lower.strip()):
            return True

        return False

    @classmethod
    def _find_module_section(cls, topic: str) -> str | None:
        """Find an official section title for a module topic (e.g. sales -> Sales module)."""
        for title in SectionRegistry.get_titles():
            module_match = re.match(rf"^{re.escape(topic)}\s+module$", title, re.IGNORECASE)
            if module_match:
                return title
        return None

    @staticmethod
    def _dedupe_sections(sections: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for section in sections:
            key = section.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(section)
        return ordered

    @classmethod
    def matched_faq_topics(cls, query_lower: str) -> list[str]:
        """Return FAQ topic keys whose keywords appear in the query."""
        matched: list[str] = []
        for topic, keywords in FAQ_TOPIC_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                matched.append(topic)
        return matched

    @classmethod
    def expand_query_keywords(cls, keywords: set[str]) -> set[str]:
        """Expand keyword set using QUERY_SYNONYMS groups."""
        expanded = set(keywords)
        for keyword in keywords:
            for group_key, synonyms in QUERY_SYNONYMS.items():
                if keyword in synonyms or keyword == group_key:
                    expanded.update(synonyms)
                    expanded.add(group_key)
        return expanded

    @classmethod
    def expanded_query_terms(cls, query_lower: str, query_tokens: set[str]) -> set[str]:
        """All search terms derived from the query and synonym groups."""
        terms = set(query_tokens)
        terms.update(re.findall(r"\b[a-z0-9]{2,}\b", query_lower))
        return cls.expand_query_keywords(terms)

    @classmethod
    def section_priority_score(
        cls,
        section_title_lower: str,
        target_sections: list[str],
        query_lower: str,
        query_tokens: set[str],
    ) -> float:
        """Higher score means the section is a better match for the query intent."""
        if not section_title_lower or not target_sections:
            return 0.0

        score = 0.0
        target_lowers = [section.lower() for section in target_sections]

        if section_title_lower in target_lowers:
            rank = target_lowers.index(section_title_lower)
            score += max(0.55 - (rank * 0.08), 0.20)

        for phrase, sections in sorted(
            QUERY_INTENT_SECTIONS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if phrase in query_lower and section_title_lower in {
                section.lower() for section in sections
            }:
                score += 0.40
                break

        if CREATE_SALE_RE.search(query_lower) and section_title_lower == "creating a sale":
            score += 0.35

        if CREATE_WAREHOUSE_RE.search(query_lower) and section_title_lower.startswith("step 2"):
            score += 0.35

        merged_synonyms = cls.merged_section_synonyms()
        for keyword, mapped_title in sorted(
            merged_synonyms.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if mapped_title.lower() != section_title_lower:
                continue
            if keyword in query_lower or keyword in query_tokens:
                score += 0.12 + min(len(keyword.split()) * 0.05, 0.15)

        for topic in cls.matched_faq_topics(query_lower):
            topic_sections = FAQ_TOPIC_TO_SECTIONS.get(topic, [])
            if any(section.lower() == section_title_lower for section in topic_sections):
                score += 0.18

        if PURCHASE_ORDER_RE.search(query_lower):
            if section_title_lower == "purchase orders":
                score += 0.35
            elif section_title_lower == "order processing":
                score -= 0.20

        if POS_FEATURE_RE.search(query_lower):
            if "point of sale" in section_title_lower or section_title_lower.startswith("omnichannel"):
                score += 0.30
            elif section_title_lower == "key features":
                score -= 0.15

        compound_module = COMPOUND_MODULE_RE.search(query_lower)
        if compound_module:
            topic = MODULE_TOPIC_ALIASES.get(compound_module.group(1).lower(), compound_module.group(1).lower())
            expected = cls._find_module_section(topic)
            if expected and section_title_lower == expected.lower():
                score += 0.35

        compound_management = COMPOUND_MANAGEMENT_RE.search(query_lower)
        if compound_management:
            topic = compound_management.group(1).lower()
            expected_sections = MANAGEMENT_TOPIC_SECTIONS.get(topic, [])
            if any(section.lower() == section_title_lower for section in expected_sections):
                score += 0.30
            elif section_title_lower == "inventory management" and topic not in ("inventory",):
                score -= 0.25

        if section_title_lower in GENERIC_SECTIONS:
            specific_hits = cls._count_specific_topic_hits(query_lower, query_tokens)
            if specific_hits >= 2:
                score *= 0.55
            elif specific_hits == 1:
                score *= 0.75

        return min(score, 1.0)

    @classmethod
    def chunk_topic_relevance(cls, text_lower: str, query_lower: str, query_tokens: set[str]) -> float:
        """Score how well chunk text aligns with query topics and synonyms."""
        if not text_lower:
            return 0.0

        expanded = cls.expanded_query_terms(query_lower, query_tokens)
        query_words = {term for term in expanded if len(term) > 2}
        text_words = set(re.findall(r"\b[a-z]{3,}\b", text_lower))

        overlap = len(query_words & text_words)
        if query_words:
            score = overlap / len(query_words)
        else:
            score = 0.0

        for topic in cls.matched_faq_topics(query_lower):
            topic_keywords = FAQ_TOPIC_KEYWORDS.get(topic, [])
            query_has = any(kw in query_lower for kw in topic_keywords)
            text_has = any(kw in text_lower for kw in topic_keywords)
            if query_has and text_has:
                score = max(score, 0.45)

        if PURCHASE_ORDER_RE.search(query_lower):
            if "purchase order" in text_lower or "supplier" in text_lower:
                score = max(score, 0.55)
            if "browse the shared catalogue" in text_lower or "checkout" in text_lower:
                score *= 0.5

        compound_module = COMPOUND_MODULE_RE.search(query_lower)
        if compound_module:
            topic = MODULE_TOPIC_ALIASES.get(
                compound_module.group(1).lower(), compound_module.group(1).lower()
            )
            if topic in text_lower or f"{topic} module" in text_lower:
                score = max(score, 0.50)

        return min(score, 1.0)

    @classmethod
    def pick_best_document(
        cls,
        documents: list[dict[str, Any]],
        query_lower: str,
        query_tokens: set[str],
        target_sections: list[str],
    ) -> dict[str, Any] | None:
        """Select the best chunk for a query among candidate documents."""
        if not documents:
            return None

        target_lowers = {section.lower() for section in target_sections}

        candidates = documents
        if target_sections:
            section_matches = [
                doc for doc in documents if doc.get("section_title_lower") in target_lowers
            ]
            if section_matches:
                candidates = section_matches

        return max(
            candidates,
            key=lambda doc: (
                cls.section_priority_score(
                    doc.get("section_title_lower", ""),
                    target_sections,
                    query_lower,
                    query_tokens,
                ),
                cls.chunk_topic_relevance(
                    doc.get("text_lower", ""),
                    query_lower,
                    query_tokens,
                ),
                -cls._target_section_rank(
                    doc.get("section_title_lower", ""),
                    target_sections,
                ),
            ),
        )

    @staticmethod
    def _target_section_rank(section_title_lower: str, target_sections: list[str]) -> int:
        """Lower rank means higher priority in the target list."""
        target_lowers = [section.lower() for section in target_sections]
        if section_title_lower in target_lowers:
            return target_lowers.index(section_title_lower)
        return len(target_sections)

    @classmethod
    def _prioritize_specific_sections(
        cls,
        query_lower: str,
        query_tokens: set[str],
        targets: list[str],
    ) -> list[str]:
        """Keep intent order but move generic intro sections behind specific matches."""
        if len(targets) <= 1:
            return targets

        specific_hits = cls._count_specific_topic_hits(query_lower, query_tokens)
        if specific_hits == 0:
            return targets

        specific = [section for section in targets if section.lower() not in GENERIC_SECTIONS]
        generic = [section for section in targets if section.lower() in GENERIC_SECTIONS]
        if not specific:
            return targets
        return specific + generic

    @classmethod
    def _count_specific_topic_hits(cls, query_lower: str, query_tokens: set[str]) -> int:
        hits = 0
        generic = {"direkt", "about", "tell", "me", "the", "what", "how", "is", "are"}
        merged_synonyms = cls.merged_section_synonyms()

        for keyword in sorted(merged_synonyms, key=len, reverse=True):
            if keyword in generic:
                continue
            if len(keyword) <= 2:
                continue
            if keyword in query_lower or keyword in query_tokens:
                hits += 1

        for topic in cls.matched_faq_topics(query_lower):
            if topic not in ("getting_started", "developer"):
                hits += 1

        return hits
