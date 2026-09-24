"""Rule-based extraction optimized for Direkt manual document content."""

import re

from src.preprocessing.heading_patterns import (
    COMPANY_FIELDS_RE,
    DASHBOARD_METRICS_RE,
    DIREKT_MODULES_RE,
    FAQ_HEADING_RE,
    GST_TYPES_RE,
    GSTR_REPORTS_RE,
    MAJOR_HEADING_RE,
    MARKDOWN_HEADING_RE,
    MODULE_FEATURES_RE,
    NAV_PATH_RE,
    NUMBERED_SECTION_RE,
    PAYMENT_METHODS_RE,
    PRODUCT_FIELDS_RE,
    PROSE_SENTENCE_SPLIT_RE,
    SECURITY_FEATURES_RE,
    SHARE_CHANNELS_RE,
    STEP_HEADING_RE,
    WAREHOUSE_FIELDS_RE,
    is_list_item,
    is_section_heading,
    is_table_row,
)
from src.utils.config import FAQ_TOPIC_KEYWORDS, MAX_PARAGRAPHS, QUERY_SYNONYMS, SECTION_TITLE_SYNONYMS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FAQ_QA_PATTERN = re.compile(
    r"Q:\s*(.+?)\s+A:\s*(.+?)(?=\s*Q:|$)", re.IGNORECASE | re.DOTALL
)

# Split FAQ prose sentences on topic boundaries for multi-topic paragraphs
FAQ_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)

GENERIC_KEYWORDS = frozenset(
    {
        "information", "details", "about", "tell", "explain", "describe",
        "direkt", "many", "much", "business", "manage", "help", "what",
        "does", "can", "how",
    }
)

STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "what", "when",
        "where", "why", "how", "which", "who", "this", "that", "these",
        "those", "and", "or", "but", "if", "about", "tell", "please", "me",
        "available", "get", "apply", "use", "using",
    }
)

MULTI_TOPIC_CONNECTOR_RE = re.compile(r"\b(?:and|also|plus)\b", re.IGNORECASE)


class ParagraphExtractor:
    """Extract query-relevant paragraphs from indexed document chunks."""

    def extract(
        self,
        chunk_text: str,
        query: str,
        max_paragraphs: int = MAX_PARAGRAPHS,
    ) -> str:
        """Return the most relevant paragraph(s) for the query."""
        if not chunk_text.strip():
            return ""

        targeted = self._match_targeted_fact(chunk_text, query)
        if targeted:
            return targeted

        faq_answer = self._match_faq(chunk_text, query)
        if faq_answer:
            return faq_answer

        blocks = self._parse_blocks(chunk_text)
        if not blocks:
            return ""

        sections = self._group_into_sections(blocks)
        if not sections:
            return ""

        if len(sections) == 1:
            heading = sections[0].get("heading", "")
            if self._section_title_bonus(query, heading) >= 0.10:
                return self._extract_from_section(
                    sections[0], query, chunk_text, max_paragraphs
                )

        if self._is_multi_topic_query(query):
            return self._extract_multi_topic(sections, query, max_paragraphs)

        best_section = self._find_best_section(sections, query)
        if best_section is None:
            return ""

        return self._extract_from_section(best_section, query, chunk_text, max_paragraphs)

    def _extract_multi_topic(
        self,
        sections: list[dict],
        query: str,
        max_paragraphs: int,
    ) -> str:
        """Return paragraphs from multiple sections when query spans topics."""
        scored = [(self._score_section(section, query), section) for section in sections]
        scored.sort(key=lambda x: x[0], reverse=True)

        selected_texts: list[str] = []
        seen: set[str] = set()

        for score, section in scored:
            if score <= 0 or len(selected_texts) >= max_paragraphs:
                break
            text = self._extract_from_section(section, query, "", 1)
            if text and text not in seen:
                selected_texts.append(text)
                seen.add(text)

        if selected_texts:
            return "\n\n".join(selected_texts)
        return self._extract_from_section(scored[0][1], query, "", max_paragraphs)

    def _extract_from_section(
        self,
        section: dict,
        query: str,
        chunk_text: str,
        max_paragraphs: int,
    ) -> str:
        raw_paragraphs = [b["text"] for b in section["blocks"] if b["type"] == "paragraph"]
        if len(raw_paragraphs) == 1 and len(raw_paragraphs[0].split()) <= 90:
            return self._strip_section_heading(raw_paragraphs[0], section.get("heading", ""))

        paragraphs = self._expand_paragraph_units(raw_paragraphs)

        if not paragraphs:
            return ""

        scored = [(self._score_paragraph(p, query), p) for p in paragraphs]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score = scored[0][0]

        if best_score < 0.30:
            return ""

        threshold = best_score * 0.92
        selected = [(s, p) for s, p in scored if s >= threshold][:max_paragraphs]

        if len(selected) > 1 and selected[0][0] >= selected[1][0] * 1.5:
            selected = selected[:1]

        if chunk_text:
            selected.sort(
                key=lambda item: chunk_text.find(item[1]) if item[1] in chunk_text else 0
            )

        result = "\n".join(p for _, p in selected)
        result = self._strip_section_heading(result, section.get("heading", ""))
        logger.debug("Extracted %d paragraph(s) for query: %s", len(selected), query[:50])
        return result

    @staticmethod
    def _strip_section_heading(text: str, heading: str) -> str:
        """Remove a leading section heading duplicated in chunk text."""
        stripped = text.strip()
        if not heading:
            return stripped

        heading_lower = heading.lower()

        # Heading on its own line followed by body text
        prefix = f"{heading}\n"
        if stripped.lower().startswith(prefix.lower()):
            remainder = stripped[len(prefix):].strip()
            return remainder if remainder else stripped

        # Heading as the entire first line
        first_line, _, rest = stripped.partition("\n")
        if first_line.strip().lower() == heading_lower:
            return rest.strip() if rest.strip() else stripped

        # Do not strip when heading is a word prefix (e.g. "GST" from "GST Management")
        return stripped
    
    @staticmethod
    def _heading_from_text(text: str) -> str:
        """
        Extract the first heading from a chunk of text.
        Returns an empty string if no heading is found.
        """
        if not text.strip():
            return ""

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if (
                MAJOR_HEADING_RE.match(line)
                or MARKDOWN_HEADING_RE.match(line)
                or NUMBERED_SECTION_RE.match(line)
                or STEP_HEADING_RE.match(line)
                or is_section_heading(line)
            ):
                return line

        return ""

    def _match_targeted_fact(self, text: str, query: str) -> str:
        """Extract a precise fact when the query asks for specific Direkt details."""
        query_lower = query.lower()

        if (
            ("direkt" in query_lower or "direct" in query_lower)
            and any(
                w in query_lower
                for w in (
                    "introduce", "introduction", "overview", "benefit",
                    "advantage", "what is direkt", "about direkt",
                )
            )
        ):
            body = self._strip_section_heading(text, self._heading_from_text(text))
            if body.strip():
                return body.strip()

        if any(w in query_lower for w in ("payment method", "payment methods", "pay with")):
            match = PAYMENT_METHODS_RE.search(text)
            if match:
                methods = match.group(1).strip().rstrip(",")
                return f"Direkt supports multiple payment methods, including {methods}."

        if any(w in query_lower for w in ("payment", "payments", "upi", "cash", "cheque")):
            if "cash" in text.lower() and "upi" in text.lower():
                match = PAYMENT_METHODS_RE.search(text)
                if match:
                    methods = match.group(1).strip().rstrip(",")
                    return f"Direkt supports multiple payment methods, including {methods}."

        if any(w in query_lower for w in ("cgst", "sgst", "igst", "gst type", "gst calculate")):
            match = GST_TYPES_RE.search(text)
            if match:
                return f"Direkt automatically calculates {match.group(1)}."

        if any(w in query_lower for w in ("gstr", "gstr-1", "gstr-3b", "gst report", "gst filing")):
            match = GSTR_REPORTS_RE.search(text)
            if match:
                return f"Direkt generates {match.group(1)}."

        if any(w in query_lower for w in ("create company", "setup company", "company details", "gstin", "pan")):
            match = NAV_PATH_RE.search(text)
            if match and "company" in text.lower():
                return text.strip()
            if COMPANY_FIELDS_RE.search(text) and "company" in query_lower:
                return text.strip()

        if any(w in query_lower for w in ("warehouse", "add warehouse", "create warehouse")):
            match = NAV_PATH_RE.search(text)
            if match and "warehouse" in text.lower():
                return text.strip()
            if WAREHOUSE_FIELDS_RE.search(text) and "warehouse" in query_lower:
                return text.strip()

        if any(w in query_lower for w in ("add product", "new product", "product details", "sku")):
            match = NAV_PATH_RE.search(text)
            if match and "product" in text.lower():
                return text.strip()
            if PRODUCT_FIELDS_RE.search(text) and "product" in query_lower:
                return text.strip()

        if any(w in query_lower for w in ("pos", "create sale", "create a sale", "billing")):
            if "pos" in text.lower() or "scan" in text.lower():
                return text.strip()

        if any(w in query_lower for w in ("module", "modules")):
            match = DIREKT_MODULES_RE.search(text)
            if match:
                return (
                    f"Direkt provides comprehensive features including "
                    f"{match.group(1)} modules for complete business management."
                )
            for match in MODULE_FEATURES_RE.finditer(text):
                module_name = match.group(1)
                if module_name.lower() in query_lower:
                    return f"{module_name} helps {match.group(2).strip()}."

        if any(w in query_lower for w in ("dashboard", "analytics", "revenue", "profit", "metrics")):
            if DASHBOARD_METRICS_RE.search(text):
                return text.strip()

        if any(w in query_lower for w in ("security", "authentication", "backup", "permission", "encrypted")):
            if SECURITY_FEATURES_RE.search(text):
                return text.strip()

        if any(w in query_lower for w in ("whatsapp", "sms", "qr", "share", "catalogue")):
            if SHARE_CHANNELS_RE.search(text):
                return text.strip()

        if "navigate" in query_lower or "go to" in query_lower or "where" in query_lower:
            match = NAV_PATH_RE.search(text)
            if match:
                return f"Go to {match.group(1)}."

        return ""

    def _match_faq(self, text: str, query: str) -> str:
        """Return FAQ answer when query matches structured or prose FAQ content."""
        for match in FAQ_QA_PATTERN.finditer(text):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if self._faq_question_matches(query, question):
                return answer

        if FAQ_HEADING_RE.search(text) or "frequently asked questions" in text.lower():
            body = text
            for heading in ("Frequently Asked Questions",):
                if text.lower().startswith(heading.lower()):
                    body = text[len(heading):].strip()
                    break

            sentences = FAQ_SENTENCE_SPLIT_RE.split(body)
            best_sentence = ""
            best_score = 0.0
            query_lower = query.lower()

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                score = self._score_paragraph(sentence, query)
                score += self._faq_topic_boost(sentence, query_lower)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

            if best_sentence and best_score >= 0.40:
                return best_sentence

        return ""

    @staticmethod
    def _faq_topic_boost(sentence: str, query_lower: str) -> float:
        """Boost FAQ sentence score when it matches a query topic."""
        sentence_lower = sentence.lower()
        boost = 0.0

        for topic, keywords in FAQ_TOPIC_KEYWORDS.items():
            topic_in_query = any(kw in query_lower for kw in keywords)
            topic_in_sentence = any(kw in sentence_lower for kw in keywords)
            if topic_in_query and topic_in_sentence:
                boost = max(boost, 0.18)

        return boost

    def _faq_question_matches(self, query: str, faq_question: str) -> bool:
        """Check if user query matches an FAQ question."""
        query_kw = set(self._extract_keywords(query, exclude_generic=True))
        faq_kw = set(self._extract_keywords(faq_question, exclude_generic=False))
        if not query_kw or not faq_kw:
            return False

        overlap = query_kw & faq_kw
        if len(overlap) >= max(1, len(query_kw) * 0.5):
            return True

        expanded_query = self._expand_with_synonyms(query_kw)
        return len(expanded_query & faq_kw) >= max(1, len(query_kw) * 0.4)

    def _parse_blocks(self, text: str) -> list[dict[str, str]]:
        """Parse text into heading and paragraph blocks."""
        blocks: list[dict[str, str]] = []
        current_paragraph: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if current_paragraph:
                    blocks.append(
                        {"type": "paragraph", "text": self._join_paragraph_lines(current_paragraph)}
                    )
                    current_paragraph = []
                continue

            if self._is_heading(stripped):
                if current_paragraph:
                    blocks.append(
                        {"type": "paragraph", "text": self._join_paragraph_lines(current_paragraph)}
                    )
                    current_paragraph = []
                heading_type = (
                    "major_heading"
                    if STEP_HEADING_RE.match(stripped) or MAJOR_HEADING_RE.match(stripped)
                    else "minor_heading"
                )
                blocks.append({"type": heading_type, "text": stripped})
            elif is_list_item(stripped) or is_table_row(stripped):
                if current_paragraph:
                    blocks.append(
                        {"type": "paragraph", "text": self._join_paragraph_lines(current_paragraph)}
                    )
                    current_paragraph = []
                blocks.append({"type": "list_item", "text": stripped})
            else:
                current_paragraph.append(stripped)

        if current_paragraph:
            blocks.append(
                {"type": "paragraph", "text": self._join_paragraph_lines(current_paragraph)}
            )

        return blocks

    def _group_into_sections(self, blocks: list[dict[str, str]]) -> list[dict]:
        """Group blocks into sections bounded by headings."""
        sections: list[dict] = []
        current: dict | None = None

        for block in blocks:
            if block["type"] in ("major_heading", "minor_heading"):
                if current and (current["blocks"] or current.get("heading")):
                    sections.append(current)
                current = {"heading": block["text"], "blocks": []}
            else:
                if current is None:
                    current = {"heading": "", "blocks": []}
                current["blocks"].append(block)

        if current and (current["blocks"] or current.get("heading")):
            sections.append(current)

        return sections

    def _find_best_section(self, sections: list[dict], query: str) -> dict | None:
        """Return only a section with a meaningful relevance score."""
        if not sections:
            return None

        scored = [(self._score_section(section, query), section) for section in sections]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score = scored[0][0]

        if best_score < 0.30:
            return None

        return scored[0][1]

    def _score_section(self, section: dict, query: str) -> float:
        keywords = self._extract_keywords(query, exclude_generic=True)
        if not keywords:
            keywords = self._extract_keywords(query, exclude_generic=False)
        if not keywords:
            return 0.0

        heading = section.get("heading", "")
        text_parts = [heading] + [
            b["text"] for b in section["blocks"] if b["type"] == "paragraph"
        ]
        combined = " ".join(text_parts).lower()
        expanded = self._expand_with_synonyms(set(keywords))

        matched = sum(1 for kw in expanded if kw in combined)
        coverage = matched / len(expanded)

        heading_bonus = 0.0
        if heading:
            heading_lower = heading.lower()
            heading_bonus = sum(0.45 for kw in expanded if kw in heading_lower)

        section_bonus = self._section_title_bonus(query, heading)
        return min(coverage + heading_bonus + section_bonus, 1.0)

    def _section_title_bonus(self, query: str, heading: str) -> float:
        if not heading:
            return 0.0

        query_lower = query.lower()
        heading_lower = heading.lower()
        bonus = 0.0

        for keyword, section_title in SECTION_TITLE_SYNONYMS.items():
            if keyword in query_lower and section_title.lower() == heading_lower:
                bonus = max(bonus, 0.15)

        if heading_lower in query_lower:
            bonus = max(bonus, 0.10)

        return bonus

    def _score_paragraph(self, paragraph: str, query: str) -> float:
        keywords = self._extract_keywords(query, exclude_generic=True)
        if not keywords:
            keywords = self._extract_keywords(query, exclude_generic=False)
        if not keywords:
            return 0.0

        expanded = self._expand_with_synonyms(set(keywords))
        paragraph_lower = paragraph.lower()
        matched = 0
        total_occurrences = 0

        for keyword in expanded:
            count = len(
                re.findall(r"\b" + re.escape(keyword) + r"\w*\b", paragraph_lower)
            )
            if count > 0:
                matched += 1
                total_occurrences += count

        if matched == 0:
            return 0.0

        coverage = matched / len(expanded)
        frequency = min(total_occurrences / (len(expanded) * 2), 1.0)
        return (coverage * 0.7) + (frequency * 0.3)

    def _expand_with_synonyms(self, keywords: set[str]) -> set[str]:
        expanded = set(keywords)
        for keyword in keywords:
            for group_key, synonyms in QUERY_SYNONYMS.items():
                if keyword in synonyms or keyword == group_key:
                    expanded.update(synonyms)
                    expanded.add(group_key)
        return expanded

    def _expand_paragraph_units(self, paragraphs: list[str]) -> list[str]:
        expanded: list[str] = []
        for paragraph in paragraphs:
            if FAQ_QA_PATTERN.search(paragraph):
                expanded.append(paragraph)
                continue

            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
            if len(lines) > 1:
                expanded.extend(lines)
                continue

            # Split long FAQ-style prose into sentences for finer matching
            if len(paragraph.split()) > 50:
                sentences = FAQ_SENTENCE_SPLIT_RE.split(paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
                if len(sentences) > 1:
                    expanded.extend(sentences)
                    continue

            sentences = PROSE_SENTENCE_SPLIT_RE.split(paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > 1:
                expanded.extend(sentences)
                continue

            expanded.append(paragraph)
        return expanded

    @staticmethod
    def _join_paragraph_lines(lines: list[str]) -> str:
        if len(lines) == 1:
            return lines[0]
        return "\n".join(lines)

    @staticmethod
    def _extract_keywords(query: str, exclude_generic: bool = False) -> list[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
        keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        if exclude_generic:
            specific = [k for k in keywords if k not in GENERIC_KEYWORDS]
            if specific:
                return specific
        return keywords

    @staticmethod
    def _is_heading(line: str) -> bool:
        if MAJOR_HEADING_RE.match(line) or MARKDOWN_HEADING_RE.match(line):
            return True
        if NUMBERED_SECTION_RE.match(line):
            return True
        if STEP_HEADING_RE.match(line):
            return True
        return is_section_heading(line)

    @staticmethod
    def _is_multi_topic_query(query: str) -> bool:
        return bool(MULTI_TOPIC_CONNECTOR_RE.search(query))
