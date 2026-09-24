"""Section-aware text chunking optimized for official uploaded documents."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from src.preprocessing.heading_patterns import (
    FAQ_HEADING_RE,
    MODULE_HEADING_RE,
    PROSE_SENTENCE_SPLIT_RE,
    STEP_HEADING_RE,
    is_list_item,
    is_section_heading,
    is_table_row,
)
from src.readers.base_reader import DocumentSegment
from src.utils.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, FAQ_TOPIC_KEYWORDS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TextChunk:
    """A chunk of text ready for embedding and storage."""

    text: str
    chunk_id: str
    document_name: str
    file_type: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Split document text into section-aligned chunks for accurate retrieval."""

    def __init__(
        self,
        chunk_size_words: int = CHUNK_SIZE_WORDS,
        chunk_overlap_words: int = CHUNK_OVERLAP_WORDS,
    ) -> None:
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

    def chunk_segments(
        self,
        segments: list[DocumentSegment],
        document_name: str,
        file_type: str,
    ) -> list[TextChunk]:
        """Chunk document segments using official section boundaries."""
        all_chunks: list[TextChunk] = []
        chunk_counter = 0

        for segment in segments:
            if not segment.text.strip():
                continue

            sections = self._split_into_sections(segment.text)

            for section_title, section_text in sections:
                if not section_text.strip():
                    continue

                section_type = self._detect_section_type(section_title, section_text)
                section_chunks = self._chunk_section(section_text, section_title)

                for chunk_text in section_chunks:
                    chunk_id = self._generate_chunk_id(
                        document_name, chunk_counter, chunk_text
                    )
                    all_chunks.append(
                        TextChunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            document_name=document_name,
                            file_type=file_type,
                            page_number=segment.page_number,
                            metadata={
                                **segment.metadata,
                                "chunk_index": chunk_counter,
                                "section_title": section_title,
                                "section_type": section_type,
                            },
                        )
                    )
                    chunk_counter += 1

        logger.info(
            "Created %d chunk(s) from document '%s'", len(all_chunks), document_name
        )
        return all_chunks

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split text at official document heading boundaries."""
        lines = text.split("\n")
        sections: list[tuple[str, str]] = []
        current_title = ""
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if is_section_heading(stripped):
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        prefix = f"{current_title}\n" if current_title else ""
                        sections.append((current_title, f"{prefix}{body}".strip()))
                current_title = stripped
                current_lines = []
            elif is_list_item(stripped) or is_table_row(stripped):
                current_lines.append(stripped)
            else:
                current_lines.append(stripped)

        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                prefix = f"{current_title}\n" if current_title else ""
                sections.append((current_title, f"{prefix}{body}".strip()))

        if not sections and text.strip():
            return [("", text.strip())]

        return sections

    def _chunk_section(self, text: str, section_title: str) -> list[str]:
        """Keep short sections intact; split only long or multi-topic sections."""
        word_count = len(text.split())
        if word_count <= self.chunk_size_words:
            if FAQ_HEADING_RE.match(section_title) and "Q:" not in text:
                topic_chunks = self._split_prose_faq_by_topic(text, section_title)
                if topic_chunks and len(topic_chunks) > 1:
                    return topic_chunks
            return [text]

        if FAQ_HEADING_RE.match(section_title) or "Q:" in text:
            faq_chunks = self._split_faq_section(text, section_title)
            if faq_chunks:
                return faq_chunks
            topic_chunks = self._split_prose_faq_by_topic(text, section_title)
            if topic_chunks:
                return topic_chunks
            prose_chunks = self._split_prose_faq_section(text, section_title)
            if prose_chunks:
                return prose_chunks

        heading_prefix = section_title if section_title else ""
        body = text
        if heading_prefix and text.startswith(heading_prefix):
            body = text[len(heading_prefix):].strip()

        if not body:
            return [text]

        body_words = body.split()
        chunks: list[str] = []
        start = 0

        while start < len(body_words):
            end = min(start + self.chunk_size_words, len(body_words))
            chunk_body = " ".join(body_words[start:end])

            if heading_prefix:
                chunk_text = f"{heading_prefix}\n{chunk_body}"
            else:
                chunk_text = chunk_body

            chunks.append(chunk_text.strip())

            if end >= len(body_words):
                break
            start = end - self.chunk_overlap_words

        return chunks

    def _split_faq_section(self, text: str, section_title: str) -> list[str]:
        """Split FAQ content into individual Q&A pair chunks."""
        body = text
        if section_title and text.startswith(section_title):
            body = text[len(section_title):].strip()

        faq_pattern = re.compile(
            r"Q:\s*.+?\s+A:\s*.+?(?=\s*Q:|$)", re.DOTALL | re.IGNORECASE
        )
        pairs = faq_pattern.findall(body)
        if not pairs:
            return []

        prefix = f"{section_title}\n" if section_title else ""
        return [f"{prefix}{pair.strip()}" for pair in pairs if pair.strip()]

    def _split_prose_faq_section(self, text: str, section_title: str) -> list[str]:
        """Split prose FAQ paragraphs into sentence-level topic chunks."""
        body = text
        if section_title and text.startswith(section_title):
            body = text[len(section_title):].strip()

        sentences = PROSE_SENTENCE_SPLIT_RE.split(body)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return []

        prefix = f"{section_title}\n" if section_title else ""
        return [f"{prefix}{sentence}" for sentence in sentences]

    def _split_prose_faq_by_topic(self, text: str, section_title: str) -> list[str]:
        """Group prose FAQ sentences by topic for more precise retrieval."""
        body = text
        if section_title and text.startswith(section_title):
            body = text[len(section_title):].strip()

        sentences = PROSE_SENTENCE_SPLIT_RE.split(body)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return []

        topic_groups: dict[str, list[str]] = {}
        ungrouped: list[str] = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            matched_topic = ""
            best_hits = 0

            for topic, keywords in FAQ_TOPIC_KEYWORDS.items():
                hits = sum(1 for kw in keywords if kw in sentence_lower)
                if hits > best_hits:
                    best_hits = hits
                    matched_topic = topic

            if matched_topic and best_hits > 0:
                topic_groups.setdefault(matched_topic, []).append(sentence)
            else:
                ungrouped.append(sentence)

        if not topic_groups:
            return []

        prefix = f"{section_title}\n" if section_title else ""
        chunks: list[str] = []
        for sentences_group in topic_groups.values():
            chunks.append(f"{prefix}{' '.join(sentences_group)}")

        if ungrouped:
            chunks.append(f"{prefix}{' '.join(ungrouped)}")

        return chunks if len(chunks) > 1 else []

    @staticmethod
    def _detect_section_type(title: str, text: str) -> str:
        if FAQ_HEADING_RE.match(title) or text.strip().startswith("Q:"):
            return "faq"
        if STEP_HEADING_RE.match(title):
            return "setup_step"
        if MODULE_HEADING_RE.match(title):
            return "feature_module"
        if title:
            return "manual_section"
        return "general"

    @staticmethod
    def _generate_chunk_id(document_name: str, index: int, text: str) -> str:
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        safe_name = document_name.replace(" ", "_").replace(".", "_")
        return f"{safe_name}_chunk_{index}_{content_hash}"
