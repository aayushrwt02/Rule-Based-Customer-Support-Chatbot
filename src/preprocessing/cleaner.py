"""Text cleaning — preserves document structure and reflows soft-wrapped lines."""

import re
import unicodedata

from src.preprocessing.heading_patterns import is_list_item, is_section_heading, is_table_row
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_PATTERN = re.compile(r"[^\S\n]+")
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
UNWANTED_CHARS_PATTERN = re.compile(
    r"[^\w\s.,;:!?\'\"()\[\]{}\-/@#$%&*+=<>|\\~₹→\n]", re.UNICODE
)


class TextCleaner:
    """Clean and normalize extracted document text while preserving structure."""

    def clean(self, text: str) -> str:
        """Apply full cleaning pipeline to raw text."""
        if not text or not text.strip():
            return ""

        cleaned = text
        cleaned = self._normalize_unicode(cleaned)
        cleaned = self._remove_control_characters(cleaned)
        cleaned = self._remove_unwanted_characters(cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._reflow_soft_wrapped_lines(cleaned)
        cleaned = self._preserve_section_breaks(cleaned)
        return cleaned.strip()

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        return CONTROL_CHAR_PATTERN.sub("", text)

    @staticmethod
    def _remove_unwanted_characters(text: str) -> str:
        return UNWANTED_CHARS_PATTERN.sub(" ", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        lines = text.split("\n")
        normalized_lines = [MULTI_SPACE_PATTERN.sub(" ", line).strip() for line in lines]
        return "\n".join(normalized_lines)

    def _reflow_soft_wrapped_lines(self, text: str) -> str:
        """
        Merge soft-wrapped lines into full paragraphs.

        Headings and blank lines stay separate; body lines are joined with spaces.
        """
        lines = text.split("\n")
        output: list[str] = []
        paragraph_parts: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_parts:
                output.append(" ".join(paragraph_parts))
                paragraph_parts.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                if output and output[-1] != "":
                    output.append("")
                continue

            if is_section_heading(stripped):
                flush_paragraph()
                output.append(stripped)
                continue

            if is_list_item(stripped) or is_table_row(stripped):
                flush_paragraph()
                output.append(stripped)
                continue

            paragraph_parts.append(stripped)

        flush_paragraph()
        return "\n".join(output)

    @staticmethod
    def _preserve_section_breaks(text: str) -> str:
        result = EXCESS_NEWLINES_PATTERN.sub("\n\n", text)
        lines = result.split("\n")
        cleaned_lines: list[str] = []
        prev_blank = False

        for line in lines:
            if not line.strip():
                if not prev_blank:
                    cleaned_lines.append("")
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False

        return "\n".join(cleaned_lines).strip()
