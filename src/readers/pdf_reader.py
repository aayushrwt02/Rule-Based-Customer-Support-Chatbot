"""PDF document reader using PyMuPDF."""

from pathlib import Path

import fitz

from src.readers.base_reader import BaseReader, DocumentSegment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFReader(BaseReader):
    """Extract text from PDF files page by page."""

    def read(self, file_path: Path) -> list[DocumentSegment]:
        segments: list[DocumentSegment] = []

        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text.strip():
                        segments.append(
                            DocumentSegment(
                                text=text,
                                page_number=page_num + 1,
                                metadata={"source_page": page_num + 1},
                            )
                        )
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", file_path, exc)
            raise RuntimeError(f"Could not read PDF file: {file_path}") from exc

        logger.info("PDF reader extracted %d page(s) from %s", len(segments), file_path.name)
        return segments
