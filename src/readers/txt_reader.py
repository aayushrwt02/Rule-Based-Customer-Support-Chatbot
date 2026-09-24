"""Plain text document reader — preserves handbook structure for section chunking."""

from pathlib import Path

from src.readers.base_reader import BaseReader, DocumentSegment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TXTReader(BaseReader):
    """Extract text from plain text files as a single segment for section-aware chunking."""

    def read(self, file_path: Path) -> list[DocumentSegment]:
        segments: list[DocumentSegment] = []

        try:
            text = self._read_file(file_path)
            if text.strip():
                segments.append(
                    DocumentSegment(
                        text=text,
                        page_number=1,
                        metadata={"source": file_path.name},
                    )
                )
        except Exception as exc:
            logger.error("Failed to read TXT %s: %s", file_path, exc)
            raise RuntimeError(f"Could not read text file: {file_path}") from exc

        logger.info("TXT reader extracted content from %s", file_path.name)
        return segments

    @staticmethod
    def _read_file(file_path: Path) -> str:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode text file: {file_path}")
