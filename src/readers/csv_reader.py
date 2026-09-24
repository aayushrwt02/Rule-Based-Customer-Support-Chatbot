"""CSV document reader using pandas."""

from pathlib import Path

import pandas as pd

from src.readers.base_reader import BaseReader, DocumentSegment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CSVReader(BaseReader):
    """Extract text from CSV files."""

    def read(self, file_path: Path) -> list[DocumentSegment]:
        segments: list[DocumentSegment] = []

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return segments

            lines: list[str] = [" | ".join(str(col) for col in df.columns)]
            for row_index, row in df.iterrows():
                row_text = " | ".join(
                    str(val) for val in row.values if pd.notna(val) and str(val).strip()
                )
                if row_text.strip():
                    lines.append(row_text)

            chunk_size = 50
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i : i + chunk_size]
                segments.append(
                    DocumentSegment(
                        text="\n".join(chunk_lines),
                        page_number=(i // chunk_size) + 1,
                        metadata={"row_start": i, "row_end": min(i + chunk_size, len(lines))},
                    )
                )
        except Exception as exc:
            logger.error("Failed to read CSV %s: %s", file_path, exc)
            raise RuntimeError(f"Could not read CSV file: {file_path}") from exc

        logger.info("CSV reader extracted %d segment(s) from %s", len(segments), file_path.name)
        return segments
