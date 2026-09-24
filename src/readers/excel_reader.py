"""Excel document reader using pandas and openpyxl."""

from pathlib import Path

import pandas as pd

from src.readers.base_reader import BaseReader, DocumentSegment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExcelReader(BaseReader):
    """Extract text from XLSX files, sheet by sheet."""

    def read(self, file_path: Path) -> list[DocumentSegment]:
        segments: list[DocumentSegment] = []

        try:
            excel_file = pd.ExcelFile(file_path)
            for sheet_index, sheet_name in enumerate(excel_file.sheet_names):
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if df.empty:
                    continue

                lines: list[str] = [f"Sheet: {sheet_name}"]
                lines.append(" | ".join(str(col) for col in df.columns))

                for _, row in df.iterrows():
                    row_text = " | ".join(
                        str(val) for val in row.values if pd.notna(val) and str(val).strip()
                    )
                    if row_text.strip():
                        lines.append(row_text)

                text = "\n".join(lines)
                if text.strip():
                    segments.append(
                        DocumentSegment(
                            text=text,
                            page_number=sheet_index + 1,
                            metadata={"sheet_name": sheet_name},
                        )
                    )
        except Exception as exc:
            logger.error("Failed to read Excel %s: %s", file_path, exc)
            raise RuntimeError(f"Could not read Excel file: {file_path}") from exc

        logger.info(
            "Excel reader extracted %d sheet(s) from %s", len(segments), file_path.name
        )
        return segments
