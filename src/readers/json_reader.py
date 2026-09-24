"""JSON document reader."""

import json
from pathlib import Path

from src.readers.base_reader import BaseReader, DocumentSegment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JSONReader(BaseReader):
    """Extract text from JSON files by flattening nested structures."""

    def read(self, file_path: Path) -> list[DocumentSegment]:
        segments: list[DocumentSegment] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            flattened_lines = self._flatten_json(data)
            if flattened_lines:
                chunk_size = 50
                for i in range(0, len(flattened_lines), chunk_size):
                    chunk = flattened_lines[i : i + chunk_size]
                    segments.append(
                        DocumentSegment(
                            text="\n".join(chunk),
                            page_number=(i // chunk_size) + 1,
                            metadata={"entry_start": i},
                        )
                    )
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", file_path, exc)
            raise RuntimeError(f"Could not parse JSON file: {file_path}") from exc
        except Exception as exc:
            logger.error("Failed to read JSON %s: %s", file_path, exc)
            raise RuntimeError(f"Could not read JSON file: {file_path}") from exc

        logger.info("JSON reader extracted %d segment(s) from %s", len(segments), file_path.name)
        return segments

    def _flatten_json(self, data: object, prefix: str = "") -> list[str]:
        """Recursively flatten JSON into key-value text lines."""
        lines: list[str] = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                lines.extend(self._flatten_json(value, new_prefix))
        elif isinstance(data, list):
            for index, item in enumerate(data):
                new_prefix = f"{prefix}[{index}]"
                lines.extend(self._flatten_json(item, new_prefix))
        else:
            lines.append(f"{prefix}: {data}")

        return lines
