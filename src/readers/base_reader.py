"""Base reader interface and shared data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentSegment:
    """A segment of text extracted from a document with optional metadata."""

    text: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReader(ABC):
    """Abstract base class for document readers."""

    @abstractmethod
    def read(self, file_path: Path) -> list[DocumentSegment]:
        """Read a document and return a list of text segments."""
        ...
