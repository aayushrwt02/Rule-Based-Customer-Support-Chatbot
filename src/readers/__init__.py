"""Document reader factory for automatic file type detection."""

from pathlib import Path

from src.readers.base_reader import BaseReader, DocumentSegment
from src.readers.csv_reader import CSVReader
from src.readers.docx_reader import DOCXReader
from src.readers.excel_reader import ExcelReader
from src.readers.json_reader import JSONReader
from src.readers.pdf_reader import PDFReader
from src.readers.txt_reader import TXTReader
from src.utils.file_manager import detect_file_type
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

READER_REGISTRY: dict[str, type[BaseReader]] = {
    "pdf": PDFReader,
    "docx": DOCXReader,
    "xlsx": ExcelReader,
    "csv": CSVReader,
    "json": JSONReader,
    "txt": TXTReader,
}


def get_reader(file_type: str) -> BaseReader:
    """Return the appropriate reader instance for the given file type."""
    if file_type not in READER_REGISTRY:
        raise ValueError(f"No reader available for file type: {file_type}")
    return READER_REGISTRY[file_type]()


def read_document(file_path: Path) -> list[DocumentSegment]:
    """Automatically detect file type and read the document."""
    file_type = detect_file_type(file_path)
    logger.info("Detected file type '%s' for %s", file_type, file_path.name)
    reader = get_reader(file_type)
    return reader.read(file_path)
