"""File management utilities for document handling and path validation."""

import hashlib
import shutil
from pathlib import Path

from src.utils.config import (
    DOCUMENTS_DIR,
    DOCUMENTS_FINGERPRINT_FILE,
    INDEX_VERSION,
    INDEX_VERSION_FILE,
    SUPPORTED_EXTENSIONS,
    VECTOR_DB_DIR,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_stored_documents() -> list[Path]:
    """Return all supported documents in documents/ sorted by name."""
    if not DOCUMENTS_DIR.exists():
        return []

    return sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def get_stored_document() -> Path | None:
    """Return the most recently modified document in documents/."""
    documents = get_stored_documents()
    if not documents:
        return None
    return max(documents, key=lambda path: path.stat().st_mtime)


def compute_documents_fingerprint() -> str:
    """Hash document names and modification times to detect changes."""
    documents = get_stored_documents()
    if not documents:
        return ""

    parts: list[str] = []
    for doc in documents:
        stat = doc.stat()
        parts.append(f"{doc.name}:{stat.st_mtime_ns}:{stat.st_size}")

    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def save_documents_fingerprint() -> None:
    """Persist fingerprint after successful indexing."""
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_FINGERPRINT_FILE.write_text(
        compute_documents_fingerprint(), encoding="utf-8"
    )


def documents_have_changed() -> bool:
    """Check whether documents/ content differs from last indexed state."""
    if not DOCUMENTS_FINGERPRINT_FILE.exists():
        return True
    stored = DOCUMENTS_FINGERPRINT_FILE.read_text(encoding="utf-8").strip()
    return stored != compute_documents_fingerprint()


def is_vector_db_initialized() -> bool:
    """Check whether the ChromaDB persistent store has been created and populated."""
    chroma_db_file = VECTOR_DB_DIR / "chroma.sqlite3"
    if not chroma_db_file.exists():
        return False

    if not INDEX_VERSION_FILE.exists():
        logger.info("Index version file missing — re-index required.")
        return False

    if INDEX_VERSION_FILE.read_text(encoding="utf-8").strip() != INDEX_VERSION:
        logger.info("Index version mismatch — re-index required.")
        return False

    if documents_have_changed():
        logger.info("Documents folder changed — re-index required.")
        return False

    try:
        from src.database.chroma_manager import ChromaManager

        manager = ChromaManager()
        return manager.count() > 0
    except Exception as exc:
        logger.warning("Could not verify vector database state: %s", exc)
        return False


def has_processed_data() -> bool:
    """Return True when locally saved processed data matches current documents."""
    return is_vector_db_initialized()


def save_index_version() -> None:
    """Persist the current index version and document fingerprint after indexing."""
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_VERSION_FILE.write_text(INDEX_VERSION, encoding="utf-8")
    save_documents_fingerprint()
    logger.info("Saved index version: %s", INDEX_VERSION)


def validate_document_path(path_str: str) -> Path | None:
    """Validate a user-provided document path."""
    if not path_str or not path_str.strip():
        return None

    try:
        path = Path(path_str.strip()).expanduser().resolve()
    except (OSError, ValueError) as exc:
        logger.warning("Invalid path format: %s — %s", path_str, exc)
        return None

    if not path.exists():
        logger.warning("Path does not exist: %s", path)
        return None

    if not path.is_file():
        logger.warning("Path is not a file: %s", path)
        return None

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        logger.warning(
            "Unsupported file type '%s'. Supported: %s", path.suffix, supported
        )
        return None

    return path


def detect_file_type(file_path: Path) -> str:
    """Detect document type from file extension."""
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")
    return SUPPORTED_EXTENSIONS[ext]


def copy_document_to_storage(source_path: Path) -> Path:
    """Copy an uploaded document into documents/, avoiding name collisions."""
    ensure_directories()
    dest_path = DOCUMENTS_DIR / source_path.name

    if dest_path.exists():
        stem = source_path.stem
        suffix = source_path.suffix
        counter = 1
        while dest_path.exists():
            dest_path = DOCUMENTS_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(source_path, dest_path)
    logger.info("Document copied to: %s", dest_path)
    return dest_path


def prompt_for_document_path() -> Path:
    """Prompt the user until a valid document path is provided."""
    supported = ", ".join(ext.upper().lstrip(".") for ext in SUPPORTED_EXTENSIONS)

    print("\n" + "=" * 60)
    print("  Chatbot — First-Time Setup")
    print("=" * 60)
    print("\nNo documents found in the documents/ folder.")
    print("Please provide a company document to index.")
    print(f"\nSupported formats: {supported}")
    print("\nEnter the full path to your document.")
    print("Example: C:\\Users\\Documents\\employee_handbook.pdf\n")

    while True:
        user_input = input("Enter document path: ").strip()

        if (user_input.startswith('"') and user_input.endswith('"')) or (
            user_input.startswith("'") and user_input.endswith("'")
        ):
            user_input = user_input[1:-1]

        validated = validate_document_path(user_input)
        if validated is not None:
            return validated

        print(
            "\nError: Invalid path or unsupported file type.\n"
            "Please enter a valid absolute or relative path to a supported document.\n"
        )
