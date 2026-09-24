import os
import shutil
import sys
import warnings
from pathlib import Path

# ============================================================
# Environment settings
# ============================================================

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests.*"
)


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports from your project
# ============================================================

from src.chatbot.chatbot import Chatbot
from src.database.chroma_manager import ChromaManager
from src.embeddings.embedding_model import EmbeddingModel
from src.preprocessing.chunker import TextChunker
from src.preprocessing.cleaner import TextCleaner
from src.readers import read_document
from src.search.retriever import HybridRetriever
from src.search.query_mapper import QueryMapper
from src.preprocessing.section_registry import SectionRegistry

from src.utils.config import VECTOR_DB_DIR

from src.utils.file_manager import (
    copy_document_to_storage,
    detect_file_type,
    ensure_directories,
    get_stored_documents,
    has_processed_data,
    prompt_for_document_path,
    save_index_version,
)

from src.utils.logger import setup_logger


# ============================================================
# Logger
# ============================================================

logger = setup_logger("main")


# ============================================================
# Vector database
# ============================================================

def _clear_vector_db() -> None:
    """
    Remove the existing vector database for re-indexing.
    """

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info("Cleared vector database for re-indexing.")


# ============================================================
# Prepare document chunks
# ============================================================

def _prepare_chunks_for_document(document_path: Path) -> list:
    """
    Read, clean, and chunk a single document without embedding.
    """

    file_type = detect_file_type(document_path)
    document_name = document_path.name

    segments = read_document(document_path)

    if not segments:
        raise RuntimeError(
            f"No text could be extracted from: {document_name}"
        )

    # Clean text
    cleaner = TextCleaner()

    for segment in segments:
        segment.text = cleaner.clean(segment.text)

    # Remove empty segments
    segments = [
        segment
        for segment in segments
        if segment.text.strip()
    ]

    if not segments:
        raise RuntimeError(
            f"No usable text in: {document_name}"
        )

    # Chunk text
    chunker = TextChunker()

    chunks = chunker.chunk_segments(
        segments,
        document_name,
        file_type
    )

    if not chunks:
        raise RuntimeError(
            f"No chunks created from: {document_name}"
        )

    logger.info(
        "Prepared %d chunk(s) from %s.",
        len(chunks),
        document_name
    )

    return chunks


# ============================================================
# Index all documents
# ============================================================

def index_all_documents(show_progress: bool = False) -> int:
    """
    Index every document in documents/ with a single
    embedding batch.
    """

    documents = get_stored_documents()

    if not documents:
        raise RuntimeError(
            "No documents found in documents/ folder."
        )

    _clear_vector_db()

    all_chunks = []

    for i, doc_path in enumerate(documents, 1):

        if show_progress:
            print(
                f"  [{i}/{len(documents)}] "
                f"Reading and chunking {doc_path.name}..."
            )

        all_chunks.extend(
            _prepare_chunks_for_document(doc_path)
        )

    if show_progress:
        print(
            f"  Generating embeddings for "
            f"{len(all_chunks)} sections..."
        )

    # Create embeddings
    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        [chunk.text for chunk in all_chunks]
    )

    # Store in Chroma
    chroma_manager = ChromaManager()

    chroma_manager.add_chunks(
        all_chunks,
        embeddings
    )

    # Store section titles
    section_titles = [
        str(chunk.metadata.get("section_title", ""))
        for chunk in all_chunks
        if chunk.metadata.get("section_title")
    ]

    SectionRegistry.clear_and_set(
        section_titles
    )

    QueryMapper.clear_cache()

    save_index_version()

    logger.info(
        "Indexed %d document(s), %d total chunk(s).",
        len(documents),
        len(all_chunks)
    )

    return len(all_chunks)


# ============================================================
# CLI document setup
# ============================================================

def setup_processed_data() -> None:
    """
    Ensure processed data exists and matches documents/ folder.

    This function is used by the terminal/CLI application.
    """

    if has_processed_data():

        docs = get_stored_documents()

        logger.info(
            "Processed data ready — %d document(s) indexed.",
            len(docs)
        )

        return

    documents = get_stored_documents()

    # Documents already exist
    if documents:

        print(
            f"\nIndexing {len(documents)} document(s) "
            "from documents/ folder..."
        )

        print("Please wait...\n")

        total = index_all_documents(
            show_progress=True
        )

        print(
            f"Processing complete "
            f"({total} sections indexed). "
            "Starting chatbot...\n"
        )

        return

    # No documents - CLI asks user for one
    logger.info(
        "No documents in folder — prompting for upload."
    )

    source_path = prompt_for_document_path()

    stored_path = copy_document_to_storage(
        source_path
    )

    print(
        f"\nDocument saved: {stored_path.name}"
    )

    print(
        "Processing your document, please wait...\n"
    )

    index_all_documents(
        show_progress=True
    )

    print(
        "Processing complete. Starting chatbot...\n"
    )


# ============================================================
# CREATE CHATBOT
# ============================================================

def create_chatbot() -> Chatbot:
    """
    Create and initialize the Chatbot.

    This function is shared by:
        - CLI
        - Streamlit UI
    """

    ensure_directories()

    logger.info(
        "Initializing chatbot..."
    )

    chroma_manager = ChromaManager()

    embedding_model = EmbeddingModel()

    # Warm up embedding model
    embedding_model.encode_query(
        "warmup"
    )

    retriever = HybridRetriever(
        chroma_manager,
        embedding_model
    )

    chatbot = Chatbot(
        retriever
    )

    logger.info(
        "Chatbot initialized successfully."
    )

    return chatbot


# ============================================================
# STREAMLIT INITIALIZATION
# ============================================================

def initialize_for_ui() -> Chatbot:
    """
    Initialize the chatbot for Streamlit.

    Unlike setup_processed_data(), this function NEVER calls
    input() or asks for a document through the terminal.

    If documents exist but are not indexed, they are indexed.
    If no documents exist, it raises an error so the Streamlit
    UI can ask the user to upload one.
    """

    ensure_directories()

    documents = get_stored_documents()

    # Nothing uploaded yet
    if not documents:
        raise RuntimeError(
            "No documents found. "
            "Please upload a document first."
        )

    # Process documents if needed
    if not has_processed_data():

        logger.info(
            "No processed data found. "
            "Indexing documents for Streamlit..."
        )

        index_all_documents(
            show_progress=False
        )

    return create_chatbot()


# ============================================================
# CHATBOT RESPONSE HELPER
# ============================================================

def ask_chatbot(chatbot: Chatbot, question: str):
    """
    Send a question to the Chatbot.

    This supports several common method names so the UI can
    work with your existing Chatbot implementation.

    Supported methods:
        ask()
        chat()
        get_response()
        generate_response()
        answer()
        respond()
    """

    possible_methods = [
        "ask",
        "chat",
        "get_response",
        "generate_response",
        "answer",
        "respond",
    ]

    for method_name in possible_methods:

        method = getattr(
            chatbot,
            method_name,
            None
        )

        if callable(method):

            logger.info(
                "Using Chatbot.%s()",
                method_name
            )

            return method(question)

    raise AttributeError(
        "Your Chatbot class does not have a supported "
        "question-answer method.\n\n"
        "Please add a method such as:\n\n"
        "    def ask(self, question):\n"
        "        ...\n\n"
        "to src/chatbot/chatbot.py"
    )


# ============================================================
# CLI APPLICATION
# ============================================================

def main() -> None:
    """
    Terminal/CLI application entry point.
    """

    logger.info(
        "Application started."
    )

    try:

        setup_processed_data()

        chatbot = create_chatbot()

        # Start terminal chatbot
        chatbot.start()

    except KeyboardInterrupt:

        print(
            "\nGoodbye!"
        )

    except Exception as exc:

        logger.exception(
            "Application error: %s",
            exc
        )

        print(
            "\nSomething went wrong. "
            "Please try again or check "
            "logs/chatbot.log for details."
        )


# ============================================================
# RUN CLI ONLY
# ============================================================

if __name__ == "__main__":
    main()