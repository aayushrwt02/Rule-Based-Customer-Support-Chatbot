"""ChromaDB vector database manager."""

from typing import Any

import chromadb
from chromadb.config import Settings

from src.preprocessing.chunker import TextChunk
from src.utils.config import COLLECTION_NAME, VECTOR_DB_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChromaManager:
    """Manage persistent ChromaDB storage for document chunks."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy-initialize the ChromaDB persistent client."""
        if self._client is None:
            VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(VECTOR_DB_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB client initialized at %s", VECTOR_DB_DIR)
        return self._client

    @property
    def collection(self):
        """Get or create the document collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": 16,
                    "hnsw:construction_ef": 100,
                    "description": "Official knowledge base from uploaded documents/",
                    "source": "documents/",
                },
            )
            logger.info("ChromaDB collection '%s' ready.", COLLECTION_NAME)
        return self._collection

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store text chunks with their embeddings and metadata."""
        if not chunks:
            logger.warning("No chunks to add to the database.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._build_metadata(chunk) for chunk in chunks]

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info("Stored %d chunk(s) in ChromaDB.", len(chunks))

    def query_semantic(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        section_titles: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic similarity search, optionally filtered by section."""
        count = self.collection.count()
        if count == 0:
            return []

        where_filter = None
        if section_titles and len(section_titles) <= 3:
            where_filter = {"section_title": {"$in": section_titles}}

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            try:
                filtered_count = self.collection.count(where=where_filter)
                if filtered_count > 0:
                    query_kwargs["where"] = where_filter
                    query_kwargs["n_results"] = min(top_k, filtered_count)
            except Exception:
                pass

        results = self.collection.query(**query_kwargs)

        return self._format_results(results)

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Retrieve all stored documents for keyword/regex search."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        formatted: list[dict[str, Any]] = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                formatted.append(
                    {
                        "chunk_id": doc_id,
                        "text": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        "score": 0.0,
                    }
                )

        return formatted

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    @staticmethod
    def _build_metadata(chunk: TextChunk) -> dict[str, str | int | float | bool]:
        """Build ChromaDB-compatible metadata from a TextChunk."""
        metadata: dict[str, str | int | float | bool] = {
            "document_name": chunk.document_name,
            "file_type": chunk.file_type,
            "chunk_id": chunk.chunk_id,
        }

        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number

        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        return metadata

    @staticmethod
    def _format_results(results: dict) -> list[dict[str, Any]]:
        """Format ChromaDB query results into a standard structure."""
        formatted: list[dict[str, Any]] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            # Convert cosine distance to similarity score (0-1)
            similarity = max(0.0, 1.0 - distance)

            formatted.append(
                {
                    "chunk_id": doc_id,
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": similarity,
                }
            )

        return formatted
