"""Local embedding generation using SentenceTransformers."""

import os
from typing import Sequence

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    """Wrapper for local SentenceTransformer embedding generation."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            logger.info("Loading embedding model: %s (this may take a moment)...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully.")
        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
        """Generate embeddings for a list of text strings."""
        if not texts:
            return []

        logger.info("Generating embeddings for %d text(s)...", len(texts))
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()

        return [emb.tolist() for emb in embeddings]

    def encode_query(self, query: str) -> list[float]:
        """Generate a normalized embedding for a single query string."""
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding.tolist()
