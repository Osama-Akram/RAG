"""
Embedding module for RAG system.

Handles creating vector embeddings for document chunks using sentence-transformers.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Creates embeddings for text using sentence-transformers.

    Supports CPU and GPU inference with configurable model.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: Name of sentence-transformers model
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dim={self.embedding_dim})")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Embedded {len(texts)} documents")
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query.

        Args:
            text: Query text

        Returns:
            numpy array of embedding with shape (1, embedding_dim)
        """
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
