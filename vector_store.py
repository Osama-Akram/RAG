"""
Vector store module for RAG system.

Handles storing and retrieving document embeddings using FAISS.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from ingestion import Document, DocumentChunk
from embedding import EmbeddingEngine

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for document retrieval.

    Supports index saving/loading and metadata management.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        index_path: Optional[str] = None,
        index_type: str = "flat",
    ):
        """
        Initialize vector store.

        Args:
            embedding_engine: Embedding engine for encoding
            index_path: Path to save/load index
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.embedding_engine = embedding_engine
        self.index_path = Path(index_path) if index_path else None
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_to_index: dict = {}

        # Create index directory if needed
        if self.index_path:
            self.index_path.mkdir(parents=True, exist_ok=True)

        # Try to load existing index
        if self.index_path:
            self.load_index()

    def build_index(self, documents: List[Document]) -> None:
        """
        Build index from documents.

        Args:
            documents: List of Document objects with chunks
        """
        # Collect all texts and metadata
        all_texts = []
        all_chunks = []

        for doc in documents:
            for chunk in doc.chunks:
                all_texts.append(chunk.text)
                all_chunks.append(chunk)

        if not all_texts:
            logger.warning("No documents to index")
            return

        # Create embeddings
        embeddings = self.embedding_engine.embed_documents(all_texts)

        # Create FAISS index
        dimension = self.embedding_engine.dimension
        self.index = self._create_index(dimension, len(embeddings))

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))

        # Store chunks for retrieval
        self.chunks = all_chunks
        self.chunk_id_to_index = {
            chunk.chunk_id: i for i, chunk in enumerate(self.chunks)
        }

        logger.info(f"Built index with {len(self.chunks)} chunks")

        # Save index if path provided
        if self.index_path:
            self.save_index()

    def _create_index(self, dimension: int, n_vectors: int) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "flat":
            return faiss.IndexFlatL2(dimension)
        elif self.index_type == "ivf":
            nlist = min(256, n_vectors // 39)
            quantizer = faiss.IndexFlatL2(dimension)
            return faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            logger.warning(f"Unknown index type: {self.index_type}, using flat")
            return faiss.IndexFlatL2(dimension)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (chunk, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")

        # Embed query
        query_embedding = self.embedding_engine.embed_query(query).astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Return results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((chunk, float(dist)))

        return results

    def save_index(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None or not self.index_path:
            return

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

        # Save chunks metadata
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(self.index_path / "chunks.json", "w") as f:
            json.dump(chunks_data, f)

        logger.info(f"Saved index to {self.index_path}")

    def load_index(self) -> bool:
        """Load index and metadata from disk."""
        if not self.index_path:
            return False

        index_file = self.index_path / "index.faiss"
        chunks_file = self.index_path / "chunks.json"

        if not index_file.exists() or not chunks_file.exists():
            logger.warning(f"Index files not found at {self.index_path}")
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_file))

        # Load chunks
        with open(chunks_file, "r") as f:
            chunks_data = json.load(f)

        self.chunks = [
            DocumentChunk(**data) for data in chunks_data
        ]
        self.chunk_id_to_index = {
            chunk.chunk_id: i for i, chunk in enumerate(self.chunks)
        }

        logger.info(f"Loaded index with {len(self.chunks)} chunks")
        return True

    @property
    def num_vectors(self) -> int:
        """Return number of vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal
