"""
Retrieval module for RAG system.

Handles retrieving relevant document chunks for a given query.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from ingestion import DocumentChunk
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    chunks: List[DocumentChunk]
    scores: List[float]
    query: str

    @property
    def context(self) -> str:
        """Get concatenated context from chunks."""
        return "\n\n---\n\n".join(chunk.text for chunk in self.chunks)


class Retriever:
    """
    Retrieves relevant document chunks for queries.

    Uses vector similarity search with configurable top-k.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store to search
            top_k: Number of results to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Override number of results

        Returns:
            RetrievalResult with chunks and scores
        """
        k = top_k if top_k is not None else self.top_k
        results = self.vector_store.search(query, top_k=k)

        chunks = [chunk for chunk, _ in results]
        scores = [score for _, score in results]

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            query=query,
        )

    def retrieve_with_filter(
        self,
        query: str,
        source_file: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve chunks with optional source file filter.

        Args:
            query: Query text
            source_file: Optional source file name to filter by
            top_k: Override number of results

        Returns:
            RetrievalResult with filtered chunks
        """
        result = self.retrieve(query, top_k)

        if source_file:
            filtered_chunks = [
                chunk for chunk in result.chunks
                if chunk.source_file == source_file
            ]
            # Recalculate scores for filtered results
            filtered_scores = [
                score for chunk, score in zip(result.chunks, result.scores)
                if chunk.source_file == source_file
            ]

            # Adjust to match filtered chunks
            result.chunks = filtered_chunks
            result.scores = filtered_scores

        return result
