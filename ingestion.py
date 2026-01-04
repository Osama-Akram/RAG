"""
Document ingestion module for RAG system.

Handles loading, chunking, and metadata extraction from documents.
Supports PDF and TXT file formats with configurable chunking strategies.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pypdf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text with associated metadata."""

    text: str
    chunk_id: str
    source_file: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """Represents a complete document with all its chunks."""

    source_file: str
    chunks: List[DocumentChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert document to dictionary for serialization."""
        return {
            "source_file": self.source_file,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


class DocumentLoader:
    """Base class for document loaders."""

    def load(self, file_path: str) -> str:
        """Load document content from file."""
        raise NotImplementedError


class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""

    def load(self, file_path: str) -> str:
        """
        Extract text content from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If PDF parsing fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            with open(path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                text_parts = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(text)
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num} in {file_path}: {e}"
                        )
                        continue

                full_text = "\n\n".join(text_parts)
                logger.info(f"Loaded PDF: {file_path} ({len(full_text)} chars)")
                return full_text

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise


class TXTLoader(DocumentLoader):
    """Loader for plain text documents."""

    def load(self, file_path: str) -> str:
        """
        Load text content from TXT file.

        Args:
            file_path: Path to TXT file

        Returns:
            File content

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                logger.info(f"Loaded TXT: {file_path} ({len(text)} chars)")
                return text
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            try:
                with open(path, "r", encoding="latin-1") as file:
                    text = file.read()
                    logger.info(f"Loaded TXT (latin-1): {file_path} ({len(text)} chars)")
                    return text
            except Exception as e:
                logger.error(f"Error loading TXT {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise


class Chunker:
    """
    Splits documents into overlapping chunks for effective retrieval.

    Uses sentence-aware chunking when possible to preserve semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 50,
    ):
        """
        Initialize chunker with configuration.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Document text to chunk
            source_file: Name of source file for metadata

        Returns:
            List of DocumentChunk objects

        Note:
            Uses sentence boundary detection where possible to avoid
            breaking mid-sentence. Falls back to character-based splitting.
        """
        chunks = []

        # Try sentence-aware chunking first
        chunks = self._sentence_aware_chunk(text, source_file)

        # If sentence-aware chunking failed or produced no chunks, fall back
        if not chunks:
            chunks = self._simple_chunk(text, source_file)

        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    def _sentence_aware_chunk(
        self, text: str, source_file: str
    ) -> List[DocumentChunk]:
        """
        Chunk text while respecting sentence boundaries.

        Args:
            text: Document text
            source_file: Source file name

        Returns:
            List of chunks
        """
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            potential_chunk = (
                current_chunk + " " + sentence if current_chunk else sentence
            )

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's large enough
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_id = self._generate_chunk_id(source_file, chunk_index)
                    chunk = DocumentChunk(
                        text=current_chunk.strip(),
                        chunk_id=chunk_id,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        metadata={"char_count": len(current_chunk)},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                current_chunk = self._add_overlap(current_chunk, sentence)

        # Add final chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunk_id = self._generate_chunk_id(source_file, chunk_index)
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                source_file=source_file,
                chunk_index=chunk_index,
                metadata={"char_count": len(current_chunk)},
            )
            chunks.append(chunk)

        return chunks

    def _simple_chunk(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Simple character-based chunking with overlap.

        Args:
            text: Document text
            source_file: Source file name

        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = self._generate_chunk_id(source_file, chunk_index)
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    metadata={"char_count": len(chunk_text)},
                )
                chunks.append(chunk)
                chunk_index += 1

            start = end - self.chunk_overlap

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using punctuation patterns.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        import re

        # Simple sentence splitting - can be enhanced with spacy/nltk
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _add_overlap(self, previous_chunk: str, next_sentence: str) -> str:
        """
        Add overlap from previous chunk to next.

        Args:
            previous_chunk: Text of previous chunk
            next_sentence: Text to start new chunk with

        Returns:
            Text for new chunk with overlap
        """
        if len(previous_chunk) <= self.chunk_overlap:
            return next_sentence

        overlap_text = previous_chunk[-self.chunk_overlap:]
        return overlap_text + " " + next_sentence

    @staticmethod
    def _generate_chunk_id(source_file: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        unique_string = f"{source_file}_{chunk_index}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]


class DocumentIngestor:
    """
    Main orchestrator for document ingestion pipeline.

    Handles loading documents from directory and creating chunks.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize document ingestor.

        Args:
            data_dir: Directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.loaders = {
            ".pdf": PDFLoader(),
            ".txt": TXTLoader(),
        }

    def ingest(self) -> List[Document]:
        """
        Load and chunk all documents from data directory.

        Returns:
            List of Document objects with chunks

        Raises:
            ValueError: If data directory doesn't exist
        """
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        documents = []
        file_paths = self._get_document_files()

        logger.info(f"Found {len(file_paths)} documents in {self.data_dir}")

        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                document = self._process_document(file_path)
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        total_chunks = sum(len(doc.chunks) for doc in documents)
        logger.info(f"Ingested {len(documents)} documents with {total_chunks} chunks")

        return documents

    def _get_document_files(self) -> List[Path]:
        """Get all supported document files from data directory."""
        files = []

        for ext in self.loaders.keys():
            files.extend(self.data_dir.glob(f"*{ext}"))
            files.extend(self.data_dir.glob(f"*{ext.upper()}"))

        return sorted(files)

    def _process_document(self, file_path: Path) -> Document:
        """
        Process a single document: load and chunk.

        Args:
            file_path: Path to document

        Returns:
            Document object with chunks
        """
        ext = file_path.suffix.lower()

        if ext not in self.loaders:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = self.loaders[ext]
        text = loader.load(str(file_path))

        if not text.strip():
            logger.warning(f"Empty document: {file_path}")
            return Document(source_file=file_path.name, chunks=[])

        chunks = self.chunker.chunk(text, file_path.name)

        return Document(source_file=file_path.name, chunks=chunks)

    def save_documents(self, documents: List[Document], output_path: str) -> None:
        """
        Save documents to JSON file for inspection/debugging.

        Args:
            documents: List of Document objects
            output_path: Path to save JSON
        """
        data = [doc.to_dict() for doc in documents]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(documents)} documents to {output_path}")
