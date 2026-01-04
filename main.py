#!/usr/bin/env python3
"""
Main RAG system entry point.

A complete Retrieval-Augmented Generation system for document question answering.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingestion import DocumentIngestor
from embedding import EmbeddingEngine
from vector_store import VectorStore
from retrieval import Retriever
from generation import Generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Main RAG system class orchestrating all components."""

    def __init__(self, config: dict):
        """Initialize RAG system with configuration."""
        self.config = config

        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(
            model_name=config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            device=config.get("EMBEDDING_DEVICE", "cpu"),
        )

        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_engine=self.embedding_engine,
            index_path=config.get("VECTOR_STORE_PATH"),
            index_type=config.get("INDEX_TYPE", "flat"),
        )

        # Initialize retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=config.get("TOP_K", 5),
        )

        # Initialize generator (only if API key available)
        self.generator = None
        if config.get("OPENAI_API_KEY"):
            self.generator = Generator(
                api_key=config["OPENAI_API_KEY"],
                model=config.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                base_url=config.get("OPENAI_BASE_URL"),
            )
        else:
            logger.warning("No OpenAI API key provided - generation disabled")

    def ingest(self, data_dir: str = "./data") -> None:
        """Ingest documents and build index."""
        logger.info(f"Ingesting documents from {data_dir}")

        # Load and chunk documents
        ingestor = DocumentIngestor(
            data_dir=data_dir,
            chunk_size=self.config.get("CHUNK_SIZE", 512),
            chunk_overlap=self.config.get("CHUNK_OVERLAP", 128),
        )
        documents = ingestor.ingest()

        # Build vector index
        self.vector_store.build_index(documents)

        logger.info(f"Ingestion complete. Indexed {self.vector_store.num_vectors} chunks")

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question using the RAG pipeline."""
        # Retrieve relevant chunks
        result = self.retriever.retrieve(question, top_k=top_k)

        if not result.chunks:
            return "No relevant documents found."

        # Generate answer
        if self.generator:
            gen_result = self.generator.generate_with_sources(
                question,
                result,
            )
            answer = gen_result.answer
        else:
            # Just return context if no generator
            answer = f"Found {len(result.chunks)} relevant passages:\n\n{result.context}"

        return answer

    def interactive(self) -> None:
        """Run interactive query session."""
        print("\n=== RAG System Ready ===")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                question = input("You: ").strip()
                if question.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                if not question:
                    continue

                answer = self.query(question)
                print(f"\nAssistant: {answer}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()

    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cpu"),
        "VECTOR_STORE_PATH": os.getenv("VECTOR_STORE_PATH", "./output/vector_store"),
        "INDEX_TYPE": os.getenv("INDEX_TYPE", "flat"),
        "TOP_K": int(os.getenv("TOP_K", 5)),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 512)),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 128)),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System for document question answering"
    )
    parser.add_argument(
        "--ingest",
        metavar="DIR",
        help="Ingest documents from directory",
    )
    parser.add_argument(
        "--query",
        metavar="QUESTION",
        help="Ask a single question",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Create RAG system
    rag = RAGSystem(config)

    if args.ingest:
        rag.ingest(args.ingest)
    elif args.query:
        answer = rag.query(args.query, top_k=args.top_k)
        print(f"\n{answer}")
    elif args.interactive:
        rag.interactive()
    else:
        # Default: run interactive mode
        rag.interactive()


if __name__ == "__main__":
    main()
