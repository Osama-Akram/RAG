"""
Generation module for RAG system.

Handles LLM-based answer generation using OpenAI.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation operation."""

    answer: str
    sources: List[str]
    prompt_tokens: int
    completion_tokens: int


class Generator:
    """
    Generates answers using LLM with retrieved context.

    Uses OpenAI API with retry logic for reliability.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """
        Initialize generator.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Optional base URL for API
            max_tokens: Maximum tokens in response
            temperature: Temperature for sampling (0 = deterministic)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate answer using context.

        Args:
            query: User question
            context: Retrieved context to use
            system_prompt: Optional system prompt override

        Returns:
            GenerationResult with answer and metadata
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Use the provided context to answer "
                "the user's question. If the answer is not in the context, say "
                "you don't know. Be concise and accurate."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        # Extract source files from context
        sources = self._extract_sources(context)

        return GenerationResult(
            answer=answer,
            sources=sources,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def generate_with_sources(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate answer using retrieval result.

        Args:
            query: User question
            retrieval_result: Result from retriever
            system_prompt: Optional system prompt override

        Returns:
            GenerationResult with answer
        """
        context = retrieval_result.context
        result = self.generate(query, context, system_prompt)
        result.sources = list(set(chunk.source_file for chunk in retrieval_result.chunks))
        return result

    def _extract_sources(self, context: str) -> List[str]:
        """Extract source file names from context."""
        # This is a simple heuristic - in practice, sources
        # should be tracked separately
        import re
        sources = re.findall(r"Source:\s*(\S+)", context)
        return list(set(sources))
