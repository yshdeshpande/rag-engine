"""LLM generation with retrieved context.

Supports three providers:
  - ollama: Free, local inference via Ollama (default)
  - anthropic: Claude API
  - openai: OpenAI API

All providers use the same prompt structure from prompts.py.
Ollama is called via its OpenAI-compatible REST API, so no extra
packages are needed beyond httpx.
"""

from dataclasses import dataclass

import httpx

from src.chunking.base import Chunk
from src.generation.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_user_message,
    format_context,
)


@dataclass
class GenerationResult:
    """Container for a generated answer plus metadata."""

    answer: str
    model: str
    usage: dict  # token counts from the API
    context_passages: str  # the formatted context sent to the LLM


class RAGGenerator:
    """Generate answers from retrieved chunks using an LLM.

    Provider routing:
      - "ollama"    -> local Ollama server (OpenAI-compatible endpoint)
      - "anthropic" -> Anthropic Messages API
      - "openai"    -> OpenAI Chat Completions API
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key

    def generate(
        self,
        query: str,
        retrieved_chunks: list[tuple[Chunk, float]],
    ) -> GenerationResult:
        """Build prompt from retrieved chunks and call the LLM."""
        if not retrieved_chunks:
            return GenerationResult(
                answer=NO_CONTEXT_ANSWER,
                model=self.model,
                usage={},
                context_passages="",
            )

        context = format_context(retrieved_chunks)
        user_message = build_user_message(query, context)

        if self.provider == "ollama":
            return self._call_ollama(user_message, context)
        elif self.provider == "anthropic":
            return self._call_anthropic(user_message, context)
        elif self.provider == "openai":
            return self._call_openai(user_message, context)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, user_message: str, context: str) -> GenerationResult:
        """Call Ollama via its OpenAI-compatible /v1/chat/completions endpoint."""
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            answer=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            context_passages=context,
        )

    def _call_anthropic(self, user_message: str, context: str) -> GenerationResult:
        """Call Anthropic Messages API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return GenerationResult(
            answer=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            context_passages=context,
        )

    def _call_openai(self, user_message: str, context: str) -> GenerationResult:
        """Call OpenAI Chat Completions API."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        choice = response.choices[0]
        return GenerationResult(
            answer=choice.message.content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            context_passages=context,
        )
