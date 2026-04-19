"""Generation evaluation metrics: faithfulness, relevance, citation accuracy.

Two categories of metrics:

  Programmatic (no API calls):
    - citation_accuracy: checks [N] references map to real passage numbers
    - answer_context_overlap: token-level grounding signal

  LLM-as-judge (requires Anthropic API key):
    - faithfulness: extracts claims from the answer, verifies each against context
    - answer_relevance: scores how well the answer addresses the question

The LLM-judge approach follows the RAGAS-style decomposition: break the
answer into atomic claims, then verify each independently. This gives a
fine-grained score (0.0–1.0) rather than a binary pass/fail.
"""

import json
import re

# ---------------------------------------------------------------------------
# Programmatic metrics (no API calls)
# ---------------------------------------------------------------------------

def citation_accuracy(answer: str, num_passages: int) -> dict[str, float]:
    """Check whether citations in the answer reference valid passages.

    The RAG generator cites sources as [1], [2], etc. This checks:
      1. How many citation markers appear in the answer
      2. How many reference a passage that actually exists (1..num_passages)

    Returns:
        total_citations: count of [N] markers found
        valid_citations: count referencing a real passage
        accuracy: valid / total (1.0 if no citations found)
    """
    # Find all [N] patterns, excluding common false positives like [0]
    refs = re.findall(r"\[(\d+)\]", answer)

    if not refs:
        return {"total_citations": 0, "valid_citations": 0, "accuracy": 1.0}

    total = len(refs)
    valid = sum(1 for r in refs if 1 <= int(r) <= num_passages)

    return {
        "total_citations": float(total),
        "valid_citations": float(valid),
        "accuracy": valid / total,
    }


def answer_context_overlap(answer: str, context: str) -> float:
    """Fraction of answer tokens that also appear in the context.

    A simple grounding signal: high overlap suggests the answer draws
    from the context rather than hallucinating. Very cheap to compute.

    Note: this is a necessary but not sufficient condition for faithfulness.
    An answer can have high overlap but still misrepresent the context.
    """
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))

    if not answer_tokens:
        return 0.0

    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization, matching the sparse retriever's approach."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# LLM-as-judge metrics
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are an expert evaluator for retrieval-augmented generation systems.

Given a CONTEXT, a QUESTION, and an ANSWER, do the following:

1. Extract every atomic factual claim from the ANSWER.
   - An atomic claim is the smallest unit of information that can be
     independently verified (e.g., "The court ruled in favor of the defendant").
   - Ignore meta-statements like "According to the context" or "Based on the passages".

2. For each claim, determine if it is SUPPORTED by the context.
   - "supported": the context contains evidence that directly backs this claim.
   - "unsupported": the context does not contain evidence, or contradicts the claim.

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{
  "claims": [
    {"text": "claim text here", "verdict": "supported"},
    {"text": "another claim", "verdict": "unsupported"}
  ]
}

If the answer contains no factual claims (e.g., "I don't have enough information"),
return: {"claims": []}"""

_RELEVANCE_PROMPT = """\
You are an expert evaluator for retrieval-augmented generation systems.

Given a QUESTION and an ANSWER, score how well the answer addresses the question.

Scoring criteria:
- 1.0: The answer directly and completely addresses the question.
- 0.7–0.9: The answer addresses the question but misses some aspects or includes unnecessary detail.
- 0.4–0.6: The answer is partially relevant but misses key aspects of the question.
- 0.1–0.3: The answer is mostly irrelevant but contains a small relevant detail.
- 0.0: The answer is completely irrelevant or refuses to answer despite sufficient context.

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{"score": 0.85, "reasoning": "Brief explanation of the score"}"""


def faithfulness(
    question: str,
    answer: str,
    context: str,
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
) -> dict:
    """Score how faithful the answer is to the provided context.

    Decomposes the answer into atomic claims, then checks each against
    the context using an LLM judge. This is the RAGAS-style approach.

    Args:
        question: The original user question.
        answer: The generated answer to evaluate.
        context: The formatted context passages that were provided to the generator.
        model: Anthropic model to use as judge.
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).

    Returns:
        claims: list of {"text": str, "verdict": "supported"|"unsupported"}
        num_claims: total number of claims extracted
        num_supported: number of supported claims
        score: num_supported / num_claims (1.0 if no claims)
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}"""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        system=_FAITHFULNESS_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    result = _parse_json(response.content[0].text)
    claims = result.get("claims", [])

    num_claims = len(claims)
    num_supported = sum(1 for c in claims if c.get("verdict") == "supported")

    return {
        "claims": claims,
        "num_claims": num_claims,
        "num_supported": num_supported,
        "score": num_supported / num_claims if num_claims > 0 else 1.0,
    }


def answer_relevance(
    question: str,
    answer: str,
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
) -> dict:
    """Score how well the answer addresses the question.

    Uses an LLM judge to produce a 0.0–1.0 relevance score with reasoning.

    Args:
        question: The original user question.
        answer: The generated answer to evaluate.
        model: Anthropic model to use as judge.
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).

    Returns:
        score: float 0.0–1.0
        reasoning: brief explanation from the judge
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"""QUESTION:
{question}

ANSWER:
{answer}"""

    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        system=_RELEVANCE_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    result = _parse_json(response.content[0].text)

    return {
        "score": float(result.get("score", 0.0)),
        "reasoning": result.get("reasoning", ""),
    }


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def compute_programmatic(
    answer: str,
    context: str,
    num_passages: int,
) -> dict[str, float]:
    """Run all programmatic (free) generation metrics. Returns a flat dict."""
    citations = citation_accuracy(answer, num_passages)
    overlap = answer_context_overlap(answer, context)

    return {
        "citation_total": citations["total_citations"],
        "citation_valid": citations["valid_citations"],
        "citation_accuracy": citations["accuracy"],
        "context_overlap": overlap,
    }


def compute_llm_judge(
    question: str,
    answer: str,
    context: str,
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
) -> dict[str, float]:
    """Run all LLM-judge generation metrics. Returns a flat dict.

    Requires an Anthropic API key (set ANTHROPIC_API_KEY or pass api_key).
    """
    faith = faithfulness(question, answer, context, model=model, api_key=api_key)
    relevance = answer_relevance(question, answer, model=model, api_key=api_key)

    return {
        "faithfulness": faith["score"],
        "faithfulness_claims": faith["num_claims"],
        "faithfulness_supported": faith["num_supported"],
        "answer_relevance": relevance["score"],
        "answer_relevance_reasoning": relevance["reasoning"],
    }
