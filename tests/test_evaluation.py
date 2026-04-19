"""Tests for evaluation metrics (retrieval + generation)."""

import math

import pytest

from src.evaluation.generation_metrics import (
    _parse_json,
    answer_context_overlap,
    citation_accuracy,
    compute_programmatic,
)
from src.evaluation.retrieval_metrics import (
    average_precision,
    compute_all,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5  # 2/4

    def test_k_larger_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # k=5 but only 2 retrieved → 2/5
        assert precision_at_k(retrieved, relevant, k=5) == 0.4

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=3) == 0.0

    def test_k_one(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=1) == 1.0

    def test_k_one_miss(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=1) == 0.0


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_found(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=4) == 1.0

    def test_none_found(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, k=3) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_hit_at_rank_1(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_first_hit_at_rank_3(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_uses_first(self):
        # MRR only cares about the first relevant hit
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# average precision
# ---------------------------------------------------------------------------

class TestAveragePrecision:
    def test_perfect_ranking(self):
        # All relevant docs at the top
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        # AP = (1/3) * (1/1 + 2/2 + 3/3) = 1.0
        assert average_precision(retrieved, relevant) == 1.0

    def test_worst_ranking(self):
        # All relevant docs at the bottom
        retrieved = ["x", "y", "a", "b", "c"]
        relevant = {"a", "b", "c"}
        # AP = (1/3) * (1/3 + 2/4 + 3/5) = (1/3)*(0.333 + 0.5 + 0.6) = 0.4778
        expected = (1 / 3) * (1 / 3 + 2 / 4 + 3 / 5)
        assert average_precision(retrieved, relevant) == pytest.approx(expected)

    def test_single_relevant(self):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        assert average_precision(retrieved, relevant) == 0.5  # 1/2

    def test_no_relevant_found(self):
        assert average_precision(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert average_precision(["a", "b"], set()) == 0.0


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c", "x"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_one_relevant_at_first(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        # DCG = 1/log2(2) = 1.0, IDCG = 1/log2(2) = 1.0
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_one_relevant_at_third(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        # DCG = 1/log2(4), IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(4)) / 1.0
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_all_expected_keys(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        result = compute_all(retrieved, relevant, k_values=[1, 3, 5])

        assert "mrr" in result
        for k in [1, 3, 5]:
            assert f"precision@{k}" in result
            assert f"recall@{k}" in result
            assert f"ndcg@{k}" in result
            assert f"ap@{k}" in result

    def test_values_are_consistent(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = compute_all(retrieved, relevant, k_values=[3])

        assert result["mrr"] == 1.0
        assert result["precision@3"] == 1.0
        assert result["recall@3"] == 1.0
        assert result["ndcg@3"] == 1.0


# ===========================================================================
# Generation metrics (programmatic — no API calls)
# ===========================================================================

# ---------------------------------------------------------------------------
# citation_accuracy
# ---------------------------------------------------------------------------

class TestCitationAccuracy:
    def test_all_valid(self):
        answer = "The court ruled [1] in favor of the plaintiff [2]."
        result = citation_accuracy(answer, num_passages=3)
        assert result["total_citations"] == 2
        assert result["valid_citations"] == 2
        assert result["accuracy"] == 1.0

    def test_some_invalid(self):
        answer = "According to [1] and [5], the law applies."
        result = citation_accuracy(answer, num_passages=3)
        assert result["total_citations"] == 2
        assert result["valid_citations"] == 1
        assert result["accuracy"] == 0.5

    def test_all_invalid(self):
        answer = "Sources [10] and [20] confirm this."
        result = citation_accuracy(answer, num_passages=3)
        assert result["total_citations"] == 2
        assert result["valid_citations"] == 0
        assert result["accuracy"] == 0.0

    def test_no_citations(self):
        answer = "The answer is simply that the law applies here."
        result = citation_accuracy(answer, num_passages=3)
        assert result["total_citations"] == 0
        assert result["accuracy"] == 1.0

    def test_repeated_citation(self):
        answer = "Per [1], the ruling [1] was clear."
        result = citation_accuracy(answer, num_passages=2)
        assert result["total_citations"] == 2
        assert result["valid_citations"] == 2

    def test_boundary_passage_number(self):
        answer = "Source [3] states this."
        assert citation_accuracy(answer, num_passages=3)["valid_citations"] == 1
        assert citation_accuracy(answer, num_passages=2)["valid_citations"] == 0


# ---------------------------------------------------------------------------
# answer_context_overlap
# ---------------------------------------------------------------------------

class TestAnswerContextOverlap:
    def test_full_overlap(self):
        context = "the cat sat on the mat"
        answer = "the cat sat on the mat"
        assert answer_context_overlap(answer, context) == 1.0

    def test_no_overlap(self):
        context = "the cat sat on the mat"
        answer = "quantum entanglement discovered"
        assert answer_context_overlap(answer, context) == 0.0

    def test_partial_overlap(self):
        context = "the cat sat on the mat"
        answer = "the cat jumped over the fence"
        # answer tokens: {the, cat, jumped, over, fence} = 5
        # overlap: {the, cat} = 2, but "the" and "cat" → 2/5
        overlap = answer_context_overlap(answer, context)
        assert 0.0 < overlap < 1.0

    def test_empty_answer(self):
        assert answer_context_overlap("", "some context") == 0.0

    def test_case_insensitive(self):
        assert answer_context_overlap("The CAT", "the cat sat") == 1.0


# ---------------------------------------------------------------------------
# _parse_json helper
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"score": 0.8}') == {"score": 0.8}

    def test_markdown_fenced(self):
        text = '```json\n{"score": 0.8}\n```'
        assert _parse_json(text) == {"score": 0.8}

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"score": 0.8}\n```'
        assert _parse_json(text) == {"score": 0.8}

    def test_whitespace_padding(self):
        assert _parse_json('  {"score": 0.8}  ') == {"score": 0.8}


# ---------------------------------------------------------------------------
# compute_programmatic
# ---------------------------------------------------------------------------

class TestComputeProgrammatic:
    def test_returns_all_keys(self):
        result = compute_programmatic(
            answer="The ruling [1] confirms [2] the decision.",
            context="The ruling confirms the decision was made.",
            num_passages=3,
        )
        assert "citation_total" in result
        assert "citation_valid" in result
        assert "citation_accuracy" in result
        assert "context_overlap" in result

    def test_values_consistent(self):
        result = compute_programmatic(
            answer="completely unrelated quantum physics",
            context="the court ruled in favor of the plaintiff",
            num_passages=2,
        )
        assert result["citation_total"] == 0
        assert result["context_overlap"] == 0.0
