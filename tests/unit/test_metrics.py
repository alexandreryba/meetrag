import pytest

from polyrag.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    set_precision_recall,
)

RANKED = ["a", "b", "a", "c", "d"]


def test_hit_recall_and_mrr() -> None:
    assert hit_at_k(RANKED, {"c"}, 3) == 0.0
    assert hit_at_k(RANKED, {"c"}, 4) == 1.0
    assert recall_at_k(RANKED, {"a", "c", "z"}, 5) == pytest.approx(2 / 3)
    assert reciprocal_rank(RANKED, {"b"}) == 0.5
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_ndcg_counts_each_document_once() -> None:
    assert ndcg_at_k(["a", "x"], {"a"}, 2) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / 1.585, rel=1e-3)
    assert ndcg_at_k(["a", "a"], {"a", "b"}, 2) == pytest.approx(1 / (1 + 1 / 1.585), rel=1e-3)


def test_set_precision_recall() -> None:
    assert set_precision_recall(["policies", "support"], ["policies"]) == (0.5, 1.0)
    assert set_precision_recall([], []) == (1.0, 1.0)
    assert set_precision_recall([], ["analytics"]) == (0.0, 0.0)


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) == 0.0
