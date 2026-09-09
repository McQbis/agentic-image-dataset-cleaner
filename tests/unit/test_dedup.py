from __future__ import annotations

from core.tools.dedup import _build_clusters, deduplicate
from core.schemas import SampleMetrics


def _sample(
    sample_id: str,
    phash: str,
    blur_variance: float = 100.0,
) -> SampleMetrics:
    return SampleMetrics(
        sample_id=sample_id,
        path=f"/fake/{sample_id}.jpg",
        phash=phash,
        blur_variance=blur_variance,
    )


def test_build_clusters_merges_transitive_duplicates_into_one_cluster():
    """Transitive duplicates must form a single connected component.

    A is similar to B and B is similar to C, while A is not directly
    similar to C. All three samples must still belong to the same cluster.
    """
    dup_map = {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B"],
    }

    clusters = _build_clusters(dup_map)

    assert len(clusters) == 1
    assert clusters[0] == {"A", "B", "C"}


def test_build_clusters_keeps_disjoint_pairs_separate():
    """Disjoint duplicate groups must remain separate clusters."""
    dup_map = {
        "A": ["B"],
        "B": ["A"],
        "X": ["Y"],
        "Y": ["X"],
    }

    clusters = _build_clusters(dup_map)

    assert len(clusters) == 2
    assert {"A", "B"} in clusters
    assert {"X", "Y"} in clusters


def test_deduplicate_keeps_sharper_representative():
    """Deduplication must keep the sharpest sample in a duplicate cluster."""
    same_hash = "0" * 16

    sharp = _sample(
        "sharp",
        same_hash,
        blur_variance=2000.0,
    )
    blurry = _sample(
        "blurry",
        same_hash,
        blur_variance=50.0,
    )

    result = deduplicate([sharp, blurry])

    assert "sharp" in result.kept_sample_ids
    assert "blurry" in result.removed_sample_ids
    assert len(result.clusters) == 1


def test_deduplicate_keeps_unrelated_samples():
    """Unrelated samples must never be removed by deduplication."""
    standalone_a = _sample(
        "standalone_a",
        "aaaaaaaaaaaaaaaa",
    )
    standalone_b = _sample(
        "standalone_b",
        "bbbbbbbbbbbbbbbb",
    )

    result = deduplicate([standalone_a, standalone_b])

    assert set(result.kept_sample_ids) == {
        "standalone_a",
        "standalone_b",
    }
    assert result.removed_sample_ids == []
    assert result.clusters == []


def test_deduplicate_removes_corrupt_samples():
    """Corrupt samples must be excluded from the kept dataset."""
    corrupt = SampleMetrics(
        sample_id="bad",
        path="/fake/bad.jpg",
        is_corrupt=True,
        error="boom",
    )
    good = _sample(
        "good",
        "cccccccccccccccc",
    )

    result = deduplicate([corrupt, good])

    assert "good" in result.kept_sample_ids
    assert "bad" in result.removed_sample_ids
    assert "bad" not in result.kept_sample_ids