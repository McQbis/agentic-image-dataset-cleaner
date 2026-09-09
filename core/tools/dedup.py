"""
Deterministic deduplication stage.

This stage intentionally runs before quality scoring and agent-based review.
Duplicate resolution is a mechanical decision: within each near-duplicate
cluster, retain the strongest representative and discard the remaining
members.

Keeping this concern separate prevents duplicate samples from consuming
the downstream review budget or obscuring genuinely independent quality
issues such as blur, exposure, or noise.
"""

from __future__ import annotations

from core.tools.dataset_analysis import find_near_duplicates
from core.schemas import DedupResult, SampleMetrics


def _build_clusters(dup_map: dict[str, list[str]]) -> list[set[str]]:
    """
    Convert the pairwise duplicate graph into connected components.

    `find_near_duplicates` returns local relationships (A ~ B), while the
    deduplication decision must operate on the complete connected component.
    This distinction matters for transitive chains such as A ~ B ~ C, where
    A and C may not be directly matched but still belong to the same cluster.

    A simple BFS is sufficient here because the graph is unweighted and we
    only need connected components.
    """
    visited: set[str] = set()
    clusters: list[set[str]] = []

    for start in dup_map:
        if start in visited:
            continue

        cluster = {start}
        queue = [start]

        while queue:
            node = queue.pop()

            for neighbor in dup_map.get(node, []):
                if neighbor in cluster:
                    continue

                cluster.add(neighbor)
                queue.append(neighbor)

        visited |= cluster
        clusters.append(cluster)

    return clusters


def deduplicate(
    samples: list[SampleMetrics],
    max_hamming: int = 4,
    representative_key: str = "blur_variance",
) -> DedupResult:
    """
    Remove near-duplicate samples while retaining one representative per
    connected duplicate cluster.

    The representative is selected deterministically using `representative_key`.
    The metric is expected to follow the convention:

        higher value == better representative

    By default, `blur_variance` is used as a proxy for image sharpness.

    Corrupted samples are excluded from duplicate detection because they
    cannot provide a meaningful image representation. They are nevertheless
    propagated to the `removed_sample_ids` output so downstream stages can
    treat them as rejected inputs.

    Args:
        samples: Precomputed metrics for all dataset samples.
        max_hamming: Maximum perceptual-hash Hamming distance considered a
            near-duplicate match.
        representative_key: SampleMetrics attribute used to select the
            strongest cluster representative.

    Returns:
        DedupResult containing retained samples, removed samples, and the
        resolved duplicate clusters.
    """
    # Corrupted files cannot participate in perceptual deduplication.
    # Keep them separate so they can still be reported as rejected inputs.
    valid = [sample for sample in samples if not sample.is_corrupt]
    corrupt = [sample for sample in samples if sample.is_corrupt]

    # The duplicate detector works with sample IDs. Keep O(1) access to the
    # corresponding metrics when resolving clusters below.
    by_id = {sample.sample_id: sample for sample in valid}

    # Build the pairwise near-duplicate graph. Clustering is performed
    # separately because pairwise matches are not yet equivalent to the
    # final deduplication units.
    dup_map = find_near_duplicates(
        valid,
        max_hamming=max_hamming,
    )

    clusters = _build_clusters(dup_map)

    # Track removals by ID rather than by object identity. This keeps the
    # result independent of SampleMetrics implementation details and makes
    # the final output stable and easy to serialize.
    to_remove: set[str] = set()
    cluster_summaries: list[list[str]] = []

    for cluster in clusters:
        members = [
            by_id[sample_id]
            for sample_id in cluster
            if sample_id in by_id
        ]

        # A connected component with a single member is not a duplicate
        # cluster and therefore requires no action.
        if len(members) < 2:
            continue

        # Select exactly one canonical representative. `max()` makes the
        # decision deterministic as long as the underlying metric is
        # deterministic.
        best = max(
            members,
            key=lambda sample: getattr(sample, representative_key),
        )

        # Every other member is redundant and will be excluded from
        # downstream processing.
        for member in members:
            if member.sample_id != best.sample_id:
                to_remove.add(member.sample_id)

        # Preserve the complete cluster for observability/debugging. This is
        # useful when validating deduplication behaviour on real datasets.
        cluster_summaries.append(sorted(cluster))

    # Preserve the original sample order for retained/removed IDs rather
    # than depending on set iteration order.
    kept = [
        sample.sample_id
        for sample in valid
        if sample.sample_id not in to_remove
    ]

    removed = [
        sample.sample_id
        for sample in valid
        if sample.sample_id in to_remove
    ]

    # Corrupted inputs are always rejected, but are intentionally appended
    # after duplicate removals so the two rejection reasons remain
    # distinguishable at the data-model level.
    removed.extend(sample.sample_id for sample in corrupt)

    return DedupResult(
        kept_sample_ids=kept,
        removed_sample_ids=removed,
        clusters=cluster_summaries,
    )