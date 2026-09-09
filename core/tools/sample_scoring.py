"""
Deterministic quality scoring stage.

The scorer converts heterogeneous image-quality metrics into a common
"badness" scale, combines them using configurable weights, and produces
a single composite score used to rank samples for downstream review.

Two details are deliberate and essential to the scoring model:

1. MAX-FLOOR
   A weighted average can hide a catastrophic failure in one metric behind
   unusually good values in other, potentially correlated metrics.

   Example: strong blur removes high-frequency detail, which can make a
   noise estimator report an artificially "good" result. The floor ensures
   that an extreme failure on any single metric retains sufficient influence
   over the final score.

2. TIEBREAK
   Binary or low-cardinality metrics can produce large groups of samples
   with identical floor scores. The weighted sum is retained as a secondary
   sort key so ordering within such groups remains deterministic and does not
   depend on filesystem or input ordering.
"""

from __future__ import annotations

import numpy as np

from core.tools.dedup import find_near_duplicates
from core.schemas import CriteriaSet, MetricDirection, SampleMetrics, ScoredSample


# Minimum fraction of the worst individual metric that must survive in the
# composite score. A value of 0.85 means that an extreme failure cannot be
# reduced below 85% of its individual badness by averaging it with better
# metrics.
MAX_FLOOR_FACTOR = 0.85


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """
    Convert raw metric values into deterministic ordinal percentile ranks.

    Ranking instead of normalizing raw values is intentional: metrics such as
    blur variance, brightness and exposure percentages have incompatible
    scales and distributions. Rank normalization puts them onto a common
    [0, 1] domain without assuming a particular statistical distribution.

    The implementation uses ordinal ranks rather than scipy.stats.rankdata,
    keeping this utility dependency-free and deterministic.

    Ties receive distinct ordinal positions based on the original array
    ordering. This is acceptable here because the secondary weighted score
    acts as a deterministic tiebreaker downstream.
    """
    order = values.argsort().argsort()
    n = len(values)

    return order / max(n - 1, 1)


def compute_scores(
    samples: list[SampleMetrics],
    criteria: CriteriaSet,
    max_floor_factor: float = MAX_FLOOR_FACTOR,
) -> list[ScoredSample]:
    """
    Compute deterministic composite badness scores for valid samples.

    Corrupted samples are excluded from scoring because their image metrics
    are not meaningful. They should already have been classified as invalid
    by the ingestion/validation stage.

    The scoring pipeline is:

        raw metrics
            -> percentile ranks
            -> metric-specific badness
            -> weighted aggregate
            -> max-floor protection
            -> deterministic ranking

    Args:
        samples:
            Samples with precomputed deterministic image metrics.
        criteria:
            Metric definitions, directions and relative weights.
        max_floor_factor:
            Strength of the max-floor constraint. Must normally be in [0, 1].
            Higher values make individual catastrophic failures dominate more
            strongly.

    Returns:
        Scored valid samples sorted from worst to best.
    """
    valid = [sample for sample in samples if not sample.is_corrupt]

    # There is nothing meaningful to rank when all inputs are invalid.
    if not valid:
        return []

    if not criteria.metrics:
        # Keep the function well-defined even for an empty criteria set.
        # No metric means no evidence of badness.
        return [
            ScoredSample(
                sample_id=sample.sample_id,
                path=sample.path,
                composite_score=0.0,
                weighted_sum_score=0.0,
                per_metric_badness={},
                raw_metrics={},
            )
            for sample in valid
        ]

    if not 0.0 <= max_floor_factor <= 1.0:
        raise ValueError(
            f"max_floor_factor must be in [0, 1], "
            f"got {max_floor_factor}"
        )

    n = len(valid)

    # Store normalized badness separately for each metric. Keeping this
    # representation makes the individual contributions observable and
    # allows the same values to be reused when constructing the final score.
    badness = {
        metric.name: np.zeros(n, dtype=float)
        for metric in criteria.metrics
    }

    for metric in criteria.metrics:
        raw = np.array(
            [getattr(sample, metric.name) for sample in valid],
            dtype=float,
        )

        rank = _percentile_rank(raw)

        # Convert every metric to a common convention:
        #
        #     0.0 -> least problematic
        #     1.0 -> most problematic
        #
        # The direction determines whether high or low raw values represent
        # worse quality according to the configured criterion.
        if metric.direction == MetricDirection.LOW_IS_BAD:
            badness[metric.name] = 1.0 - rank
        else:
            badness[metric.name] = rank

    # Normalize weights so the absolute magnitude of configured weights does
    # not change the score scale. Only their relative proportions matter.
    total_weight = sum(metric.weight for metric in criteria.metrics)

    if total_weight <= 0:
        raise ValueError(
            f"Criteria weights must sum to a positive value, "
            f"got {total_weight}"
        )

    weighted_sum = np.zeros(n, dtype=float)

    for metric in criteria.metrics:
        normalized_weight = metric.weight / total_weight
        weighted_sum += normalized_weight * badness[metric.name]

    # The weighted average represents the overall quality profile, but by
    # itself it allows one severe defect to be diluted by unrelated metrics.
    #
    # The floor is therefore based on the worst individual metric for each
    # sample. This is intentionally a MAX rather than another weighted term:
    # it establishes a hard lower bound on how much an extreme defect can
    # contribute to the final ranking.
    max_badness_per_sample = np.max(
        np.stack(
            [badness[metric.name] for metric in criteria.metrics],
            axis=0,
        ),
        axis=0,
    )

    composite = np.maximum(
        weighted_sum,
        max_floor_factor * max_badness_per_sample,
    )

    # Deduplication is normally performed as an earlier pipeline stage.
    # This lookup is kept as metadata for observability and backwards
    # compatibility: the scorer records whether a sample belongs to a
    # near-duplicate relationship, but does not use that relationship to
    # modify the score.
    dup_map = find_near_duplicates(valid)

    results: list[ScoredSample] = []

    for index, sample in enumerate(valid):
        per_metric = {
            metric_name: round(float(values[index]), 4)
            for metric_name, values in badness.items()
        }

        # Preserve raw metrics alongside normalized badness. This makes the
        # score explainable: consumers can inspect both the original signal
        # and its contribution to the ranking.
        raw_metrics = {
            "blur_variance": sample.blur_variance,
            "brightness_mean": sample.brightness_mean,
            "overexposed_pct": sample.overexposed_pct,
            "underexposed_pct": sample.underexposed_pct,
            "contrast_std": sample.contrast_std,
            "noise_estimate": sample.noise_estimate,
            "aspect_ratio": sample.aspect_ratio,
            "width": sample.width,
            "height": sample.height,
            "has_near_duplicate": sample.sample_id in dup_map,
        }

        results.append(
            ScoredSample(
                sample_id=sample.sample_id,
                path=sample.path,
                composite_score=round(float(composite[index]), 4),
                weighted_sum_score=round(float(weighted_sum[index]), 4),
                per_metric_badness=per_metric,
                raw_metrics=raw_metrics,
            )
        )

    # Sort worst-first because the next pipeline stage consumes the first N
    # samples. The weighted sum is a deliberate secondary key: it makes
    # ordering deterministic when multiple samples share the same max-floor
    # value.
    results.sort(
        key=lambda result: (
            result.composite_score,
            result.weighted_sum_score,
        ),
        reverse=True,
    )

    return results


def get_worst_n(
    scored: list[ScoredSample],
    n: int,
) -> list[ScoredSample]:
    """
    Return the N highest-priority samples for downstream review.

    `compute_scores` already returns samples in worst-first order, so this is
    intentionally just a bounded slice rather than another sort.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    return scored[:n]