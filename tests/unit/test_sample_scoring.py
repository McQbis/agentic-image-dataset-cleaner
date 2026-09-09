from __future__ import annotations

from core.tools.sample_scoring import compute_scores, get_worst_n
from core.schemas import (
    CriteriaSet,
    MetricCriterion,
    MetricDirection,
    SampleMetrics,
)


def _sample(sample_id: str, **kwargs) -> SampleMetrics:
    defaults = {
        "path": f"/fake/{sample_id}.jpg",
        "blur_variance": 1000.0,
        "overexposed_pct": 0.0,
        "underexposed_pct": 0.0,
        "contrast_std": 60.0,
        "noise_estimate": 5.0,
    }
    defaults.update(kwargs)

    return SampleMetrics(sample_id=sample_id, **defaults)


def test_low_is_bad_direction_ranks_lowest_value_as_worst():
    """LOW_IS_BAD must rank the sample with the lowest value as worst."""
    samples = [
        _sample("sharp", blur_variance=3000.0),
        _sample("medium", blur_variance=1000.0),
        _sample("blurry", blur_variance=10.0),
    ]
    criteria = CriteriaSet(
        problem_statement="x",
        metrics=[
            MetricCriterion(
                name="blur_variance",
                direction=MetricDirection.LOW_IS_BAD,
                weight=1.0,
                rationale="x",
            ),
        ],
    )

    scored = compute_scores(samples, criteria)

    assert scored[0].sample_id == "blurry"
    assert scored[-1].sample_id == "sharp"


def test_high_is_bad_direction_ranks_highest_value_as_worst():
    """HIGH_IS_BAD must rank the sample with the highest value as worst."""
    samples = [
        _sample("clean", overexposed_pct=0.0),
        _sample("medium", overexposed_pct=10.0),
        _sample("overexposed", overexposed_pct=90.0),
    ]
    criteria = CriteriaSet(
        problem_statement="x",
        metrics=[
            MetricCriterion(
                name="overexposed_pct",
                direction=MetricDirection.HIGH_IS_BAD,
                weight=1.0,
                rationale="x",
            ),
        ],
    )

    scored = compute_scores(samples, criteria)

    assert scored[0].sample_id == "overexposed"


def test_max_floor_prevents_single_axis_disaster_from_being_cancelled_out():
    """A catastrophic value on one metric must not be cancelled out by good values on other metrics.

    This regression test ensures that a sample with extremely bad blur
    but unusually low noise remains in the worst-N set even when its
    weighted sum would otherwise rank it lower.
    """
    samples = [
        # Very blurry, but also unusually low-noise.
        _sample(
            "extreme_blur_low_noise",
            blur_variance=1.0,
            noise_estimate=0.5,
        ),
        # Moderately bad across all metrics, with no single extreme value.
        _sample(
            "moderately_bad_everywhere",
            blur_variance=800.0,
            noise_estimate=8.0,
            overexposed_pct=5.0,
            underexposed_pct=5.0,
        ),
    ] + [
        _sample(
            f"normal_{index}",
            blur_variance=1000.0 + index * 50,
        )
        for index in range(10)
    ]

    criteria = CriteriaSet(
        problem_statement="x",
        metrics=[
            MetricCriterion(
                name="blur_variance",
                direction=MetricDirection.LOW_IS_BAD,
                weight=0.3,
                rationale="x",
            ),
            MetricCriterion(
                name="noise_estimate",
                direction=MetricDirection.HIGH_IS_BAD,
                weight=0.3,
                rationale="x",
            ),
            MetricCriterion(
                name="overexposed_pct",
                direction=MetricDirection.HIGH_IS_BAD,
                weight=0.2,
                rationale="x",
            ),
            MetricCriterion(
                name="underexposed_pct",
                direction=MetricDirection.HIGH_IS_BAD,
                weight=0.2,
                rationale="x",
            ),
        ],
    )

    scored = compute_scores(samples, criteria)
    worst_samples = get_worst_n(scored, 5)
    worst_ids = {sample.sample_id for sample in worst_samples}

    assert "extreme_blur_low_noise" in worst_ids


def test_tiebreak_uses_weighted_sum_instead_of_input_order():
    """Ties on the max-floor score must be resolved using weighted_sum_score.

    This regression test covers metrics where multiple samples can reach
    the same floor value. A sample with worse secondary metrics must rank
    higher even when it appears later in the input list.
    """
    samples = [
        _sample(
            "first_in_list",
            blur_variance=1.0,
            noise_estimate=1.0,
        ),
        _sample(
            "worse_other_metrics",
            blur_variance=1.0,
            noise_estimate=50.0,
        ),
    ] + [
        _sample(
            f"normal_{index}",
            blur_variance=1000.0 + index * 10,
        )
        for index in range(5)
    ]

    criteria = CriteriaSet(
        problem_statement="x",
        metrics=[
            MetricCriterion(
                name="blur_variance",
                direction=MetricDirection.LOW_IS_BAD,
                weight=0.5,
                rationale="x",
            ),
            MetricCriterion(
                name="noise_estimate",
                direction=MetricDirection.HIGH_IS_BAD,
                weight=0.5,
                rationale="x",
            ),
        ],
    )

    scored = compute_scores(samples, criteria)

    first_index = next(
        index
        for index, sample in enumerate(scored)
        if sample.sample_id == "first_in_list"
    )
    worse_index = next(
        index
        for index, sample in enumerate(scored)
        if sample.sample_id == "worse_other_metrics"
    )

    assert worse_index < first_index


def test_get_worst_n_respects_limit():
    """get_worst_n must return at most the requested number of samples."""
    samples = [
        _sample(f"sample_{index}", blur_variance=100.0 + index)
        for index in range(20)
    ]
    criteria = CriteriaSet(
        problem_statement="x",
        metrics=[
            MetricCriterion(
                name="blur_variance",
                direction=MetricDirection.LOW_IS_BAD,
                weight=1.0,
                rationale="x",
            ),
        ],
    )

    scored = compute_scores(samples, criteria)

    assert len(get_worst_n(scored, 5)) == 5
    assert len(get_worst_n(scored, 100)) == 20


def test_corrupt_samples_are_excluded_from_scoring():
    """Corrupt samples must not be included in quality scoring."""
    samples = [
        _sample("good", blur_variance=1000.0),
        SampleMetrics(
            sample_id="corrupt",
            path="/fake/c.jpg",
            is_corrupt=True,
            error="x",
        ),
    ]
    criteria = CriteriaSet(
        problem_statement="x",
        metrics=[
            MetricCriterion(
                name="blur_variance",
                direction=MetricDirection.LOW_IS_BAD,
                weight=1.0,
                rationale="x",
            ),
        ],
    )

    scored = compute_scores(samples, criteria)

    assert len(scored) == 1
    assert scored[0].sample_id == "good"