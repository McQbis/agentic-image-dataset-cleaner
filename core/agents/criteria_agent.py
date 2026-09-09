"""
Step 2: Agent 1 defines the quality problem and metric priorities.

The agent is responsible for interpreting the dataset in context: identifying
relevant quality issues, selecting useful metrics, assigning weights, and
describing the resulting quality problem.

Deterministic metric semantics remain owned by the application:
- metric direction is derived from the metric definition,
- only supported metrics are exposed to the agent,
- severity hints identify the relevant tail of each distribution.

This separation is intentional: the LLM provides contextual judgment, while
the application enforces rules that must remain deterministic and independent
of model behavior.
"""

from __future__ import annotations

import json

from groq import Groq
from pydantic import ValidationError

from core.agents.groq_client import robust_chat_json
from core.schemas import CriteriaSet, MetricDirection


# Only metrics with a well-defined monotonic interpretation are exposed to
# the agent. Other deterministic signals remain available to the pipeline,
# but are not part of the Agent 1 scoring contract.
AVAILABLE_METRICS = [
    "blur_variance",
    "overexposed_pct",
    "underexposed_pct",
    "contrast_std",
    "noise_estimate",
]


# Metric direction is a property of the metric definition, not an LLM decision.
#
# brightness_mean and aspect_ratio are intentionally excluded because their
# relationship with quality is not monotonic: both unusually low and unusually
# high values can be problematic.
#
# near_duplicate is also excluded because deduplication is handled by a
# separate deterministic pipeline step.
METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "blur_variance": MetricDirection.LOW_IS_BAD,
    "overexposed_pct": MetricDirection.HIGH_IS_BAD,
    "underexposed_pct": MetricDirection.HIGH_IS_BAD,
    "contrast_std": MetricDirection.LOW_IS_BAD,
    "noise_estimate": MetricDirection.HIGH_IS_BAD,
}


SYSTEM_PROMPT = """You are an expert in image data quality engineering.

You receive:
1. aggregate statistics for the dataset,
2. a "severity_hints" object containing the relevant distribution tail for
   each metric and its relationship to the median.

severity_hints are computed deterministically:
- low_is_bad -> the lower tail (p1) is relevant,
- high_is_bad -> the upper tail (p99) is relevant.

Do not evaluate a metric based only on its median. A dataset can have a healthy
median while still containing a significant number of severe outliers.

Your task:
1. Describe the quality problems suggested by the dataset statistics.
2. Select the metrics that should influence quality ranking in this round.
3. Assign each selected metric a weight in [0,1], reflecting its importance
   to the specific problems present in this dataset.
4. Briefly explain the rationale for each selected metric.

Metric semantics:
- blur_variance: Laplacian variance; higher = sharper image, lower = blur.
- overexposed_pct: percentage of overexposed pixels; higher = worse.
- underexposed_pct: percentage of underexposed pixels; higher = worse.
- contrast_std: standard deviation of brightness; low values may indicate
  flat, low-contrast images.
- noise_estimate: estimated noise level; higher = worse.

Available metrics: {metrics}

Do not determine metric direction. Direction is a fixed property defined by
the application and will be enforced independently of your response.

Respond with valid JSON only:
{{
  "problem_statement": "...",
  "metrics": [
    {{
      "name": "...",
      "weight": 0.0-1.0,
      "rationale": "..."
    }}
  ]
}}

Do not include any text outside the JSON object.
""".format(metrics=", ".join(AVAILABLE_METRICS))


def _compute_severity_hints(dataset_stats: dict) -> dict:
    """
    Prepare distribution-tail signals that help the LLM assess severity.

    The median describes a typical sample, but the pipeline is interested in
    identifying the worst samples. The relevant tail therefore depends on the
    metric direction. The tail-to-median ratio gives the model additional
    context about the magnitude of the deviation without asking it to perform
    the calculation itself.
    """
    hints: dict = {}

    for name, direction in METRIC_DIRECTIONS.items():
        stats = dataset_stats.get(name)

        if not isinstance(stats, dict) or "p50" not in stats:
            continue

        median = stats["p50"]

        if direction == MetricDirection.LOW_IS_BAD:
            tail_label = "p1"
        else:
            tail_label = "p99"

        tail_value = stats.get(tail_label)

        if tail_value is None or median == 0:
            continue

        hints[name] = {
            "direction": direction.value,
            "relevant_percentile": tail_label,
            "relevant_percentile_value": tail_value,
            "median": median,
            "tail_to_median_ratio": round(tail_value / median, 4),
        }

    return hints


def formulate_criteria(
    client: Groq,
    model: str,
    dataset_stats: dict,
    reasoning_effort: str = "low",
    include_reasoning: bool = False,
    previous_feedback: str | None = None,
    round_number: int = 1,
) -> CriteriaSet:
    """
    Ask Agent 1 to select metric priorities and weights for the current round.

    The agent may adapt metric priorities between rounds, but it cannot change
    deterministic metric semantics. Its response is validated against the
    canonical Pydantic schema and metric directions are enforced afterwards.

    Raises:
        ValueError: If the model response violates the schema or references
            an unsupported metric.
    """
    severity_hints = _compute_severity_hints(dataset_stats)

    user_content = (
        "Dataset statistics:\n"
        f"{json.dumps(dataset_stats, ensure_ascii=False, indent=2)}\n\n"
        "Severity hints:\n"
        f"{json.dumps(severity_hints, ensure_ascii=False, indent=2)}"
    )

    if previous_feedback:
        user_content += (
            f"\n\nThis is round {round_number}. "
            "Feedback from the previous round:\n"
            f"{previous_feedback}"
        )

    data = robust_chat_json(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        reasoning_effort=reasoning_effort,
        include_reasoning=include_reasoning,
    )

    try:
        # The round number is controlled by the pipeline, not by the model.
        data["round_number"] = round_number
        criteria = CriteriaSet.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            "Agent 1 returned JSON that does not match the expected schema. "
            f"Data: {data}"
        ) from exc

    _validate_and_apply_metric_directions(criteria)

    return criteria


def _validate_and_apply_metric_directions(criteria: CriteriaSet) -> None:
    """
    Validate the selected metrics and enforce their canonical directions.

    Direction is deliberately treated as derived application data rather than
    model output. This prevents an incorrect LLM response from changing the
    semantics used by the scoring stage.
    """
    for criterion in criteria.metrics:
        if criterion.name not in AVAILABLE_METRICS:
            raise ValueError(
                f"Agent 1 selected an unsupported metric: {criterion.name}"
            )

        criterion.direction = METRIC_DIRECTIONS[criterion.name]