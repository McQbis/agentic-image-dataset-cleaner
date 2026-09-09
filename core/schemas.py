"""
Canonical data contracts for the pipeline.

All cross-layer data structures are defined as Pydantic models. This is
intentional: the same schemas are used across deterministic processing,
Celery task boundaries, FastAPI responses and LangGraph state.

Keeping these contracts in a dependency-light module makes `schemas.py` the
bottom layer of the application import graph. It should not depend on
metrics, agents, orchestration or infrastructure modules.

Pydantic is used consistently instead of dataclasses so that serialization
and validation semantics remain identical across API, workers and the
orchestration layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared types / constants
# ---------------------------------------------------------------------------

MetricValue: TypeAlias = float | int | bool

DEFAULT_N_WORST_TO_VLM = 20
DEFAULT_MAX_ROUNDS = 5
DEFAULT_REPORT_BATCH_SIZE = 10

MIN_SEVERITY = 1
MAX_SEVERITY = 5


# ---------------------------------------------------------------------------
# Step 1: deterministic analysis of a single sample
# ---------------------------------------------------------------------------

class SampleMetrics(BaseModel):
    """
    Deterministic metrics extracted from one input image.

    This model represents the output of the image-analysis boundary. A
    corrupt sample is still represented by the same schema; `is_corrupt`
    and `error` carry the failure state instead of raising it through the
    dataset-level pipeline.
    """

    sample_id: str
    path: str

    # A failed image decode/analysis is represented as data so one malformed
    # input cannot abort processing of the entire dataset.
    is_corrupt: bool = False
    error: str | None = None

    # Image geometry.
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0

    # Quality metrics.
    #
    # `blur_variance` is calculated after histogram equalization to reduce
    # exposure-related bias. `blur_variance_raw` is retained for diagnostics
    # and debugging rather than downstream scoring.
    blur_variance: float = 0.0
    blur_variance_raw: float = 0.0

    brightness_mean: float = 0.0
    overexposed_pct: float = 0.0
    underexposed_pct: float = 0.0
    contrast_std: float = 0.0
    noise_estimate: float = 0.0

    # Perceptual hash used exclusively for near-duplicate detection.
    # It is deliberately separate from quality metrics.
    phash: str = ""


# ---------------------------------------------------------------------------
# Step 1.5: deterministic deduplication
# ---------------------------------------------------------------------------

class DedupResult(BaseModel):
    """
    Result of deterministic near-duplicate resolution.

    `clusters` contains the duplicate relationships discovered before a
    representative is selected. `kept_sample_ids` and `removed_sample_ids`
    describe the resulting dataset membership.
    """

    kept_sample_ids: list[str]
    removed_sample_ids: list[str]
    clusters: list[list[str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 2: criteria selected by Agent 1
# ---------------------------------------------------------------------------

class MetricDirection(StrEnum):
    """
    Direction in which a metric represents badness.

    The enum describes the domain semantics of the metric, not the direction
    of the final composite score.
    """

    LOW_IS_BAD = "low_is_bad"
    HIGH_IS_BAD = "high_is_bad"


class MetricCriterion(BaseModel):
    """
    One quality criterion used by the deterministic scorer.

    `direction` is not trusted from the LLM response. The agent proposes
    which metrics matter, while the application owns the canonical metric
    semantics and overwrites the direction programmatically.
    """

    name: str

    # Canonical direction is injected/validated by application code after
    # the LLM response. The default only makes the schema constructible.
    direction: MetricDirection = MetricDirection.LOW_IS_BAD

    # Relative contribution to the weighted score.
    weight: float = Field(ge=0.0, le=1.0)

    # Human-readable explanation of why this criterion matters.
    rationale: str


class CriteriaSet(BaseModel):
    """
    Complete scoring configuration produced for one pipeline round.
    """

    problem_statement: str
    metrics: list[MetricCriterion]
    round_number: int = 1


# ---------------------------------------------------------------------------
# Step 3: deterministic scoring result
# ---------------------------------------------------------------------------

class ScoredSample(BaseModel):
    """
    Quality-ranking result for one valid sample.

    Both the composite score and the weighted sum are retained. The weighted
    sum is useful as a deterministic secondary ordering key when multiple
    samples receive the same max-floor score.
    """

    sample_id: str
    path: str

    # Final score after applying the max-floor constraint.
    composite_score: float

    # Raw weighted aggregate before the max-floor constraint.
    weighted_sum_score: float

    # Normalized [0, 1] badness per configured metric.
    per_metric_badness: dict[str, float]

    # Original deterministic measurements retained for explainability and
    # debugging. This also allows downstream consumers to inspect the signal
    # without recomputing image metrics.
    raw_metrics: dict[str, MetricValue]


# ---------------------------------------------------------------------------
# Step 4: VLM evaluation
# ---------------------------------------------------------------------------

class ProblemType(StrEnum):
    """
    Canonical problem taxonomy returned by visual evaluation.
    """

    BLUR = "blur"
    EXPOSURE = "exposure"
    NOISE = "noise"
    DUPLICATE = "duplicate"
    COMPOSITION = "composition"
    CORRUPT = "corrupt"

    # Explicit failure state. This must never be interpreted as "no problem":
    # the VLM was unavailable and therefore no visual judgment was made.
    VLM_UNAVAILABLE = "vlm_unavailable"

    OTHER = "other"
    NONE = "none"


class VisualFinding(BaseModel):
    """
    VLM's visual assessment of one sample.

    `evaluation_failed` distinguishes a genuine visual conclusion from an
    infrastructure/model failure. In particular, an unavailable VLM must
    never silently become `confirmed=False`.
    """

    sample_id: str
    confirmed: bool

    visual_description: str
    problem_type: ProblemType

    severity: int = Field(
        ge=MIN_SEVERITY,
        le=MAX_SEVERITY,
    )

    # True means that no real visual evaluation was completed.
    evaluation_failed: bool = False


# ---------------------------------------------------------------------------
# Step 5: Agent 3 report
# ---------------------------------------------------------------------------

class SuggestedAction(StrEnum):
    """Action proposed for a sample after diagnosis."""

    DELETE = "delete"
    RELABEL = "relabel"
    RECROP = "recrop"
    ENHANCE = "enhance"
    KEEP_REVIEW = "keep_review"
    KEEP = "keep"


class ReportItem(BaseModel):
    """
    Actionable diagnosis for one reviewed sample.
    """

    sample_id: str
    diagnosis: str
    suggested_action: SuggestedAction

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    # Optional context copied from earlier pipeline stages.
    path: str = ""
    composite_score: float = 0.0


class FinalReport(BaseModel):
    """
    Human-facing report generated for one completed pipeline round.
    """

    overall_summary: str
    next_round_proposal: str | None = None
    items: list[ReportItem]


# ---------------------------------------------------------------------------
# Orchestration: run and round state
# ---------------------------------------------------------------------------

class RunStatus(StrEnum):
    """
    Persisted lifecycle state of a pipeline run.

    These values are part of the API contract, so changing or removing one
    should be treated as a backwards-compatible schema change rather than
    an internal refactor.
    """

    PENDING = "pending"

    # Deterministic analysis + deduplication.
    ANALYZING = "analyzing"

    # Agent 1 is selecting/configuring criteria.
    AWAITING_CRITERIA = "awaiting_criteria"

    # Deterministic scoring using the selected criteria.
    SCORING = "scoring"

    # VLM evaluations are running, potentially as a Celery group.
    AWAITING_VLM = "awaiting_vlm"

    # Agent 3 is producing the final report.
    AWAITING_REPORT = "awaiting_report"

    # The iterative loop is paused until the user decides whether another
    # round should be executed.
    AWAITING_USER_DECISION = "awaiting_user_decision"

    COMPLETED = "completed"
    FAILED = "failed"


class RoundResult(BaseModel):
    """
    Complete result of one criteria -> scoring -> VLM -> reporting round.
    """

    round_number: int
    criteria: CriteriaSet

    n_scored: int
    n_worst: int

    findings: list[VisualFinding]
    report: FinalReport


class PipelineConfig(BaseModel):
    """
    User-defined configuration for one pipeline run.

    This model is intentionally independent of orchestration details. It
    describes what the pipeline should do, not how workers execute it.
    """

    dataset_dir: str

    n_worst_to_vlm: int = DEFAULT_N_WORST_TO_VLM
    max_rounds: int = DEFAULT_MAX_ROUNDS
    report_batch_size: int = DEFAULT_REPORT_BATCH_SIZE

    text_model: str = "openai/gpt-oss-120b"
    vision_model: str = "qwen/qwen3.8-27b"

    @field_validator(
        "n_worst_to_vlm",
        "max_rounds",
        "report_batch_size",
    )
    @classmethod
    def must_be_positive(cls, value: int) -> int:
        """Ensure all execution limits represent a meaningful workload."""
        if value <= 0:
            raise ValueError("value must be positive")

        return value


class PipelineRun(BaseModel):
    """
    Canonical state of one pipeline execution.

    This model deliberately serves two boundaries:

        LangGraph state <-> API representation

    Using the same contract on both sides avoids conversion layers and keeps
    persisted state, worker results and API responses structurally aligned.

    The model is fully serializable and therefore suitable for transport
    across process boundaries.
    """

    run_id: str
    status: RunStatus = RunStatus.PENDING

    config: PipelineConfig

    # UTC-aware timestamps avoid ambiguity when runs are created by workers
    # operating in different local timezones.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Dataset-level progress.
    n_samples_total: int = 0
    n_samples_corrupt: int = 0

    # Result of the pre-scoring deduplication stage.
    dedup_result: DedupResult | None = None

    # Completed/active rounds. Keeping the full round history allows the
    # feedback loop to remain auditable instead of only exposing the latest
    # decision.
    rounds: list[RoundResult] = Field(default_factory=list)

    # Populated after cleanup when a physical clean dataset is materialized.
    clean_dataset_dir: str | None = None

    # Terminal failure information.
    error: str | None = None

    # Control signal for the user-driven feedback loop. `None` means that no
    # decision has been made yet; it is meaningful primarily in
    # AWAITING_USER_DECISION state.
    user_wants_next_round: bool | None = None


# ---------------------------------------------------------------------------
# Storage: clean dataset and raw-vs-clean comparison
# ---------------------------------------------------------------------------

class RemovedSampleRecord(BaseModel):
    """
    Audit record for a sample removed during dataset cleanup.
    """

    sample_id: str
    path: str
    reason: str


class CleanupSummary(BaseModel):
    """
    Summary of the materialized clean dataset.

    Counts are kept separately for corruption and model-selected actions so
    operational failures are not confused with intentional dataset curation.
    """

    n_total: int
    n_kept: int
    n_removed_corrupt: int

    n_removed_by_action: dict[str, int] = Field(
        default_factory=dict
    )

    removed: list[RemovedSampleRecord] = Field(
        default_factory=list
    )


class MetricComparison(BaseModel):
    """
    Comparison of one aggregate metric between raw and clean datasets.

    `improvement_pct` may be unavailable when the raw baseline is zero or
    when percentage improvement has no meaningful interpretation.
    """

    raw: float
    clean: float
    improvement_pct: float | None = None
    direction: MetricDirection


class DatasetComparison(BaseModel):
    """
    Aggregate before/after report for raw versus cleaned datasets.
    """

    n_raw: int
    n_clean: int
    n_removed: int

    pct_removed: float

    n_corrupt_raw: int
    n_corrupt_clean: int

    n_near_duplicates_raw: int
    n_near_duplicates_clean: int

    # Median metric comparison is sufficient for a compact dataset-level
    # quality summary; full distributions are available earlier in the
    # deterministic analysis stage.
    metric_medians: dict[str, MetricComparison] = Field(
        default_factory=dict
    )