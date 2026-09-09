"""
State contracts and dependency ports for the LangGraph orchestration layer.

This module defines two boundaries:

1. `GraphState` is the serialization-friendly state persisted by LangGraph.
2. `PipelineDeps` is the application-level dependency port used by graph nodes.

The orchestration layer must not depend directly on Celery, Groq, FastAPI, or
other infrastructure services. Concrete implementations are injected through
LangGraph's `configurable` runtime configuration.

This separation provides:

- deterministic unit and integration tests without real infrastructure;
- production implementations backed by Celery and the Groq client;
- a small, infrastructure-independent orchestration layer;
- explicit state contracts suitable for checkpointing and process restarts.

Pydantic models remain the canonical domain contracts. LangGraph state stores
their JSON-compatible representations because graph state may be checkpointed,
serialized, and restored across worker processes.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from core.schemas import (
    CriteriaSet,
    FinalReport,
    SampleMetrics,
    ScoredSample,
    VisualFinding,
)


# ---------------------------------------------------------------------------
# Dependency ports
# ---------------------------------------------------------------------------


class PipelineDeps(Protocol):
    """
    Application-level dependency port used by the LangGraph nodes.

    Production implementations may delegate work to Celery and external
    providers. Tests can provide synchronous in-memory implementations.

    The orchestration layer depends only on this protocol and therefore remains
    independent of infrastructure concerns.
    """

    def analyze_dataset(
        self,
        dataset_dir: str,
    ) -> list[SampleMetrics]:
        """
        Analyze all supported images in a dataset directory.
        """
        ...

    def formulate_criteria(
        self,
        dataset_stats: dict[str, Any],
        previous_feedback: str | None,
        round_number: int,
    ) -> CriteriaSet:
        """
        Generate quality criteria for the current evaluation round.
        """
        ...

    def evaluate_worst(
        self,
        worst: list[ScoredSample],
    ) -> list[VisualFinding]:
        """
        Evaluate the selected worst samples with the VLM layer.

        A production implementation may dispatch one Celery task per sample
        and wait for the group result. Tests may use a synchronous fake.

        The orchestration layer does not depend on how parallelism is
        implemented.
        """
        ...

    def generate_report(
        self,
        worst: list[ScoredSample],
        findings: list[VisualFinding],
    ) -> FinalReport:
        """
        Generate the per-round quality report.
        """
        ...


# ---------------------------------------------------------------------------
# Persisted round state
# ---------------------------------------------------------------------------


class RoundStateDict(TypedDict):
    """
    JSON-compatible representation of one completed pipeline round.

    Round history is append-only from the graph's perspective. Current-round
    working data is kept separately in `GraphState`.
    """

    round_number: int
    criteria: dict[str, Any]
    n_scored: int
    n_worst: int
    findings: list[dict[str, Any]]
    report: dict[str, Any]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """
    Serializable state persisted by LangGraph.

    Pydantic domain objects are intentionally stored as dictionaries rather
    than live model instances. This keeps the state compatible with
    checkpointers and process boundaries.

    State is divided into four conceptual groups:

    - run metadata: `run_id`, `config`, `status`;
    - deterministic dataset state: `samples`, `kept_sample_ids`,
      `dedup_result`;
    - round history and current-round working state;
    - lifecycle/output state: `clean_dataset_dir`, `error`,
      `continue_next_round`.

    Fields prefixed with `_current_` represent working data for the current
    round. They are required between graph nodes but are not part of the
    historical `rounds` collection.
    """

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------

    run_id: str
    config: dict[str, Any]
    status: str

    # ------------------------------------------------------------------
    # Deterministic dataset state
    # ------------------------------------------------------------------

    samples: list[dict[str, Any]]
    kept_sample_ids: list[str]
    dedup_result: dict[str, Any]

    # ------------------------------------------------------------------
    # Round state
    # ------------------------------------------------------------------

    round_number: int
    previous_feedback: str | None
    rounds: list[RoundStateDict]

    # ------------------------------------------------------------------
    # Current-round working state
    # ------------------------------------------------------------------

    _current_criteria: dict[str, Any]
    _current_scored: list[dict[str, Any]]
    _current_worst: list[dict[str, Any]]
    _current_findings: list[dict[str, Any]]

    # ------------------------------------------------------------------
    # Lifecycle and output state
    # ------------------------------------------------------------------

    clean_dataset_dir: str | None
    error: str | None
    continue_next_round: bool | None


# ---------------------------------------------------------------------------
# State conversion helpers
# ---------------------------------------------------------------------------


def samples_from_state(
    state: GraphState,
    ids: list[str] | None = None,
) -> list[SampleMetrics]:
    """
    Restore `SampleMetrics` objects from serialized graph state.

    When `ids` is provided, only samples whose IDs occur in the requested
    collection are returned. The original dataset order is preserved.

    Args:
        state: Current LangGraph state.
        ids: Optional sample IDs used to filter the dataset.

    Returns:
        Validated `SampleMetrics` objects in their original state order.

    Raises:
        KeyError: If the required `samples` field is missing.
        pydantic.ValidationError: If persisted sample data is invalid.
    """
    raw_samples = state["samples"]

    if ids is not None:
        requested_ids = set(ids)
        raw_samples = [
            sample
            for sample in raw_samples
            if sample["sample_id"] in requested_ids
        ]

    return [
        SampleMetrics.model_validate(sample)
        for sample in raw_samples
    ]


def criteria_from_state(state: GraphState) -> CriteriaSet:
    """
    Restore the current `CriteriaSet` from graph state.
    """
    return CriteriaSet.model_validate(
        state["_current_criteria"],
    )


def scored_samples_from_state(
    state: GraphState,
) -> list[ScoredSample]:
    """
    Restore all current-round scored samples from graph state.
    """
    return [
        ScoredSample.model_validate(sample)
        for sample in state["_current_scored"]
    ]


def worst_samples_from_state(
    state: GraphState,
) -> list[ScoredSample]:
    """
    Restore the samples selected for VLM evaluation.
    """
    return [
        ScoredSample.model_validate(sample)
        for sample in state["_current_worst"]
    ]


def findings_from_state(
    state: GraphState,
) -> list[VisualFinding]:
    """
    Restore current-round VLM findings from graph state.
    """
    return [
        VisualFinding.model_validate(finding)
        for finding in state["_current_findings"]
    ]


def report_from_round(
    round_state: RoundStateDict,
) -> FinalReport:
    """
    Restore the final report stored in a completed round.
    """
    return FinalReport.model_validate(
        round_state["report"],
    )