"""
LangGraph orchestration for the dataset quality pipeline.

The graph replaces the manual `while True` + `input()` loop from the
notebook-based prototype.

The important architectural difference is the human-in-the-loop boundary:
when another round is proposed, `interrupt()` suspends graph execution and
waits for an external `Command(resume=...)`. The worker therefore never blocks
on terminal input and the pipeline can safely run behind FastAPI and Celery.

Pipeline flow:

    analyze_and_dedup
        -> formulate_criteria
        -> score_and_select_worst
        -> evaluate_vlm
        -> generate_report
        -> review_decision
            -> formulate_criteria  (approved next round)
            -> finalize             (finished)

Responsibilities of this module:

- orchestrate pipeline stages;
- persist intermediate state through LangGraph checkpoints;
- control round transitions;
- expose the human approval boundary through `interrupt()`.

Responsibilities intentionally kept outside this module:

- deterministic image analysis;
- deduplication;
- scoring;
- VLM inference;
- report generation;
- business-specific metric semantics.

All external behavior is injected through `PipelineDeps`, which keeps the
graph independently testable and avoids importing concrete service clients
into the orchestration layer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from core.tools.dedup import deduplicate
from core.tools.dataset_analysis import summarize_stats
from core.tools.sample_scoring import compute_scores, get_worst_n
from core.orchestration.state import GraphState, PipelineDeps, samples_from_state
from core.schemas import (
    CriteriaSet,
    FinalReport,
    RunStatus,
    ScoredSample,
    VisualFinding,
)


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def _get_dependencies(config: RunnableConfig) -> PipelineDeps:
    """
    Resolve runtime dependencies injected into the LangGraph configuration.

    Concrete dependencies are deliberately kept outside graph construction.
    This allows the same graph definition to be used with production
    services, test doubles, or local implementations.

    Raises:
        RuntimeError: If the graph was invoked without PipelineDeps.
    """
    configurable = config.get("configurable", {})
    deps = configurable.get("deps")

    if deps is None:
        raise RuntimeError(
            "PipelineDeps is missing from "
            "config['configurable']['deps']. "
            "Graph nodes require injected dependencies."
        )

    return deps


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _serialize_models(items: list[Any]) -> list[dict[str, Any]]:
    """
    Serialize Pydantic models into JSON-compatible dictionaries.

    LangGraph state is intentionally kept serialization-friendly because it
    may be persisted by a checkpointer and later restored in another worker
    process.
    """
    return [item.model_dump(mode="json") for item in items]


def _load_criteria(state: GraphState) -> CriteriaSet:
    """
    Restore the current criteria from graph state.
    """
    return CriteriaSet.model_validate(state["_current_criteria"])


def _load_scored_samples(state: GraphState) -> list[ScoredSample]:
    """
    Restore the current scored samples from graph state.
    """
    return [
        ScoredSample.model_validate(sample)
        for sample in state["_current_scored"]
    ]


def _load_worst_samples(state: GraphState) -> list[ScoredSample]:
    """
    Restore the samples selected for VLM evaluation from graph state.
    """
    return [
        ScoredSample.model_validate(sample)
        for sample in state["_current_worst"]
    ]


def _load_findings(state: GraphState) -> list[VisualFinding]:
    """
    Restore VLM findings from graph state.
    """
    return [
        VisualFinding.model_validate(finding)
        for finding in state["_current_findings"]
    ]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def node_analyze_and_dedup(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Analyze the dataset and remove deterministic duplicates before scoring.

    This is the only initial analysis stage. Its output becomes the stable
    input set for all subsequent rounds.
    """
    deps = _get_dependencies(config)
    dataset_dir = state["config"]["dataset_dir"]

    samples = deps.analyze_dataset(dataset_dir)
    dedup_result = deduplicate(samples)

    return {
        "status": RunStatus.SCORING.value,
        "samples": _serialize_models(samples),
        "kept_sample_ids": dedup_result.kept_sample_ids,
        "dedup_result": dedup_result.model_dump(mode="json"),
        "round_number": 1,
        "previous_feedback": None,
        "rounds": [],
    }


def node_formulate_criteria(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Generate quality criteria for the current pipeline round.

    The criteria agent receives deterministic dataset statistics and optional
    feedback from the previous round. The resulting criteria are persisted in
    graph state before scoring begins.
    """
    deps = _get_dependencies(config)

    kept_samples = samples_from_state(
        state,
        state["kept_sample_ids"],
    )
    dataset_stats = summarize_stats(kept_samples)

    criteria = deps.formulate_criteria(
        dataset_stats=dataset_stats,
        previous_feedback=state.get("previous_feedback"),
        round_number=state["round_number"],
    )

    return {
        "status": RunStatus.SCORING.value,
        "_current_criteria": criteria.model_dump(mode="json"),
    }


def node_score_and_select_worst(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Score all retained samples and select the worst samples for VLM review.

    Deterministic scoring happens before any visual-model call. Duplicate
    samples have already been removed by the initial analysis stage.
    """
    kept_samples = samples_from_state(
        state,
        state["kept_sample_ids"],
    )
    criteria = _load_criteria(state)

    n_worst = state["config"]["n_worst_to_vlm"]

    scored_samples = compute_scores(
        kept_samples,
        criteria,
    )
    worst_samples = get_worst_n(
        scored_samples,
        n_worst,
    )

    return {
        "status": RunStatus.AWAITING_VLM.value,
        "_current_scored": _serialize_models(scored_samples),
        "_current_worst": _serialize_models(worst_samples),
    }


def node_evaluate_vlm(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Evaluate the worst-scoring samples with the vision model.

    VLM evaluation is delegated to PipelineDeps so this orchestration layer
    remains independent of the provider and execution mechanism.
    """
    deps = _get_dependencies(config)
    worst_samples = _load_worst_samples(state)

    findings = deps.evaluate_worst(worst_samples)

    return {
        "status": RunStatus.AWAITING_REPORT.value,
        "_current_findings": _serialize_models(findings),
    }


def node_generate_report(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Combine deterministic scores and VLM findings into a round report.

    The complete round result is appended to `rounds`. Historical rounds are
    therefore retained when the graph loops back for another evaluation.
    """
    deps = _get_dependencies(config)

    worst_samples = _load_worst_samples(state)
    findings = _load_findings(state)

    report = deps.generate_report(
        worst=worst_samples,
        findings=findings,
    )

    round_record = {
        "round_number": state["round_number"],
        "criteria": state["_current_criteria"],
        "n_scored": len(state["_current_scored"]),
        "n_worst": len(worst_samples),
        "findings": state["_current_findings"],
        "report": report.model_dump(mode="json"),
    }

    rounds = [
        *state.get("rounds", []),
        round_record,
    ]

    return {
        "status": RunStatus.AWAITING_USER_DECISION.value,
        "rounds": rounds,
    }


def node_review_decision(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Handle the human approval boundary between pipeline rounds.

    If Agent 3 does not propose another round, execution continues directly
    to finalization.

    If another round is proposed and the configured round limit has not been
    reached, `interrupt()` suspends graph execution. The graph resumes only
    after an external caller supplies `Command(resume=...)`.

    The resume value is treated as an approval decision, not as arbitrary
    mutable graph state.
    """
    last_round = state["rounds"][-1]
    report = FinalReport.model_validate(last_round["report"])

    if not report.next_round_proposal:
        return {
            "continue_next_round": False,
        }

    if state["round_number"] >= state["config"]["max_rounds"]:
        return {
            "continue_next_round": False,
        }

    decision = interrupt(
        {
            "reason": "next_round_approval",
            "round_number": state["round_number"],
            "next_round_proposal": report.next_round_proposal,
            "overall_summary": report.overall_summary,
            "report": last_round["report"],
        }
    )

    approved = bool(decision)

    return {
        "continue_next_round": approved,
        "previous_feedback": (
            report.next_round_proposal
            if approved
            else None
        ),
        "round_number": (
            state["round_number"] + 1
            if approved
            else state["round_number"]
        ),
    }


def node_finalize(
    state: GraphState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Mark the pipeline run as successfully completed.
    """
    return {
        "status": RunStatus.COMPLETED.value,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_review(state: GraphState) -> str:
    """
    Select the next graph node after the human review boundary.
    """
    if state.get("continue_next_round"):
        return "formulate_criteria"

    return "finalize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """
    Build and compile the dataset quality pipeline graph.

    The graph definition is intentionally free of concrete infrastructure
    dependencies. Runtime services are injected through `RunnableConfig`.

    Args:
        checkpointer: Optional LangGraph checkpointer used to persist graph
            state across `interrupt()` and subsequent resume operations.

    Returns:
        A compiled LangGraph application.
    """
    graph = StateGraph(GraphState)

    graph.add_node(
        "analyze_and_dedup",
        node_analyze_and_dedup,
    )
    graph.add_node(
        "formulate_criteria",
        node_formulate_criteria,
    )
    graph.add_node(
        "score_and_select_worst",
        node_score_and_select_worst,
    )
    graph.add_node(
        "evaluate_vlm",
        node_evaluate_vlm,
    )
    graph.add_node(
        "generate_report",
        node_generate_report,
    )
    graph.add_node(
        "review_decision",
        node_review_decision,
    )
    graph.add_node(
        "finalize",
        node_finalize,
    )

    graph.set_entry_point("analyze_and_dedup")

    graph.add_edge(
        "analyze_and_dedup",
        "formulate_criteria",
    )
    graph.add_edge(
        "formulate_criteria",
        "score_and_select_worst",
    )
    graph.add_edge(
        "score_and_select_worst",
        "evaluate_vlm",
    )
    graph.add_edge(
        "evaluate_vlm",
        "generate_report",
    )
    graph.add_edge(
        "generate_report",
        "review_decision",
    )

    graph.add_conditional_edges(
        "review_decision",
        _route_after_review,
        {
            "formulate_criteria": "formulate_criteria",
            "finalize": "finalize",
        },
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer,
    )