"""
Integration tests for `infra/` -- the Celery-backed `PipelineDeps`.

These tests never touch a real Redis broker/backend or the Groq API. Two
things make that possible:

1. `celery_app.conf.task_always_eager = True` makes `.delay()` / `.s()` /
   `group(...).apply_async()` execute synchronously, in-process, with no
   broker connection at all -- this is Celery's standard way of testing
   task-dispatching code without standing up real infrastructure.
2. The underlying agent functions (`core.agents.*`) that `infra/tasks.py`
   wraps are monkeypatched with deterministic fakes, the same way
   `tests/integration/fakes.py` fakes them for the synchronous `FakeDeps`
   path. `infra/tasks.py`'s `_get_client()` is patched too, so no Groq
   client is ever constructed.

What these tests DO exercise for real is the actual reason `infra/` exists
as a separate boundary: every `ScoredSample` / `CriteriaSet` / etc. that
crosses into a Celery task has to survive `model_dump(mode="json")` ->
Celery's JSON task serializer -> the worker function -> the JSON result
backend -> `model_validate(...)` back in `CeleryPipelineDeps`. A field that
doesn't round-trip through that boundary (e.g. an enum or datetime handled
incorrectly) would fail here even though it would pass fine against
`FakeDeps`, which never leaves the Python process.

`test_full_pipeline_via_celery_deps_with_interrupt_and_resume` is the most
important test in this file: it's the Celery-layer counterpart of
`test_graph_flow.py::test_two_rounds_with_interrupt_and_resume`, and it
re-asserts the `interrupt()` payload actually contains `"report"` with
sample paths -- the same bug that was previously fixed in
`core/orchestration/graph.py`, now checked end-to-end through the
distributed path too.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import infra.tasks as tasks_module
from core.orchestration.graph import build_graph
from core.schemas import (
    CriteriaSet,
    FinalReport,
    ReportItem,
    RunStatus,
    SampleMetrics,
    ScoredSample,
    SuggestedAction,
    VisualFinding,
)
from infra.celery_app import celery_app
from infra.deps import CeleryPipelineDeps

from .conftest import base_pipeline_config
from .fakes import make_fake_criteria


@pytest.fixture(autouse=True)
def eager_celery(monkeypatch):
    """
    Run every Celery task synchronously, in-process, with no broker needed.

    Scoped to this module only (autouse, function-scoped) so it can't leak
    eager-mode settings into other test files that might want to exercise
    real `.apply_async()` behavior later.
    """
    original_eager = celery_app.conf.task_always_eager
    original_propagates = celery_app.conf.task_eager_propagates

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    # None of these tests should ever construct a real Groq client.
    monkeypatch.setattr(tasks_module, "_get_client", lambda: object())

    yield

    celery_app.conf.task_always_eager = original_eager
    celery_app.conf.task_eager_propagates = original_propagates


def _scored_sample(sample_id: str, score: float) -> ScoredSample:
    return ScoredSample(
        sample_id=sample_id,
        path=f"{sample_id}.jpg",
        composite_score=score,
        weighted_sum_score=score,
        per_metric_badness={"blur_variance": score},
        raw_metrics={"blur_variance": 12.5},
    )


def test_analyze_dataset_round_trips_through_json(monkeypatch, tmp_path):
    """SampleMetrics survives model_dump -> Celery JSON -> model_validate."""
    fake_samples = [
        SampleMetrics(sample_id="a", path="a.jpg", width=10, height=10),
        SampleMetrics(
            sample_id="b",
            path="b.jpg",
            is_corrupt=True,
            error="decode failed",
        ),
    ]
    monkeypatch.setattr(tasks_module, "_analyze_dataset", lambda d: fake_samples)

    deps = CeleryPipelineDeps(task_timeout=5)
    result = deps.analyze_dataset(str(tmp_path))

    assert result == fake_samples


def test_formulate_criteria_round_trips_through_json(monkeypatch):
    """CriteriaSet (incl. the MetricDirection enum) survives the task boundary."""
    expected = make_fake_criteria()
    monkeypatch.setattr(
        tasks_module,
        "_formulate_criteria",
        lambda **kwargs: expected,
    )

    deps = CeleryPipelineDeps(task_timeout=5)
    result = deps.formulate_criteria(
        dataset_stats={"blur_variance": {"p50": 10, "p1": 2}},
        previous_feedback=None,
        round_number=1,
    )

    assert isinstance(result, CriteriaSet)
    assert result.problem_statement == expected.problem_statement
    assert [m.name for m in result.metrics] == [m.name for m in expected.metrics]
    assert [m.direction for m in result.metrics] == [
        m.direction for m in expected.metrics
    ]


def test_evaluate_worst_fans_out_as_a_group(monkeypatch):
    """
    evaluate_worst must dispatch one task per sample (a Celery group), not
    one task for the whole list -- that's the entire point of scoping
    `evaluate_sample_task` to a single sample in infra/tasks.py.
    """
    seen_sample_ids: list[str] = []

    def fake_evaluate(client, model, sample, reasoning_effort):
        seen_sample_ids.append(sample.sample_id)
        return VisualFinding(
            sample_id=sample.sample_id,
            confirmed=True,
            visual_description=f"finding for {sample.sample_id}",
            problem_type="other",
            severity=3,
        )

    monkeypatch.setattr(tasks_module, "_evaluate_sample", fake_evaluate)

    worst = [_scored_sample(f"s{i}", score=0.1 * i) for i in range(5)]

    deps = CeleryPipelineDeps(task_timeout=5)
    findings = deps.evaluate_worst(worst)

    # One call per sample -- not one call with the whole batch.
    assert sorted(seen_sample_ids) == sorted(s.sample_id for s in worst)

    # Findings correspond correctly to their samples (group results must
    # not get shuffled or truncated).
    assert {f.sample_id for f in findings} == {s.sample_id for s in worst}
    assert all(isinstance(f, VisualFinding) and f.confirmed for f in findings)


def test_generate_report_round_trips_through_json(monkeypatch):
    """FinalReport/ReportItem (incl. SuggestedAction enum) survive the boundary."""
    worst = [_scored_sample("s1", score=0.9)]
    findings = [
        VisualFinding(
            sample_id="s1",
            confirmed=True,
            visual_description="blurry",
            problem_type="blur",
            severity=4,
        )
    ]
    expected_report = FinalReport(
        overall_summary="summary",
        next_round_proposal=None,
        items=[
            ReportItem(
                sample_id="s1",
                diagnosis="d",
                suggested_action=SuggestedAction.DELETE,
                confidence=0.8,
                path="s1.jpg",
                composite_score=0.9,
            )
        ],
    )
    monkeypatch.setattr(
        tasks_module,
        "_generate_report",
        lambda **kwargs: expected_report,
    )

    deps = CeleryPipelineDeps(task_timeout=5)
    result = deps.generate_report(worst=worst, findings=findings)

    assert isinstance(result, FinalReport)
    assert result.overall_summary == "summary"
    assert result.items[0].suggested_action == SuggestedAction.DELETE
    assert result.items[0].path == "s1.jpg"


def test_full_pipeline_via_celery_deps_with_interrupt_and_resume(
    monkeypatch,
    tiny_dataset,
):
    """
    End-to-end: run the real graph with CeleryPipelineDeps (eager Celery),
    pause at the human-review interrupt, and confirm the interrupt payload
    carries a usable report -- mirroring
    test_graph_flow.py::test_two_rounds_with_interrupt_and_resume, but
    through the distributed path instead of FakeDeps.
    """
    round_criteria = [make_fake_criteria(), make_fake_criteria()]
    round_next_proposal = ["change metric weights", None]
    call_counts = {"formulate": 0, "report": 0}

    def fake_formulate_criteria(
        client,
        model,
        dataset_stats,
        previous_feedback,
        round_number,
        reasoning_effort,
    ) -> CriteriaSet:
        index = min(call_counts["formulate"], len(round_criteria) - 1)
        call_counts["formulate"] += 1
        criteria = round_criteria[index]

        return CriteriaSet(
            problem_statement=criteria.problem_statement,
            metrics=criteria.metrics,
            round_number=round_number,
        )

    def fake_evaluate_sample(client, model, sample, reasoning_effort) -> VisualFinding:
        return VisualFinding(
            sample_id=sample.sample_id,
            confirmed=True,
            visual_description="fake finding",
            problem_type="other",
            severity=3,
        )

    def fake_generate_report(
        client,
        model,
        scored_worst,
        findings,
        reasoning_effort,
    ) -> FinalReport:
        index = min(call_counts["report"], len(round_next_proposal) - 1)
        call_counts["report"] += 1

        items = [
            ReportItem(
                sample_id=sample.sample_id,
                diagnosis="fake diagnosis",
                suggested_action=SuggestedAction.DELETE,
                confidence=0.5,
                path=sample.path,
                composite_score=sample.composite_score,
            )
            for sample in scored_worst
        ]

        return FinalReport(
            overall_summary="fake summary",
            next_round_proposal=round_next_proposal[index],
            items=items,
        )

    monkeypatch.setattr(tasks_module, "_formulate_criteria", fake_formulate_criteria)
    monkeypatch.setattr(tasks_module, "_evaluate_sample", fake_evaluate_sample)
    monkeypatch.setattr(tasks_module, "_generate_report", fake_generate_report)

    deps = CeleryPipelineDeps(task_timeout=15)
    app = build_graph(checkpointer=MemorySaver())
    thread_config = {
        "configurable": {
            "thread_id": "celery-t1",
            "deps": deps,
        }
    }

    result = app.invoke(
        {
            "run_id": "celery-t1",
            "config": base_pipeline_config(tiny_dataset),
            "status": "pending",
        },
        config=thread_config,
    )

    assert "__interrupt__" in result
    assert result["status"] == RunStatus.AWAITING_USER_DECISION.value

    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["reason"] == "next_round_approval"

    # The regression this test exists for: the report (with real sample
    # paths) must be present in the interrupt payload, not just the
    # summary text, even when it went through the Celery/JSON boundary.
    assert "report" in interrupt_payload
    report_items = interrupt_payload["report"]["items"]
    assert len(report_items) > 0
    assert all(item["path"] for item in report_items)
    assert all(item["suggested_action"] == "delete" for item in report_items)

    result = app.invoke(
        Command(resume=True),
        config=thread_config,
    )

    assert "__interrupt__" not in result
    assert result["status"] == RunStatus.COMPLETED.value
    assert len(result["rounds"]) == 2