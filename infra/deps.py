"""
Production `PipelineDeps` implementation, backed by Celery + Redis.

`core/orchestration/state.py` defines `PipelineDeps` as the port the graph
nodes depend on. `run_locally.py` provides a synchronous, in-process
implementation for local/portfolio use. This module provides the
distributed counterpart: every method dispatches work to a Celery task
(see `infra/tasks.py`) and blocks on `.get()` until the result comes back.

Blocking on `.get()` keeps the LangGraph node functions themselves
synchronous (LangGraph calls them directly, no async plumbing needed) while
the actual work -- Groq API calls, image decoding -- happens in separate
worker processes that can be scaled independently of the process running
the graph.

`evaluate_worst` is the one place this pays off the most: since each
sample's VLM evaluation is independent, the worst-N samples are dispatched
as a single Celery `group` so a pool of workers evaluates them in parallel
instead of one-by-one, exactly as anticipated by the docstrings in
`core/agents/vlm_agent.py` and `core/orchestration/state.py`.
"""

from __future__ import annotations

from typing import Any

from celery import group

from core.orchestration.state import PipelineDeps
from core.schemas import CriteriaSet, FinalReport, SampleMetrics, ScoredSample, VisualFinding
from infra.tasks import (
    analyze_dataset_task,
    evaluate_sample_task,
    formulate_criteria_task,
    generate_report_task,
)


class CeleryPipelineDeps(PipelineDeps):
    """
    Dispatches each `PipelineDeps` method to a Celery task and waits for it.

    Args:
        task_timeout: Seconds to wait for a dispatched task (or task group)
            to complete before raising `celery.exceptions.TimeoutError`.
    """

    def __init__(self, task_timeout: float = 300.0) -> None:
        self.task_timeout = task_timeout

    def analyze_dataset(self, dataset_dir: str) -> list[SampleMetrics]:
        raw_samples = analyze_dataset_task.delay(dataset_dir).get(
            timeout=self.task_timeout,
        )
        return [SampleMetrics.model_validate(item) for item in raw_samples]

    def formulate_criteria(
        self,
        dataset_stats: dict[str, Any],
        previous_feedback: str | None,
        round_number: int,
    ) -> CriteriaSet:
        raw_criteria = formulate_criteria_task.delay(
            dataset_stats,
            previous_feedback,
            round_number,
        ).get(timeout=self.task_timeout)

        return CriteriaSet.model_validate(raw_criteria)

    def evaluate_worst(
        self,
        worst: list[ScoredSample],
    ) -> list[VisualFinding]:
        # A group, not a chain: every sample is evaluated independently, so
        # workers can pick them up in parallel rather than one after another.
        job = group(
            evaluate_sample_task.s(sample.model_dump(mode="json"))
            for sample in worst
        )

        raw_findings = job.apply_async().get(timeout=self.task_timeout)

        return [VisualFinding.model_validate(item) for item in raw_findings]

    def generate_report(
        self,
        worst: list[ScoredSample],
        findings: list[VisualFinding],
    ) -> FinalReport:
        raw_report = generate_report_task.delay(
            [sample.model_dump(mode="json") for sample in worst],
            [finding.model_dump(mode="json") for finding in findings],
        ).get(timeout=self.task_timeout)

        return FinalReport.model_validate(raw_report)
