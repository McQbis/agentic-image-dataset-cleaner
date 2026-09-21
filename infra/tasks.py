"""
Celery tasks that wrap the pipeline's domain logic from `core/`.

Each task is a thin adapter: it validates/dumps Pydantic models at the
Celery boundary (task arguments and results must be JSON-serializable) and
delegates the actual work to `core.agents` / `core.tools`. No business logic
lives here.

`evaluate_sample_task` is deliberately scoped to a single sample. That is
the unit `core/agents/vlm_agent.py` was already designed around (see its
module docstring), which is exactly what lets `infra/deps.py` fan the
worst-N samples out across a Celery `group` instead of evaluating them one
at a time.

The Groq client is created lazily, once per worker process, instead of once
per task. Worker processes are long-lived, so reusing the client avoids
reconnecting on every single task.
"""

from __future__ import annotations

import os
from typing import Any

from celery.utils.log import get_task_logger
from groq import Groq

from core.agents.criteria_agent import formulate_criteria as _formulate_criteria
from core.agents.report_agent import generate_report as _generate_report
from core.agents.vlm_agent import evaluate_sample as _evaluate_sample
from core.schemas import ScoredSample, VisualFinding
from core.tools.dataset_analysis import analyze_dataset as _analyze_dataset
from infra.celery_app import celery_app


logger = get_task_logger(__name__)

# Read once at import time. Workers are separate processes from the caller
# (API/CLI), so they need their own copy of the model/credentials
# configuration rather than inheriting it from the process that dispatched
# the task.
_TEXT_MODEL = os.getenv("TEXT_MODEL")
_VLM_MODEL = os.getenv("VLM_MODEL")
_TEXT_MODEL_REASONING_EFFORT = os.getenv("TEXT_MODEL_REASONING_EFFORT", "low")
_VISION_MODEL_REASONING_EFFORT = os.getenv("VISION_MODEL_REASONING_EFFORT", "none")

_groq_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily build a single Groq client per worker process."""
    global _groq_client

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set in the Celery worker environment."
            )

        _groq_client = Groq(api_key=api_key)

    return _groq_client


@celery_app.task(
    name="pipeline.analyze_dataset",
    bind=True,
    max_retries=1,
)
def analyze_dataset_task(self, dataset_dir: str) -> list[dict[str, Any]]:
    """Run deterministic per-image analysis on a worker process."""
    samples = _analyze_dataset(dataset_dir)
    return [sample.model_dump(mode="json") for sample in samples]


@celery_app.task(
    name="pipeline.formulate_criteria",
    bind=True,
    max_retries=2,
    retry_backoff=True,
)
def formulate_criteria_task(
    self,
    dataset_stats: dict[str, Any],
    previous_feedback: str | None,
    round_number: int,
) -> dict[str, Any]:
    """Ask Agent 1 (criteria) to run on a worker process."""
    criteria = _formulate_criteria(
        client=_get_client(),
        model=_TEXT_MODEL,
        dataset_stats=dataset_stats,
        previous_feedback=previous_feedback,
        round_number=round_number,
        reasoning_effort=_TEXT_MODEL_REASONING_EFFORT,
    )
    return criteria.model_dump(mode="json")


@celery_app.task(
    name="pipeline.evaluate_sample",
    bind=True,
    max_retries=2,
    retry_backoff=True,
)
def evaluate_sample_task(self, sample: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate a single worst-N sample with the VLM (Agent 2) on a worker.

    Kept to one sample per task on purpose: `infra/deps.py` dispatches the
    full worst-N list as a Celery `group` of these tasks, so a pool of
    workers evaluates them in parallel instead of sequentially.
    """
    scored_sample = ScoredSample.model_validate(sample)

    finding = _evaluate_sample(
        client=_get_client(),
        model=_VLM_MODEL,
        sample=scored_sample,
        reasoning_effort=_VISION_MODEL_REASONING_EFFORT,
    )
    return finding.model_dump(mode="json")


@celery_app.task(
    name="pipeline.generate_report",
    bind=True,
    max_retries=2,
    retry_backoff=True,
)
def generate_report_task(
    self,
    worst: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask Agent 3 (report) to run on a worker process."""
    scored_worst = [ScoredSample.model_validate(item) for item in worst]
    parsed_findings = [VisualFinding.model_validate(item) for item in findings]

    report = _generate_report(
        client=_get_client(),
        model=_TEXT_MODEL,
        scored_worst=scored_worst,
        findings=parsed_findings,
        reasoning_effort=_TEXT_MODEL_REASONING_EFFORT,
    )
    return report.model_dump(mode="json")
