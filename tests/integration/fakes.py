"""Fake PipelineDeps implementations for integration tests.

These implementations are deterministic and make no network or Groq API
calls. They are used to execute the real LangGraph orchestration and the real
deterministic analysis, deduplication, and scoring logic without depending on
external infrastructure.
"""

from __future__ import annotations

from core.tools.dataset_analysis import analyze_dataset as real_analyze_dataset
from core.schemas import (
    CriteriaSet,
    FinalReport,
    MetricCriterion,
    MetricDirection,
    ReportItem,
    ScoredSample,
    SuggestedAction,
    VisualFinding,
)


class FakeDeps:
    """Deterministic dependency implementation used by integration tests.

    Args:
        round_criteria: Criteria returned by the fake criteria agent for each
            round.
        round_next_proposal: Next-round proposals returned by the fake report
            agent for each round. ``None`` means that the pipeline should stop
            after that round.
    """

    def __init__(
        self,
        round_criteria: list[CriteriaSet],
        round_next_proposal: list[str | None],
    ) -> None:
        self.round_criteria = round_criteria
        self.round_next_proposal = round_next_proposal

        self.formulate_call_count = 0
        self.vlm_call_count = 0
        self.report_call_count = 0

    def analyze_dataset(self, dataset_dir: str):
        """Run the real deterministic dataset analysis."""
        return real_analyze_dataset(dataset_dir)

    def formulate_criteria(
        self,
        dataset_stats,
        previous_feedback,
        round_number: int,
    ) -> CriteriaSet:
        """Return predefined criteria for the current test round."""
        index = min(
            self.formulate_call_count,
            len(self.round_criteria) - 1,
        )
        self.formulate_call_count += 1

        criteria = self.round_criteria[index]

        return CriteriaSet(
            problem_statement=criteria.problem_statement,
            metrics=criteria.metrics,
            round_number=round_number,
        )

    def evaluate_worst(
        self,
        worst: list[ScoredSample],
    ) -> list[VisualFinding]:
        """Return deterministic fake VLM findings for the selected samples."""
        self.vlm_call_count += 1

        return [
            VisualFinding(
                sample_id=sample.sample_id,
                confirmed=True,
                visual_description="fake finding",
                problem_type="other",
                severity=3,
            )
            for sample in worst
        ]

    def generate_report(
        self,
        worst: list[ScoredSample],
        findings: list[VisualFinding],
    ) -> FinalReport:
        """Return a deterministic fake report for the current round."""
        index = min(
            self.report_call_count,
            len(self.round_next_proposal) - 1,
        )
        self.report_call_count += 1

        next_round_proposal = self.round_next_proposal[index]

        items = [
            ReportItem(
                sample_id=sample.sample_id,
                diagnosis="fake diagnosis",
                suggested_action=SuggestedAction.KEEP_REVIEW,
                confidence=0.5,
                path=sample.path,
                composite_score=sample.composite_score,
            )
            for sample in worst
        ]

        return FinalReport(
            overall_summary="fake summary",
            next_round_proposal=next_round_proposal,
            items=items,
        )


def make_fake_criteria() -> CriteriaSet:
    """Create a minimal deterministic criteria set for integration tests."""
    return CriteriaSet(
        problem_statement="fake",
        metrics=[
            MetricCriterion(
                name="blur_variance",
                direction=MetricDirection.LOW_IS_BAD,
                weight=1.0,
                rationale="fake",
            ),
        ],
    )