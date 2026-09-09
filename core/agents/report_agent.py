"""
Step 5: Agent 3 combines deterministic scores and VLM findings into a report.

Agent 3 is responsible for the final per-sample judgment:
- diagnosis,
- suggested action,
- confidence.

The implementation keeps deterministic safeguards outside the LLM:

1. Batch requests keep individual Groq requests bounded and reduce the risk of
   hitting token-per-minute limits.

2. Anchoring is treated as an observable failure mode. The prompt asks the
   model to evaluate samples independently, while the application detects a
   suspiciously uniform action distribution for manual review.

3. A failed VLM evaluation is a fact about the pipeline state, not an opinion.
   Such samples are therefore forced to `keep_review` after the LLM response,
   regardless of the action suggested by Agent 3.

Agent 3 should provide judgment, not override application-level invariants.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterator

from groq import Groq
from pydantic import ValidationError

from core.agents.groq_client import robust_chat_json
from core.schemas import (
    FinalReport,
    ReportItem,
    ScoredSample,
    SuggestedAction,
    VisualFinding,
)


ACTIONS = [action.value for action in SuggestedAction]


BATCH_SYSTEM_PROMPT = """You are a senior image data quality analyst.

You receive a BATCH of samples. Each sample contains:
- a deterministic quality score,
- per-metric badness,
- a visual diagnosis produced by a VLM.

For EACH sample, choose exactly ONE action from:
{actions}

Action definitions:
- "delete" - the sample is unusable or harmful to training quality
- "relabel" - the image itself is acceptable, but its label may be incorrect
- "recrop" - the framing is problematic and can be fixed by cropping
- "enhance" - the problem can reasonably be fixed through image enhancement
- "keep_review" - the case is ambiguous and requires human review
- "keep" - the deterministic/VLM signals are a false alarm

IMPORTANT: Avoid anchoring.

Being included in this batch does NOT automatically mean "delete".
Worst-N contains the samples that are most suspicious according to deterministic
scoring, but deterministic scoring can be wrong. Your role is independent
verification using ALL available signals.

Rules:
- If vlm_evaluation_failed=true, ALWAYS choose "keep_review".
- If vlm_confirmed=false, usually choose "keep", even when the composite score
  is high.
- If vlm_severity <= 2, usually choose "keep" or "enhance", not "delete".
- Do not assign the same action to every sample without first comparing their
  per_metric_badness, vlm_confirmed, vlm_problem_type, and vlm_severity.
- The batch is expected to contain a realistic distribution of actions.

For each sample provide:
- a concise diagnosis,
- exactly one suggested action,
- confidence in [0,1].

Respond with valid JSON only:
{{"items": [
  {{
    "sample_id": "...",
    "diagnosis": "...",
    "suggested_action": "...",
    "confidence": 0.0-1.0
  }}
]}}

Do not include any text outside the JSON object.
""".format(actions=", ".join(ACTIONS))


SUMMARY_SYSTEM_PROMPT = """You are a senior image data quality analyst.

You receive aggregated results from a dataset-quality review:
- number of samples reviewed,
- action distribution,
- average confidence,
- representative diagnoses.

Write:
1. "overall_summary": a concise 2-4 sentence summary of the main findings.
2. "next_round_proposal": a concrete proposal for changing the quality
   criteria in the next round, or null if no change is warranted.

Base the proposal on the observed failure patterns rather than inventing
problems that are not supported by the provided data.

Respond with valid JSON only:
{{"overall_summary": "...", "next_round_proposal": "..." or null}}

Do not include any text outside the JSON object.
"""


def _chunk(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield bounded batches so individual LLM requests remain predictable."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_combined_entry(
    sample: ScoredSample,
    finding: VisualFinding | None,
) -> dict[str, Any]:
    """
    Build the compact contract passed to Agent 3.

    Missing findings are treated as failed evaluation. This is safer than
    presenting missing VLM data as neutral evidence and is consistent with
    the post-processing override applied later.
    """
    return {
        "sample_id": sample.sample_id,
        "composite_score": sample.composite_score,
        "per_metric_badness": sample.per_metric_badness,
        "vlm_confirmed": finding.confirmed if finding else None,
        "vlm_description": (
            finding.visual_description
            if finding
            else "No VLM evaluation available."
        ),
        "vlm_problem_type": finding.problem_type.value if finding else None,
        "vlm_severity": finding.severity if finding else None,
        "vlm_evaluation_failed": (
            finding.evaluation_failed if finding else True
        ),
    }


def _parse_batch_items(data: dict[str, Any]) -> list[ReportItem]:
    """
    Validate the model-generated report items against the canonical schema.

    Schema validation remains local to the boundary where untrusted LLM output
    enters the application.
    """
    raw_items = data.get("items", [])

    if not isinstance(raw_items, list):
        raise ValueError("Agent 3 response contains a non-list 'items' field.")

    return [ReportItem.model_validate(item) for item in raw_items]


def _build_batch_fallback(
    batch: list[dict[str, Any]],
    error: Exception,
) -> list[ReportItem]:
    """
    Convert a failed Agent 3 batch into explicit human-review items.

    A failed report generation must not silently turn into an automated action.
    `keep_review` is the conservative fallback because no reliable judgment
    was produced for the affected samples.
    """
    return [
        ReportItem(
            sample_id=entry["sample_id"],
            diagnosis=f"Agent 3 failed to generate a diagnosis: {error}",
            suggested_action=SuggestedAction.KEEP_REVIEW,
            confidence=0.1,
        )
        for entry in batch
    ]


def _apply_evaluation_failure_override(
    items: list[ReportItem],
    findings_by_id: dict[str, VisualFinding],
) -> int:
    """
    Force failed VLM evaluations to `keep_review`.

    This is intentionally performed after LLM inference. The model is allowed
    to suggest an action, but it cannot override a known pipeline failure.
    """
    forced_count = 0

    for item in items:
        finding = findings_by_id.get(item.sample_id)

        if not finding or not finding.evaluation_failed:
            continue

        if item.suggested_action == SuggestedAction.KEEP_REVIEW:
            continue

        item.suggested_action = SuggestedAction.KEEP_REVIEW
        item.diagnosis = (
            "[VLM evaluation failed] "
            f"{item.diagnosis}"
        )
        forced_count += 1

    return forced_count


def _detect_anchoring(items: list[ReportItem]) -> None:
    """
    Emit a diagnostic warning when Agent 3 assigns one action to the whole set.

    Uniform output is not automatically incorrect, so this is deliberately
    observability rather than an automatic correction.
    """
    if len(items) < 5:
        return

    actions = {item.suggested_action for item in items}

    if len(actions) != 1:
        return

    action = items[0].suggested_action.value

    print(
        "WARNING: Agent 3 assigned action "
        f"'{action}' to all {len(items)} samples. "
        "This may indicate anchoring; manual verification is recommended."
    )


def _build_aggregate_stats(items: list[ReportItem]) -> dict[str, Any]:
    """Build compact statistics for the final report-summary request."""
    action_counts = Counter(
        item.suggested_action.value
        for item in items
    )

    return {
        "n_samples_reviewed": len(items),
        "action_counts": dict(action_counts),
        "avg_confidence": (
            round(
                sum(item.confidence for item in items) / len(items),
                3,
            )
            if items
            else 0
        ),
        "example_diagnoses": [
            f"{item.sample_id}: {item.diagnosis[:120]}"
            for item in items[:5]
        ],
    }


def _generate_summary(
    client: Groq,
    model: str,
    aggregate_stats: dict[str, Any],
    reasoning_effort: str,
    include_reasoning: bool,
) -> tuple[str, str | None]:
    """
    Generate the aggregate report summary.

    Summary generation is independent from per-sample decisions. If this
    secondary request fails, the per-sample report remains usable.
    """
    try:
        data = robust_chat_json(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SUMMARY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        aggregate_stats,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            reasoning_effort=reasoning_effort,
            include_reasoning=include_reasoning,
            max_completion_tokens=768,
        )

        overall_summary = data.get("overall_summary", "")
        next_round_proposal = data.get("next_round_proposal")

        return overall_summary, next_round_proposal

    except (ValueError, RuntimeError) as exc:
        # Keep the detailed per-sample report even when the optional summary
        # request fails. The aggregate statistics provide a deterministic
        # fallback that is still useful to the caller.
        return (
            f"(Agent 3 failed to generate the summary: {exc}) "
            f"Statistics: {aggregate_stats}",
            None,
        )


def generate_report(
    client: Groq,
    model: str,
    scored_worst: list[ScoredSample],
    findings: list[VisualFinding],
    reasoning_effort: str = "low",
    include_reasoning: bool = False,
    report_batch_size: int = 10,
) -> FinalReport:
    """
    Generate the final quality report for the worst-scoring samples.

    The function keeps LLM-facing data preparation, batch inference,
    deterministic post-processing, and aggregate summarization separate.
    This makes the safety-critical overrides explicit and keeps provider
    failures from invalidating samples that can still be reported.

    Args:
        client: Configured Groq client.
        model: Vision/text-capable model used for report generation.
        scored_worst: Samples selected by deterministic scoring.
        findings: VLM findings corresponding to the scored samples.
        reasoning_effort: Provider-specific reasoning budget.
        include_reasoning: Whether provider reasoning should be requested.
        report_batch_size: Maximum number of samples sent in one request.

    Returns:
        A validated FinalReport containing per-sample decisions and an
        aggregate summary.

    Raises:
        ValueError: If report_batch_size is not positive.
    """
    if report_batch_size <= 0:
        raise ValueError("report_batch_size must be positive")

    findings_by_id = {
        finding.sample_id: finding
        for finding in findings
    }
    scored_by_id = {
        sample.sample_id: sample
        for sample in scored_worst
    }

    entries = [
        _build_combined_entry(
            sample=sample,
            finding=findings_by_id.get(sample.sample_id),
        )
        for sample in scored_worst
    ]

    items: list[ReportItem] = []

    for batch in _chunk(entries, report_batch_size):
        try:
            data = robust_chat_json(
                client=client,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": BATCH_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            batch,
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ],
                reasoning_effort=reasoning_effort,
                include_reasoning=include_reasoning,
                max_completion_tokens=min(
                    2048,
                    200 * len(batch) + 512,
                ),
            )
            batch_items = _parse_batch_items(data)

        except (ValueError, ValidationError, RuntimeError) as exc:
            batch_items = _build_batch_fallback(
                batch=batch,
                error=exc,
            )

        items.extend(batch_items)

    # Reattach deterministic fields after LLM processing. The model never
    # controls filesystem paths or the canonical composite score.
    for item in items:
        sample = scored_by_id.get(item.sample_id)

        if sample is None:
            continue

        item.path = sample.path
        item.composite_score = sample.composite_score

    forced_count = _apply_evaluation_failure_override(
        items=items,
        findings_by_id=findings_by_id,
    )

    if forced_count:
        print(
            "WARNING: Forced "
            f"{forced_count} samples to 'keep_review' because "
            "their VLM evaluation failed."
        )

    _detect_anchoring(items)

    aggregate_stats = _build_aggregate_stats(items)

    overall_summary, next_round_proposal = _generate_summary(
        client=client,
        model=model,
        aggregate_stats=aggregate_stats,
        reasoning_effort=reasoning_effort,
        include_reasoning=include_reasoning,
    )

    return FinalReport(
        overall_summary=overall_summary,
        next_round_proposal=next_round_proposal,
        items=items,
    )