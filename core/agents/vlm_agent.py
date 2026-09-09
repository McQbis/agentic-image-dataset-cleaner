"""
Step 4: Agent VLM evaluates a single sample.

A single sample is intentionally the unit of work. The Celery task layer can
therefore distribute worst-N samples across workers instead of evaluating the
whole set sequentially.

The VLM receives both the image and deterministic scoring signals. Those
signals are hypotheses, not conclusions: the model must visually verify
whether the suspected issue is actually present.

Evaluation failures are represented explicitly through
`evaluation_failed=True`. This distinction is critical because "VLM could
not evaluate the image" is fundamentally different from "VLM evaluated the
image and found no problem".

Daily quota exhaustion is the only provider error that intentionally escapes
this function. It is a global condition for the current run, so continuing
with additional samples would only produce more failed requests.
"""

from __future__ import annotations

import base64
import io

from groq import Groq
from PIL import Image
from pydantic import ValidationError

from core.agents.groq_client import DailyQuotaExceededError, robust_chat_json
from core.schemas import ProblemType, ScoredSample, VisualFinding


SYSTEM_PROMPT = """You are an expert in image data quality assessment.

You receive:
1. an image,
2. deterministic quality signals indicating which metrics are suspicious
   and how severe the suspicion is relative to the dataset.

The deterministic score is evidence, not a conclusion. Inspect the image
yourself and determine whether the suspected issue is actually visible.

Evaluate:
- whether a quality problem is present,
- how severe it is,
- what type of problem it is,
- a concise visual description supporting your conclusion.

If the deterministic signals appear to be a false positive, say so explicitly.

Respond with valid JSON only:
{
  "confirmed": true/false,
  "visual_description": "brief description of the issue or why the image is OK",
  "problem_type": "blur|exposure|noise|duplicate|composition|corrupt|other|none",
  "severity": 1-5
}

Do not include any text outside the JSON object.
"""


def _encode_image(path: str, max_side: int) -> str:
    """
    Encode an image as a bounded JPEG data URL payload.

    Downscaling is applied only when necessary. Keeping the original
    dimensions for smaller images preserves useful visual detail while the
    max-side limit prevents unnecessarily large requests to the vision model.
    """
    image = Image.open(path).convert("RGB")

    width, height = image.size
    largest_side = max(width, height)

    if largest_side > max_side:
        scale = max_side / largest_side
        image = image.resize(
            (
                int(width * scale),
                int(height * scale),
            )
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_suspicion_text(sample: ScoredSample) -> str:
    """Build the deterministic evidence presented to the VLM for verification."""
    return (
        "Deterministic composite score (badness, 0-1): "
        f"{sample.composite_score}\n"
        f"Per-metric suspicion: {sample.per_metric_badness}\n"
        f"Raw metrics: {sample.raw_metrics}"
    )


def _build_corrupt_finding(
    sample_id: str,
    error: Exception,
) -> VisualFinding:
    """
    Represent an image decoding failure as a confirmed corruption finding.

    The VLM never received a usable image in this case, so this is an
    application-level finding rather than a model judgment.
    """
    return VisualFinding(
        sample_id=sample_id,
        confirmed=True,
        visual_description=f"Could not load image: {error}",
        problem_type=ProblemType.CORRUPT,
        severity=5,
    )


def _build_evaluation_failure(
    sample_id: str,
    error: Exception,
) -> VisualFinding:
    """
    Represent an unavailable VLM evaluation without treating it as a clean image.

    `evaluation_failed=True` is intentionally distinct from
    `confirmed=False`. Downstream agents use this flag to prevent an
    unavailable evaluation from becoming an implicit "keep" decision.
    """
    return VisualFinding(
        sample_id=sample_id,
        confirmed=False,
        visual_description=f"VLM did not return a valid evaluation: {error}",
        problem_type=ProblemType.VLM_UNAVAILABLE,
        severity=1,
        evaluation_failed=True,
    )


def evaluate_sample(
    client: Groq,
    model: str,
    sample: ScoredSample,
    max_side: int = 768,
    reasoning_effort: str = "none",
) -> VisualFinding:
    """
    Evaluate one scored sample with the vision model.

    Ordinary image, provider, and validation failures are converted into an
    explicit failed finding so the surrounding Celery batch can continue.
    Daily quota exhaustion is deliberately propagated because it affects the
    entire run and retrying other samples is unlikely to succeed.

    Returns:
        A validated VisualFinding for the sample.

    Raises:
        DailyQuotaExceededError: If the provider's daily token quota is
            exhausted.
    """
    try:
        image_base64 = _encode_image(
            path=sample.path,
            max_side=max_side,
        )
    except Exception as exc:  # noqa: BLE001 - image decoding is an I/O boundary
        # PIL can raise different exceptions depending on the file format,
        # decoder, or underlying I/O failure. At this boundary they all have
        # the same semantic meaning: the sample cannot be evaluated as an image.
        return _build_corrupt_finding(
            sample_id=sample.sample_id,
            error=exc,
        )

    user_content = [
        {
            "type": "text",
            "text": _build_suspicion_text(sample),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        },
    ]

    try:
        data = robust_chat_json(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            reasoning_effort=reasoning_effort,
        )

        # The sample ID comes from the pipeline, never from the model.
        data["sample_id"] = sample.sample_id

        return VisualFinding.model_validate(data)

    except DailyQuotaExceededError:
        # A daily quota is a run-level condition. Let the task/orchestration
        # layer stop the remaining work instead of generating misleading
        # per-sample failures for a condition that affects every request.
        raise

    except (ValueError, ValidationError, RuntimeError) as exc:
        return _build_evaluation_failure(
            sample_id=sample.sample_id,
            error=exc,
        )