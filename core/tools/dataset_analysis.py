"""
Deterministic image-quality analysis.

This module is the non-LLM part of the pipeline. It extracts reproducible
image metrics, perceptual hashes and dataset-level statistics without
maintaining shared mutable state.

The functions are intentionally stateless and idempotent, which makes
`analyze_sample` safe to execute independently in parallel workers
(e.g. Celery). The task layer is responsible only for orchestration;
the actual image analysis remains here.

Responsibilities:
    - dataset file discovery,
    - per-image quality analysis,
    - perceptual near-duplicate detection,
    - aggregate metric statistics.

No function in this module depends on the task/worker layer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import imagehash
import numpy as np
from PIL import Image

from core.schemas import SampleMetrics


SUPPORTED_EXTENSIONS = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
})

SUMMARY_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)

METRICS_TO_SUMMARIZE = (
    "blur_variance",
    "brightness_mean",
    "overexposed_pct",
    "underexposed_pct",
    "contrast_std",
    "noise_estimate",
    "aspect_ratio",
)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def list_dataset_files(dataset_dir: str) -> list[str]:
    """
    Return all supported image files below `dataset_dir`.

    Results are sorted to guarantee stable sample ordering across runs.
    Deterministic ordering is important because `analyze_dataset` derives
    sample IDs from the resulting sequence.
    """
    root = Path(dataset_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    paths = [
        str(path)
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(paths)


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    """
    Load an image and normalize its color representation to RGB.

    Converting once at the I/O boundary gives the rest of the pipeline a
    consistent representation regardless of the source format.
    """
    with Image.open(path) as image:
        return image.convert("RGB")


def _compute_geometry(
    image: np.ndarray,
) -> tuple[int, int, float]:
    """Return width, height and aspect ratio for an OpenCV image."""
    height, width = image.shape[:2]

    aspect_ratio = round(width / height, 4) if height else 0.0

    return width, height, aspect_ratio


def _compute_sharpness(
    gray: np.ndarray,
) -> tuple[float, float]:
    """
    Compute raw and exposure-normalized Laplacian variance.

    The normalized value is calculated after histogram equalization.
    This reduces the dependency between apparent sharpness and global
    exposure, preventing very dark or bright images from being classified
    as blurry solely because of their intensity distribution.

    Returns:
        (raw_variance, normalized_variance)
    """
    raw_variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    equalized = cv2.equalizeHist(gray)

    normalized_variance = float(
        cv2.Laplacian(equalized, cv2.CV_64F).var()
    )

    return raw_variance, normalized_variance


def _compute_exposure_metrics(
    gray: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute brightness and the percentage of extreme pixels.

    The exposure thresholds are deliberately fixed rather than learned from
    the dataset. This keeps the metric interpretable and comparable across
    independent dataset runs.
    """
    total_pixels = gray.size

    if total_pixels == 0:
        return 0.0, 0.0, 0.0

    brightness_mean = float(gray.mean())

    overexposed_pct = float(
        (gray >= 250).sum() / total_pixels * 100
    )

    underexposed_pct = float(
        (gray <= 5).sum() / total_pixels * 100
    )

    return (
        brightness_mean,
        overexposed_pct,
        underexposed_pct,
    )


def _compute_noise_estimate(gray: np.ndarray) -> float:
    """
    Estimate high-frequency residual energy.

    A Gaussian-smoothed image is subtracted from the original grayscale
    image. The standard deviation of the residual provides a simple,
    deterministic proxy for high-frequency noise.

    This is intentionally an estimate rather than a claim of true sensor
    noise: edges, texture and compression artifacts can also contribute to
    the residual.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    return float(residual.std())


def analyze_sample(
    path: str,
    sample_id: str,
) -> SampleMetrics:
    """
    Extract all deterministic metrics for a single image.

    The function is deliberately isolated from dataset-level state. Given
    the same file contents and processing environment, it produces the same
    metric representation and can safely be retried by a worker.

    Any I/O or decoding/processing failure is converted into
    `is_corrupt=True` on the returned sample instead of propagating the
    exception. A single malformed input must not abort analysis of the
    entire dataset.
    """
    metrics = SampleMetrics(
        sample_id=sample_id,
        path=path,
    )

    try:
        pil_image = _load_image(path)

        # pHash provides a compact perceptual representation used later for
        # near-duplicate detection. It is intentionally independent from the
        # quality score.
        metrics.phash = str(
            imagehash.phash(pil_image)
        )

        # OpenCV uses BGR channel ordering, while PIL/NumPy uses RGB.
        # Convert explicitly at the boundary to avoid silent channel swaps.
        image = cv2.cvtColor(
            np.asarray(pil_image),
            cv2.COLOR_RGB2BGR,
        )

        width, height, aspect_ratio = _compute_geometry(image)

        metrics.width = width
        metrics.height = height
        metrics.aspect_ratio = aspect_ratio

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        (
            metrics.blur_variance_raw,
            metrics.blur_variance,
        ) = _compute_sharpness(gray)

        (
            metrics.brightness_mean,
            metrics.overexposed_pct,
            metrics.underexposed_pct,
        ) = _compute_exposure_metrics(gray)

        metrics.contrast_std = float(gray.std())
        metrics.noise_estimate = _compute_noise_estimate(gray)

    except Exception as exc:  # noqa: BLE001
        # This is intentionally a broad exception boundary. Image decoding
        # libraries may raise different exception types for malformed files,
        # truncated images and unsupported encodings. At this layer they all
        # have the same domain meaning: the sample is not analyzable.
        metrics.is_corrupt = True
        metrics.error = str(exc)

    return metrics


# ---------------------------------------------------------------------------
# Dataset-level analysis
# ---------------------------------------------------------------------------

def analyze_dataset(
    dataset_dir: str,
) -> list[SampleMetrics]:
    """
    Analyze a dataset sequentially.

    This implementation is primarily intended for local execution,
    development and tests. Production workloads can distribute
    `analyze_sample` through Celery without changing the underlying analysis
    logic.

    Stable file ordering ensures stable sample IDs between runs.
    """
    paths = list_dataset_files(dataset_dir)

    return [
        analyze_sample(
            path,
            sample_id=f"sample_{index:05d}_{os.path.basename(path)}",
        )
        for index, path in enumerate(paths)
    ]


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

def find_near_duplicates(
    samples: list[SampleMetrics],
    max_hamming: int = 4,
) -> dict[str, list[str]]:
    """
    Build a symmetric near-duplicate adjacency map using pHash distance.

    Two valid samples are considered near-duplicates when their perceptual
    hashes differ by at most `max_hamming` bits.

    The function returns graph edges rather than final duplicate clusters.
    Cluster resolution belongs to the deduplication stage, which can then
    apply its own representative-selection policy.

    Corrupted samples are excluded because they do not have a meaningful
    perceptual representation.
    """
    valid = [
        sample
        for sample in samples
        if not sample.is_corrupt and sample.phash
    ]

    hashes = {
        sample.sample_id: imagehash.hex_to_hash(sample.phash)
        for sample in valid
    }

    duplicate_map: dict[str, list[str]] = {
        sample_id: []
        for sample_id in hashes
    }

    sample_ids = list(hashes)

    # This is intentionally O(n²). Dataset sizes handled by this stage are
    # small enough that exhaustive comparison is simple and predictable.
    # If the dataset grows substantially, replace this with hash bucketing
    # or another approximate-nearest-neighbor strategy.
    for left_index, left_id in enumerate(sample_ids):
        for right_id in sample_ids[left_index + 1:]:
            distance = hashes[left_id] - hashes[right_id]

            if distance <= max_hamming:
                duplicate_map[left_id].append(right_id)
                duplicate_map[right_id].append(left_id)

    # Samples without any edge are not part of a duplicate relationship and
    # do not need to be exposed to downstream clustering.
    return {
        sample_id: neighbors
        for sample_id, neighbors in duplicate_map.items()
        if neighbors
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _percentiles(
    values: Iterable[float],
    percentiles: tuple[int, ...] = SUMMARY_PERCENTILES,
) -> dict[str, float]:
    """
    Compute compact percentile statistics for a metric.

    Aggregating here rather than passing every per-sample value downstream
    keeps the representation suitable for agent prompts while retaining the
    distribution information needed for threshold selection and diagnostics.
    """
    values = list(values)

    if not values:
        return {}

    array = np.asarray(values, dtype=float)

    return {
        f"p{percentile}": round(
            float(np.percentile(array, percentile)),
            3,
        )
        for percentile in percentiles
    }


def summarize_stats(
    samples: list[SampleMetrics],
) -> dict:
    """
    Build dataset-level statistics for downstream criteria selection.

    Only valid samples contribute to metric distributions. Corrupted samples
    are reported separately rather than contaminating the statistical
    summaries with missing or invalid measurements.

    The result intentionally contains aggregate distributions instead of raw
    per-sample metrics: the latter are both unnecessarily large and not
    useful to the criteria-selection agent at this stage.
    """
    valid = [
        sample
        for sample in samples
        if not sample.is_corrupt
    ]

    stats = {
        "n_total": len(samples),
        "n_corrupt": len(samples) - len(valid),
    }

    for metric_name in METRICS_TO_SUMMARIZE:
        values = (
            getattr(sample, metric_name)
            for sample in valid
        )

        stats[metric_name] = _percentiles(values)

    duplicate_map = find_near_duplicates(valid)

    stats["n_samples_with_near_duplicates"] = len(duplicate_map)

    return stats
