"""
Shared fixtures/helpers for integration tests.

`tiny_dataset` and `base_pipeline_config` were originally defined only in
`test_graph_flow.py`. They moved here so `test_celery_deps.py` (which
exercises the same graph, but through `CeleryPipelineDeps` instead of
`FakeDeps`) can reuse them instead of duplicating a slightly different copy.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path) -> str:
    """Create a small deterministic synthetic dataset."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    rng = np.random.default_rng(42)

    for index in range(6):
        array = (rng.random((80, 80, 3)) * 255).astype("uint8")
        Image.fromarray(array).save(dataset_dir / f"image_{index}.jpg")

    return str(dataset_dir)


def base_pipeline_config(dataset_dir: str) -> dict:
    """Build the default pipeline configuration used by integration tests."""
    return {
        "dataset_dir": dataset_dir,
        "n_worst_to_vlm": 3,
        "max_rounds": 5,
        "report_batch_size": 10,
        "text_model": "x",
        "vision_model": "y",
    }