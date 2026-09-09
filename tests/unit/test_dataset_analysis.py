from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from core.tools.dataset_analysis import (
    analyze_dataset,
    analyze_sample,
    find_near_duplicates,
)


@pytest.fixture
def base_image_path(tmp_path):
    """Create a sharp synthetic image with clear edges."""
    array = np.zeros((120, 120, 3), dtype="uint8")
    array[::10, :, :] = 255
    array[:, ::10, :] = 255

    path = tmp_path / "sharp.jpg"
    Image.fromarray(array).save(path)

    return str(path)


def test_analyze_sample_flags_corrupt_file(tmp_path):
    """Invalid image files must be marked as corrupt."""
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"not a real image")

    metrics = analyze_sample(str(corrupt_path), "sample-1")

    assert metrics.is_corrupt is True
    assert metrics.error is not None


def test_blur_variance_is_lower_for_blurred_image(base_image_path, tmp_path):
    """Blurred images must have lower Laplacian variance than sharp images."""
    sharp = analyze_sample(base_image_path, "sharp")

    blurred_path = tmp_path / "blurred.jpg"
    Image.open(base_image_path).filter(
        ImageFilter.GaussianBlur(6)
    ).save(blurred_path)

    blurred = analyze_sample(str(blurred_path), "blurred")

    assert not sharp.is_corrupt
    assert not blurred.is_corrupt
    assert blurred.blur_variance < sharp.blur_variance


def test_blur_variance_is_not_confounded_by_underexposure(
    base_image_path,
    tmp_path,
):
    """Underexposure must not make a sharp image look strongly blurred.

    The sharp image is darkened after creation, while the comparison image
    is actually blurred. Histogram equalization inside the sharpness metric
    should preserve enough edge information to distinguish the two cases.
    """
    dark_path = tmp_path / "dark.jpg"
    ImageEnhance.Brightness(
        Image.open(base_image_path)
    ).enhance(0.15).save(dark_path)

    dark = analyze_sample(str(dark_path), "dark")

    blurred_path = tmp_path / "blurred.jpg"
    Image.open(base_image_path).filter(
        ImageFilter.GaussianBlur(6)
    ).save(blurred_path)

    blurred = analyze_sample(str(blurred_path), "blurred")

    assert not dark.is_corrupt
    assert not blurred.is_corrupt
    assert dark.blur_variance > blurred.blur_variance * 5


def test_overexposed_percentage_increases_for_bright_image(
    base_image_path,
    tmp_path,
):
    """Strongly brightened images must have more overexposed pixels."""
    bright_path = tmp_path / "bright.jpg"
    ImageEnhance.Brightness(
        Image.open(base_image_path)
    ).enhance(4.0).save(bright_path)

    bright = analyze_sample(str(bright_path), "bright")
    sharp = analyze_sample(base_image_path, "sharp")

    assert bright.overexposed_pct > sharp.overexposed_pct


def test_underexposed_percentage_increases_for_dark_image(
    base_image_path,
    tmp_path,
):
    """Strongly darkened images must have more underexposed pixels."""
    dark_path = tmp_path / "dark.jpg"
    ImageEnhance.Brightness(
        Image.open(base_image_path)
    ).enhance(0.1).save(dark_path)

    dark = analyze_sample(str(dark_path), "dark")
    sharp = analyze_sample(base_image_path, "sharp")

    assert dark.underexposed_pct > sharp.underexposed_pct


def test_find_near_duplicates_detects_exact_copy(
    base_image_path,
    tmp_path,
):
    """An exact copy must be detected as a near duplicate."""
    copy_path = tmp_path / "copy.jpg"
    shutil.copy(base_image_path, copy_path)

    original = analyze_sample(base_image_path, "original")
    copy = analyze_sample(str(copy_path), "copy")

    duplicate_map = find_near_duplicates([original, copy])

    assert "original" in duplicate_map
    assert "copy" in duplicate_map["original"]

    assert "copy" in duplicate_map
    assert "original" in duplicate_map["copy"]


def test_find_near_duplicates_returns_empty_for_distinct_images(tmp_path):
    """Clearly distinct images must not be reported as duplicates."""
    rng = np.random.default_rng(1)
    samples = []

    for index in range(3):
        array = (rng.random((100, 100, 3)) * 255).astype("uint8")
        path = tmp_path / f"image_{index}.jpg"

        Image.fromarray(array).save(path)

        samples.append(
            analyze_sample(str(path), f"sample-{index}")
        )

    duplicate_map = find_near_duplicates(samples)

    assert duplicate_map == {}


def test_analyze_dataset_finds_all_supported_files(tmp_path):
    """Dataset analysis must process supported image formats only."""
    rng = np.random.default_rng(2)

    for index, extension in enumerate(("jpg", "png")):
        array = (rng.random((50, 50, 3)) * 255).astype("uint8")
        path = tmp_path / f"image_{index}.{extension}"

        Image.fromarray(array).save(path)

    (tmp_path / "not_an_image.txt").write_text("ignore me")

    samples = analyze_dataset(str(tmp_path))

    assert len(samples) == 2