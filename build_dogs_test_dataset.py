"""
Build a synthetic dog image dataset for testing:

1. Clean images form the base dataset.
2. Exact duplicates are copied from the clean base.
   -> Tests deduplication only.
3. Quality degradations are generated from images that are NOT present
   in the clean base.
   -> Tests quality scoring only.
4. Corrupted files are added as invalid inputs.

This separation is intentional: degraded images must never have their
sharp/original counterpart in the dataset, otherwise deduplication could
mask broken quality scoring.

Usage:
    git clone --depth 1 \
        https://github.com/EliSchwartz/imagenet-sample-images.git \
        imagenet_tmp

    python build_dogs_test_dataset.py
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    source_dir: Path = Path("imagenet_tmp")
    output_dir: Path = Path("datasets/dogs_raw")

    clean_count: int = 70
    duplicate_count: int = 10

    degradations_per_type: int = 5
    corrupted_count: int = 4

    seed: int = 7


DOG_KEYWORDS: tuple[str, ...] = (
    "terrier",
    "retriever",
    "hound",
    "spaniel",
    "collie",
    "poodle",
    "_dog",
    "corgi",
    "husky",
    "chihuahua",
    "pointer",
    "setter",
    "sheepdog",
    "mastiff",
    "bulldog",
    "shepherd",
    "schnauzer",
    "pinscher",
    "malamute",
    "dalmatian",
    "pug",
    "beagle",
    "basenji",
    "papillon",
    "whippet",
    "kelpie",
    "komondor",
    "vizsla",
    "keeshond",
    "affenpinscher",
    "griffon",
    "pekinese",
    "maltese",
    "cairn",
    "dingo",
    "newfoundland",
    "leonberg",
    "eskimo_dog",
    "rottweiler",
    "doberman",
    "boxer",
    "samoyed",
    "pomeranian",
    "chow",
    "basset",
)


# ---------------------------------------------------------------------------
# Source discovery / validation
# ---------------------------------------------------------------------------

def is_dog_image(filename: str) -> bool:
    """Return True if filename looks like a dog ImageNet sample."""
    normalized = filename.lower()

    return (
        "spider" not in normalized
        and any(keyword.lower() in normalized for keyword in DOG_KEYWORDS)
    )


def list_dog_images(source_dir: Path) -> list[Path]:
    """Discover candidate dog images in the source directory."""
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir}"
        )

    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and is_dog_image(path.name)
    )


def validate_source(images: list[Path], config: DatasetConfig) -> None:
    """Validate that the source contains enough independent images."""
    required = (
        config.clean_count
        + 3 * config.degradations_per_type
    )

    if len(images) < required:
        raise RuntimeError(
            "Not enough dog images in source dataset: "
            f"found={len(images)}, required={required}. "
            "Decrease clean_count/degradations_per_type "
            "or provide a larger source dataset."
        )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> Image.Image:
    """Load an image and normalize it to RGB."""
    with Image.open(path) as image:
        return image.convert("RGB")


def copy_image(source: Path, destination: Path) -> None:
    """Copy an image while preserving the source file unchanged."""
    shutil.copy2(source, destination)


def save_transformed(
    source: Path,
    destination: Path,
    transform: Callable[[Image.Image], Image.Image],
) -> None:
    """Load, transform and save an image."""
    image = load_rgb(source)

    try:
        transformed = transform(image)
        transformed.save(destination)
    finally:
        image.close()


# ---------------------------------------------------------------------------
# Synthetic degradations
# ---------------------------------------------------------------------------

def blur(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply strong Gaussian blur."""
    radius = rng.uniform(5.0, 8.0)
    return image.filter(ImageFilter.GaussianBlur(radius))


def overexpose(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply strong overexposure."""
    factor = rng.choice((3.2, 3.8, 4.5))
    return ImageEnhance.Brightness(image).enhance(factor)


def underexpose(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply strong underexposure."""
    factor = rng.choice((0.1, 0.15, 0.2))
    return ImageEnhance.Brightness(image).enhance(factor)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_clean_base(
    images: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Copy the clean base images into the output dataset."""
    for source in images:
        copy_image(source, output_dir / source.name)

    return images


def build_exact_duplicates(
    base_images: list[Path],
    output_dir: Path,
    count: int,
    rng: random.Random,
) -> None:
    """
    Add exact byte-level copies of images already present in the base.

    These files intentionally contain no degradation.
    """
    if count > len(base_images):
        raise ValueError(
            f"Cannot create {count} unique duplicate pairs "
            f"from {len(base_images)} base images."
        )

    duplicate_sources = rng.sample(base_images, count)

    for index, source in enumerate(duplicate_sources):
        destination = output_dir / f"dup_{index:02d}_{source.name}"
        copy_image(source, destination)


def build_standalone_degradations(
    source_images: list[Path],
    output_dir: Path,
    count_per_type: int,
    rng: random.Random,
) -> None:
    """
    Create degraded images from sources that are absent from the base set.

    The three degradation categories intentionally use disjoint source
    images, making the quality-scoring test independent from deduplication.
    """
    required = count_per_type * 3

    if len(source_images) < required:
        raise ValueError(
            f"Need {required} standalone source images, "
            f"got {len(source_images)}."
        )

    sources = source_images[:required]

    degradation_specs = (
        ("iso_blurry", blur),
        ("iso_overexp", overexpose),
        ("iso_underexp", underexpose),
    )

    for category_index, (prefix, transform) in enumerate(degradation_specs):
        start = category_index * count_per_type
        end = start + count_per_type

        for index, source in enumerate(sources[start:end]):
            destination = output_dir / f"{prefix}_{index:02d}_{source.name}"

            save_transformed(
                source,
                destination,
                lambda image, transform=transform: transform(image, rng),
            )


def build_corrupted_files(
    output_dir: Path,
    count: int,
    rng: random.Random,
) -> None:
    """Create intentionally invalid JPEG files."""
    for index in range(count):
        destination = output_dir / f"corrupt_{index:02d}.jpg"

        # The content does not need to be a valid JPEG. The point is to test
        # the pipeline's handling of unreadable/corrupted inputs.
        destination.write_bytes(
            rng.randbytes(150)
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetStats:
    clean: int
    duplicates: int
    degradations: int
    corrupted: int

    @property
    def total(self) -> int:
        return (
            self.clean
            + self.duplicates
            + self.degradations
            + self.corrupted
        )

    def print(self, output_dir: Path) -> None:
        print(f"Dataset ready: {self.total} files in {output_dir}/")
        print(f"  - {self.clean} clean images")
        print(
            f"  - {self.duplicates} exact duplicates "
            f"(dedup test)"
        )
        print(
            f"  - {self.degradations} standalone degradations "
            f"(quality scoring test)"
        )
        print(f"  - {self.corrupted} corrupted files")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(config: DatasetConfig) -> DatasetStats:
    """Build the complete synthetic test dataset."""
    rng = random.Random(config.seed)

    dog_images = list_dog_images(config.source_dir)
    validate_source(dog_images, config)

    # Start from a clean output directory to make builds reproducible.
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle a local copy. We never mutate the discovered/sorted list.
    shuffled = dog_images.copy()
    rng.shuffle(shuffled)

    clean_images = shuffled[:config.clean_count]

    # Everything used for degradation comes strictly after the clean slice.
    standalone_images = shuffled[config.clean_count:]

    build_clean_base(
        images=clean_images,
        output_dir=config.output_dir,
    )

    build_exact_duplicates(
        base_images=clean_images,
        output_dir=config.output_dir,
        count=config.duplicate_count,
        rng=rng,
    )

    build_standalone_degradations(
        source_images=standalone_images,
        output_dir=config.output_dir,
        count_per_type=config.degradations_per_type,
        rng=rng,
    )

    build_corrupted_files(
        output_dir=config.output_dir,
        count=config.corrupted_count,
        rng=rng,
    )

    return DatasetStats(
        clean=config.clean_count,
        duplicates=config.duplicate_count,
        degradations=3 * config.degradations_per_type,
        corrupted=config.corrupted_count,
    )


def main() -> None:
    config = DatasetConfig()

    stats = build_dataset(config)
    stats.print(config.output_dir)


if __name__ == "__main__":
    main()