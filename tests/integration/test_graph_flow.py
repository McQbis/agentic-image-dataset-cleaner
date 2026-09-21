"""Integration tests for the complete LangGraph pipeline.

These tests exercise the real graph orchestration, deterministic analysis,
deduplication, and scoring logic while replacing external dependencies such
as Groq and Celery with FakeDeps.

The tests also cover the real interrupt()/Command(resume=...) mechanism,
which is critical for human-in-the-loop execution.
"""

from __future__ import annotations

import numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from PIL import Image

from core.orchestration.graph import build_graph
from core.schemas import RunStatus

from .fakes import FakeDeps, make_fake_criteria


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


def _base_config(dataset_dir: str) -> dict:
    """Build the default pipeline configuration used by integration tests."""
    return {
        "dataset_dir": dataset_dir,
        "n_worst_to_vlm": 3,
        "max_rounds": 5,
        "report_batch_size": 10,
        "text_model": "x",
        "vision_model": "y",
    }


def test_single_round_without_continuation(tiny_dataset):
    """The pipeline must complete without an interrupt when no next round is proposed."""
    deps = FakeDeps(
        round_criteria=[make_fake_criteria()],
        round_next_proposal=[None],
    )
    app = build_graph(checkpointer=MemorySaver())
    thread_config = {
        "configurable": {
            "thread_id": "t1",
            "deps": deps,
        }
    }

    result = app.invoke(
        {
            "run_id": "t1",
            "config": _base_config(tiny_dataset),
            "status": "pending",
        },
        config=thread_config,
    )

    assert "__interrupt__" not in result
    assert result["status"] == RunStatus.COMPLETED.value
    assert len(result["rounds"]) == 1
    assert deps.formulate_call_count == 1


def test_two_rounds_with_interrupt_and_resume(tiny_dataset):
    """The pipeline must pause before starting a proposed second round.

    The second round must only start after an explicit Command(resume=True).
    """
    deps = FakeDeps(
        round_criteria=[
            make_fake_criteria(),
            make_fake_criteria(),
        ],
        round_next_proposal=[
            "change metric weights",
            None,
        ],
    )
    app = build_graph(checkpointer=MemorySaver())
    thread_config = {
        "configurable": {
            "thread_id": "t2",
            "deps": deps,
        }
    }

    result = app.invoke(
        {
            "run_id": "t2",
            "config": _base_config(tiny_dataset),
            "status": "pending",
        },
        config=thread_config,
    )

    assert "__interrupt__" in result
    assert result["status"] == RunStatus.AWAITING_USER_DECISION.value

    # The graph is paused, so the criteria agent must not run again yet.
    assert deps.formulate_call_count == 1

    interrupt_payload = result["__interrupt__"][0].value

    assert interrupt_payload["reason"] == "next_round_approval"
    assert interrupt_payload["round_number"] == 1

    # Regression test: the interrupt payload must carry the full report so
    # that callers (e.g. run_locally.py) can show sample paths to the human
    # reviewer before they decide whether to approve the next round.
    assert "report" in interrupt_payload
    report_items = interrupt_payload["report"]["items"]
    assert len(report_items) > 0
    assert all(item["path"] for item in report_items)

    result = app.invoke(
        Command(resume=True),
        config=thread_config,
    )

    assert "__interrupt__" not in result
    assert result["status"] == RunStatus.COMPLETED.value
    assert len(result["rounds"]) == 2
    assert deps.formulate_call_count == 2


def test_user_declines_next_round(tiny_dataset):
    """The pipeline must stop after one round when the user declines continuation."""
    deps = FakeDeps(
        round_criteria=[make_fake_criteria()],
        round_next_proposal=["change metric weights"],
    )
    app = build_graph(checkpointer=MemorySaver())
    thread_config = {
        "configurable": {
            "thread_id": "t3",
            "deps": deps,
        }
    }

    app.invoke(
        {
            "run_id": "t3",
            "config": _base_config(tiny_dataset),
            "status": "pending",
        },
        config=thread_config,
    )

    result = app.invoke(
        Command(resume=False),
        config=thread_config,
    )

    assert result["status"] == RunStatus.COMPLETED.value
    assert len(result["rounds"]) == 1
    assert deps.formulate_call_count == 1


def test_pipeline_respects_max_rounds(tiny_dataset):
    """The pipeline must stop automatically after reaching max_rounds."""
    available_rounds = 3

    deps = FakeDeps(
        round_criteria=[make_fake_criteria()] * available_rounds,
        round_next_proposal=["always continue"] * available_rounds,
    )
    app = build_graph(checkpointer=MemorySaver())
    thread_config = {
        "configurable": {
            "thread_id": "t4",
            "deps": deps,
        }
    }

    config = _base_config(tiny_dataset)
    config["max_rounds"] = 2

    result = app.invoke(
        {
            "run_id": "t4",
            "config": config,
            "status": "pending",
        },
        config=thread_config,
    )

    # Round 1 completed and the graph should pause before round 2.
    assert "__interrupt__" in result

    result = app.invoke(
        Command(resume=True),
        config=thread_config,
    )

    # Round 2 reached max_rounds, so no third-round interrupt is allowed.
    assert "__interrupt__" not in result
    assert result["status"] == RunStatus.COMPLETED.value
    assert len(result["rounds"]) == 2