"""
Run the pipeline the same way `run_locally.py` does, but with the worst-N
VLM evaluations (and every other step) dispatched to Celery workers instead
of running in-process.

This requires a running Redis instance and at least one Celery worker
listening on `infra.celery_app` -- see `infra/docker-compose.yml`, or run
them by hand:

    redis-server
    celery -A infra.celery_app worker --loglevel=info

The LangGraph orchestration in `core/orchestration/graph.py` is completely
unaware of the difference: it only ever talks to `PipelineDeps`. Swapping
`LocalPipelineDeps` (see `run_locally.py`) for `CeleryPipelineDeps` here is
the entire change needed to move from "runs on my laptop" to "runs behind a
task queue", which is the point of keeping `infra/` separate from `core/`.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

if not ENV_FILE.exists():
    raise RuntimeError(f"Environment file not found: {ENV_FILE}")

load_dotenv(ENV_FILE)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from core.orchestration.graph import build_graph
from infra.deps import CeleryPipelineDeps


# Local runner configuration. Mirrors run_locally.py so the two scripts are
# easy to compare.
DATASET_DIR = "datasets/dogs_raw"
N_WORST_TO_VLM = 2
MAX_ROUNDS = 3
THREAD_ID = "distributed-pipeline"


def main() -> None:
    deps = CeleryPipelineDeps()

    print("Analyzing dataset via Celery workers... this dispatches tasks and waits.")

    checkpoint_saver = MemorySaver()
    graph = build_graph(checkpointer=checkpoint_saver)

    initial_state = {
        "run_id": THREAD_ID,
        "config": {
            "dataset_dir": DATASET_DIR,
            "n_worst_to_vlm": N_WORST_TO_VLM,
            "max_rounds": MAX_ROUNDS,
        },
    }

    runtime_config = {
        "configurable": {
            "thread_id": THREAD_ID,
            "deps": deps,
        }
    }

    result = graph.invoke(
        initial_state,
        config=runtime_config,
    )

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        print()
        print("=" * 80)
        print("Human review required")
        print("=" * 80)
        print(interrupt_data.get("overall_summary", ""))
        print()
        print("Sample findings:")
        report = interrupt_data.get("report", {})
        for item in report.get("items", []):
            if item.get("suggested_action") == "delete":
                print(f"DELETE: {item.get('path')}")

        answer = input("Run another round? [y/N]: ").strip().lower()
        approved = answer in {"y", "yes"}

        result = graph.invoke(
            Command(resume=approved),
            config=runtime_config,
        )

    print()
    print("=" * 80)
    print("Pipeline completed")
    print("=" * 80)
    print(f"Status: {result.get('status')}")


if __name__ == "__main__":
    main()
