from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

if not ENV_FILE.exists():
    raise RuntimeError(f"Environment file not found: {ENV_FILE}")

load_dotenv(ENV_FILE)

import os

from groq import Groq
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from core.agents.criteria_agent import formulate_criteria
from core.agents.report_agent import generate_report
from core.agents.vlm_agent import evaluate_sample
from core.tools.dataset_analysis import analyze_dataset
from core.orchestration.graph import build_graph
from core.orchestration.state import PipelineDeps
from core.schemas import SampleMetrics, ScoredSample, VisualFinding


# Environment variables.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL")
VLM_MODEL = os.getenv("VLM_MODEL")
TEXT_MODEL_REASONING_EFFORT = os.getenv("TEXT_MODEL_REASONING_EFFORT")
VISION_MODEL_REASONING_EFFORT = os.getenv("VISION_MODEL_REASONING_EFFORT")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

if not TEXT_MODEL:
    raise RuntimeError("TEXT_MODEL is not set")

if not VLM_MODEL:
    raise RuntimeError("VLM_MODEL is not set")


# Local runner configuration.
DATASET_DIR = "datasets/dogs_raw"
N_WORST_TO_VLM = 1
MAX_ROUNDS = 3
THREAD_ID = "local-pipeline"


class LocalPipelineDeps(PipelineDeps):
    def __init__(self, client: Groq) -> None:
        self.client = client

    def analyze_dataset(self, dataset_dir: str) -> list[SampleMetrics]:
        return analyze_dataset(dataset_dir)

    def formulate_criteria(
        self,
        dataset_stats: dict,
        previous_feedback: str | None,
        round_number: int,
    ):
        return formulate_criteria(
            client=self.client,
            model=TEXT_MODEL,
            dataset_stats=dataset_stats,
            previous_feedback=previous_feedback,
            round_number=round_number,
            reasoning_effort=TEXT_MODEL_REASONING_EFFORT,
        )

    def evaluate_worst(
        self,
        worst: list[ScoredSample],
    ) -> list[VisualFinding]:
        return [
            evaluate_sample(
                client=self.client,
                model=VLM_MODEL,
                sample=sample,
                reasoning_effort=VISION_MODEL_REASONING_EFFORT,
            )
            for sample in worst
        ]

    def generate_report(
        self,
        worst: list[ScoredSample],
        findings: list[VisualFinding],
    ):
        return generate_report(
            client=self.client,
            model=TEXT_MODEL,
            scored_worst=worst,
            findings=findings,
            reasoning_effort=TEXT_MODEL_REASONING_EFFORT,
        )


def main() -> None:
    client = Groq(api_key=GROQ_API_KEY)
    deps = LocalPipelineDeps(client)

    print("Analyzing dataset... It may take a few minutes for large datasets.")

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