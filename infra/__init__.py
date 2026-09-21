"""
Infrastructure layer: Celery + Redis wiring for the pipeline.

Everything in this package exists only to run `core/orchestration` behind a
distributed task queue instead of the synchronous, single-process
`run_locally.py` runner. It is deliberately kept out of `core/`:

- `core/` contains the domain logic (agents, tools, orchestration) and stays
  runnable with nothing more than `pip install -r requirements.txt`.
- `infra/` contains everything needed to scale that same logic out to
  workers (Celery tasks, the Celery app, and a `PipelineDeps` implementation
  that dispatches to those tasks instead of calling Groq directly).

This split means `core/` never imports from `infra/`, but `infra/` is free
to import from `core/` to wire it up. If `infra/` is deleted entirely, the
pipeline still works locally via `run_locally.py`.
"""
