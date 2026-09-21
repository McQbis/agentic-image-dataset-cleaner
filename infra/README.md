# `infra/` — Celery + Redis

This folder is **optional**. `core/` and `run_locally.py` work with nothing
but `pip install -r requirements.txt`. `infra/` exists as a training
exercise toward a future SaaS version of this project: it shows how the same
`core/orchestration` graph can run behind a distributed task queue instead
of a single local process, without `core/` knowing or caring.

## Why it's a separate folder

`core/orchestration/state.py` defines `PipelineDeps`, a `Protocol` that the
LangGraph nodes depend on. Nothing in `core/` imports Celery, Redis, or
anything else infrastructure-specific — it only calls methods on whatever
`PipelineDeps` implementation gets injected at runtime.

- `run_locally.py` injects `LocalPipelineDeps`: everything runs
  synchronously, in one process.
- `run_distributed.py` injects `CeleryPipelineDeps` (from `infra/deps.py`):
  every step is dispatched as a Celery task and the graph blocks until the
  result comes back.

`infra/` is allowed to import from `core/` (to wire it up). `core/` never
imports from `infra/`. That's the whole point: the domain logic (image
analysis, agents, orchestration) stays deployable on its own, and this
folder could be deleted entirely without breaking `run_locally.py`.

## What's here

| File | Purpose |
|---|---|
| `celery_app.py` | The `Celery` app instance, configured for a Redis broker/backend. |
| `tasks.py` | Thin Celery tasks wrapping `core.agents` / `core.tools`. No business logic. |
| `deps.py` | `CeleryPipelineDeps`, a `PipelineDeps` implementation that dispatches to those tasks (`evaluate_worst` uses a Celery `group` to fan the worst-N samples out across workers in parallel). |
| `docker-compose.yml` | Redis + a worker (+ optional Flower UI) for local dev. |
| `Dockerfile.worker` | Image for the Celery worker. |
| `requirements-infra.txt` | `celery[redis]`, `redis`, `flower` — kept separate from the root `requirements.txt`. |

## Running it locally

```bash
# from the repo root
pip install -r requirements.txt -r infra/requirements-infra.txt

# option A: docker compose (starts redis + a worker + flower)
docker compose -f infra/docker-compose.yml --env-file .env up --build

# option B: run redis and the worker yourself
redis-server &
celery -A infra.celery_app worker --loglevel=info

# then, in another terminal, trigger a run
python run_distributed.py
```

Flower (task-queue dashboard) is available at http://localhost:5555 when
using the compose file.

## Notes / current limitations

- `CeleryPipelineDeps` blocks on `.get()` for each stage, so the *caller*
  (e.g. `run_distributed.py`) is still synchronous end-to-end. The win is
  that each stage's actual work happens in a separate, horizontally
  scalable worker pool — most importantly `evaluate_worst`, which fans out
  as a Celery `group` instead of evaluating samples one by one.
- **The dataset directory must be mounted into the worker container.** The
  worker never receives image bytes — `run_distributed.py` sends a *path
  string* (e.g. `"datasets/dogs_raw"`), and the worker opens that same
  relative path itself, inside its own container filesystem. The
  `docker-compose.yml` here mounts `../datasets` to `/app/datasets` for
  exactly this reason. If you point `DATASET_DIR` somewhere else, or run the
  worker outside this compose file, make sure that path is reachable from
  wherever the worker process actually runs — otherwise every
  `analyze_dataset` / VLM task will fail with a file-not-found error, not a
  connection error.
- There's no FastAPI layer yet. `RunStatus` in `core/schemas.py` and the
  `AWAITING_*` states already anticipate one (a run's status could be
  polled instead of blocking on `.get()`), but that's future work, not
  something this folder implements.
- The human-review step (deciding which flagged samples actually get
  deleted before the next round starts) is currently a plain
  `input()` prompt in `run_locally.py` / `run_distributed.py`. Moving that
  to a real UI is a separate piece of work, independent of this
  Celery/Redis layer.