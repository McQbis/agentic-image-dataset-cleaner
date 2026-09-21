"""
Celery application wiring for the dataset-cleaner pipeline.

This module is pure infrastructure: it configures Celery to use Redis as
both the message broker and the result backend, and points it at
`infra.tasks` for task discovery.

Nothing about the pipeline's domain logic lives here. If a different queue
(e.g. RabbitMQ) or backend were ever needed, only this file and
`infra/tasks.py` would have to change; `core/` remains untouched.
"""

from __future__ import annotations

import os

from celery import Celery

# A single REDIS_URL is enough for local/dev use, since Redis can serve as
# both broker and result backend on different logical DBs. Splitting
# CELERY_BROKER_URL / CELERY_RESULT_BACKEND is still supported for setups
# that want to point them at different Redis instances (or swap the broker
# for something else later).
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

celery_app = Celery(
    "agentic_image_dataset_cleaner",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["infra.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Makes `flower`/monitoring able to show a "started" state, not just
    # pending/success/failure.
    task_track_started=True,
    # Acknowledge tasks only after they complete. A worker that crashes
    # mid-VLM-call should not silently lose that sample's evaluation.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Individual sample/report calls hitting the Groq API should not hang
    # a worker forever if the provider stalls.
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "120")),
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "180")),
)


if __name__ == "__main__":
    # Allows `python -m infra.celery_app worker --loglevel=info` as an
    # alternative to the `celery -A infra.celery_app worker` CLI form.
    celery_app.start()
