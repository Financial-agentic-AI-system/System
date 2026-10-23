import logging
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


app = Celery(
    "mas_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.worker.tasks"],
)

# Our own loggers (src.*): INFO shows one line per LLM call and a summary per
# debate; DEBUG adds the full prompts and responses.
logging.getLogger("src").setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
