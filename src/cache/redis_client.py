"""Redis connection for the debate status/history cache (src/cache/debate_cache.py).

Same REDIS_URL as Celery's broker/backend (src/worker/celery_app.py) — one
Redis server, two independent uses: Celery's own broker/backend keys, and
our `debate:*` keys here. decode_responses=True so callers get `str`, not
`bytes`.
"""

import os

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
