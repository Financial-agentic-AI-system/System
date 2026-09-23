from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kombu.exceptions import OperationalError as BrokerError
from redis.exceptions import RedisError

from src.api.v1.endpoints import debate, predict

API_PREFIX = "/api/v1"

app = FastAPI(title="Multi-Agent Financial System API", version="0.1.0")

app.include_router(predict.router, prefix=API_PREFIX)
app.include_router(debate.router, prefix=API_PREFIX)


@app.exception_handler(RedisError)
@app.exception_handler(BrokerError)
def _infrastructure_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """Redis (cache) or the Celery broker is down — a 503, not a stack trace."""
    return JSONResponse(
        status_code=503,
        content={"detail": f"Redis/broker unavailable: {exc}"},
    )


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
