import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status

from assistant_service.core.config import settings
from assistant_service.messaging.rabbitmq import RabbitMQWorker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=settings.log_level.upper())

    rabbitmq_worker = RabbitMQWorker(settings=settings)
    app.state.rabbitmq_worker = rabbitmq_worker

    await rabbitmq_worker.start()
    try:
        yield
    finally:
        await rabbitmq_worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/health/ready")
    def ready(response: Response) -> dict[str, str]:
        rabbitmq_worker = getattr(app.state, "rabbitmq_worker", None)
        if rabbitmq_worker is None or not rabbitmq_worker.is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "not_ready",
                "service": settings.app_name,
                "rabbitmq": "disconnected",
            }

        return {
            "status": "ready",
            "service": settings.app_name,
            "rabbitmq": "connected",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": settings.app_name, "status": "running"}

    return app


app = create_app()
