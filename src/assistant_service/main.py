from fastapi import FastAPI

from assistant_service.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": settings.app_name, "status": "running"}

    return app


app = create_app()
