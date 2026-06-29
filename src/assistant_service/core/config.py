from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "assistant-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    log_level: str = "INFO"
    internal_service_token: str = "change-me"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_back_to_llm_queue: str = "back_to_llm"
    rabbitmq_llm_to_back_queue: str = "llm_to_back"
    rabbitmq_prefetch_count: int = 1
    rabbitmq_requeue_delay_seconds: float = 1.0


settings = Settings()
