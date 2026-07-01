from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "assistant-service"
    app_env: str = "development"
    log_level: str = "INFO"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_back_to_llm_queue: str = "back_to_llm"
    rabbitmq_llm_to_back_queue: str = "llm_to_back"
    rabbitmq_prefetch_count: int = 1
    rabbitmq_requeue_delay_seconds: float = 1.0
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: int = Field(default=60, gt=0)
    redis_url: str = "redis://localhost:6379/0"
    redis_answer_cache_ttl_seconds: int = Field(default=900, gt=0)
    redis_intent_cache_ttl_seconds: int = Field(default=900, gt=0)
    redis_task_status_ttl_seconds: int = Field(default=3600, gt=0)
    redis_session_summary_ttl_seconds: int = Field(default=21600, gt=0)
    redis_answer_lock_ttl_seconds: int = Field(default=90, gt=0)

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _normalize_optional_secret(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value

    @field_validator("gemini_model", "redis_url", mode="before")
    @classmethod
    def _strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("gemini_model", "redis_url")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        if value == "":
            raise ValueError("required text setting must not be empty")
        return value


settings = Settings()
