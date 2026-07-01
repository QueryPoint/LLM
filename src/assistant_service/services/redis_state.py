import json
import logging
import uuid
from dataclasses import dataclass
from json import JSONDecodeError
from uuid import UUID

import redis.asyncio as redis
from redis.exceptions import RedisError

from assistant_service.core.config import Settings
from assistant_service.core.enums import AssistantMode
from assistant_service.services.cache_keys import (
    build_session_summary_key,
    build_task_status_key,
)

logger = logging.getLogger(__name__)

ALLOWED_TASK_STATUSES = frozenset(
    {
        "queued",
        "processing",
        "thinking",
        "searching",
        "found",
        "not_found",
        "insufficient",
        "generating",
        "completed",
        "failed",
    }
)
LOCK_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
""".strip()


@dataclass(frozen=True, slots=True)
class AnswerLock:
    key: str
    token: str


class RedisStateStore:
    def __init__(
        self,
        client: redis.Redis,
        *,
        answer_cache_ttl_seconds: int,
        intent_cache_ttl_seconds: int,
        task_status_ttl_seconds: int,
        session_summary_ttl_seconds: int,
        answer_lock_ttl_seconds: int,
    ) -> None:
        self._validate_positive_ttl(answer_cache_ttl_seconds, "answer_cache_ttl_seconds")
        self._validate_positive_ttl(intent_cache_ttl_seconds, "intent_cache_ttl_seconds")
        self._validate_positive_ttl(task_status_ttl_seconds, "task_status_ttl_seconds")
        self._validate_positive_ttl(
            session_summary_ttl_seconds,
            "session_summary_ttl_seconds",
        )
        self._validate_positive_ttl(answer_lock_ttl_seconds, "answer_lock_ttl_seconds")

        self._client = client
        self._answer_cache_ttl_seconds = answer_cache_ttl_seconds
        self._intent_cache_ttl_seconds = intent_cache_ttl_seconds
        self._task_status_ttl_seconds = task_status_ttl_seconds
        self._session_summary_ttl_seconds = session_summary_ttl_seconds
        self._answer_lock_ttl_seconds = answer_lock_ttl_seconds

    async def get_answer(self, key: str) -> str | None:
        try:
            raw_value = await self._client.get(key)
            if raw_value is None:
                return None

            payload = json.loads(raw_value)
            answer = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(answer, str):
                return None

            stripped_answer = answer.strip()
            return stripped_answer or None
        except (JSONDecodeError, TypeError):
            return None
        except RedisError as exc:
            self._log_redis_error("get_answer", exc)
            return None

    async def set_answer(self, key: str, answer: str) -> None:
        stripped_answer = answer.strip()
        if stripped_answer == "":
            return

        try:
            await self._client.set(
                key,
                json.dumps({"data": stripped_answer}, ensure_ascii=False),
                ex=self._answer_cache_ttl_seconds,
            )
        except RedisError as exc:
            self._log_redis_error("set_answer", exc)

    async def get_intent_mode(self, key: str) -> AssistantMode | None:
        try:
            raw_value = await self._client.get(key)
            if raw_value is None:
                return None

            payload = json.loads(raw_value)
            mode = payload.get("mode") if isinstance(payload, dict) else None
            if not isinstance(mode, str):
                return None
            return AssistantMode(mode)
        except (JSONDecodeError, TypeError, ValueError):
            return None
        except RedisError as exc:
            self._log_redis_error("get_intent_mode", exc)
            return None

    async def set_intent_mode(self, key: str, mode: AssistantMode) -> None:
        try:
            await self._client.set(
                key,
                json.dumps({"mode": mode.value}, ensure_ascii=False),
                ex=self._intent_cache_ttl_seconds,
            )
        except RedisError as exc:
            self._log_redis_error("set_intent_mode", exc)

    async def set_task_status(self, *, user_id: UUID, status: str) -> None:
        if status not in ALLOWED_TASK_STATUSES:
            raise ValueError(f"Unsupported task status: {status}")

        try:
            await self._client.set(
                build_task_status_key(user_id=user_id),
                json.dumps({"status": status}, ensure_ascii=False),
                ex=self._task_status_ttl_seconds,
            )
        except RedisError as exc:
            self._log_redis_error("set_task_status", exc)

    async def get_task_status(self, *, user_id: UUID) -> str | None:
        try:
            raw_value = await self._client.get(build_task_status_key(user_id=user_id))
            if raw_value is None:
                return None

            payload = json.loads(raw_value)
            status = payload.get("status") if isinstance(payload, dict) else None
            if isinstance(status, str) and status in ALLOWED_TASK_STATUSES:
                return status
            return None
        except (JSONDecodeError, TypeError):
            return None
        except RedisError as exc:
            self._log_redis_error("get_task_status", exc)
            return None

    async def clear_task_status(self, *, user_id: UUID) -> None:
        await self._delete_keys(
            "clear_task_status",
            build_task_status_key(user_id=user_id),
        )

    async def get_session_summary(self, *, user_id: UUID) -> str | None:
        try:
            raw_value = await self._client.get(build_session_summary_key(user_id=user_id))
            if raw_value is None:
                return None

            payload = json.loads(raw_value)
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if not isinstance(summary, str):
                return None
            stripped_summary = summary.strip()
            return stripped_summary or None
        except (JSONDecodeError, TypeError):
            return None
        except RedisError as exc:
            self._log_redis_error("get_session_summary", exc)
            return None

    async def set_session_summary(self, *, user_id: UUID, summary: str) -> None:
        stripped_summary = summary.strip()
        if stripped_summary == "":
            return

        try:
            await self._client.set(
                build_session_summary_key(user_id=user_id),
                json.dumps({"summary": stripped_summary}, ensure_ascii=False),
                ex=self._session_summary_ttl_seconds,
            )
        except RedisError as exc:
            self._log_redis_error("set_session_summary", exc)

    async def clear_session_summary(self, *, user_id: UUID) -> None:
        await self._delete_keys(
            "clear_session_summary",
            build_session_summary_key(user_id=user_id),
        )

    async def clear_user_state(self, *, user_id: UUID) -> None:
        await self._delete_keys(
            "clear_user_state",
            build_task_status_key(user_id=user_id),
            build_session_summary_key(user_id=user_id),
        )

    async def acquire_answer_lock(self, *, lock_key: str) -> AnswerLock | None:
        token = uuid.uuid4().hex
        try:
            acquired = await self._client.set(
                lock_key,
                token,
                nx=True,
                ex=self._answer_lock_ttl_seconds,
            )
            if acquired:
                return AnswerLock(key=lock_key, token=token)
            return None
        except RedisError as exc:
            self._log_redis_error("acquire_answer_lock", exc)
            return None

    async def release_answer_lock(self, lock: AnswerLock) -> None:
        try:
            await self._client.eval(LOCK_RELEASE_SCRIPT, 1, lock.key, lock.token)
        except RedisError as exc:
            self._log_redis_error("release_answer_lock", exc)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _delete_keys(self, operation: str, *keys: str) -> None:
        try:
            await self._client.delete(*keys)
        except RedisError as exc:
            self._log_redis_error(operation, exc)

    @staticmethod
    def _validate_positive_ttl(value: int, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _log_redis_error(operation: str, exc: RedisError) -> None:
        logger.warning(
            "Redis operation failed: operation=%s error_type=%s",
            operation,
            type(exc).__name__,
        )


def create_redis_state_store_from_settings(settings: Settings) -> RedisStateStore:
    return RedisStateStore(
        client=redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        ),
        answer_cache_ttl_seconds=settings.redis_answer_cache_ttl_seconds,
        intent_cache_ttl_seconds=settings.redis_intent_cache_ttl_seconds,
        task_status_ttl_seconds=settings.redis_task_status_ttl_seconds,
        session_summary_ttl_seconds=settings.redis_session_summary_ttl_seconds,
        answer_lock_ttl_seconds=settings.redis_answer_lock_ttl_seconds,
    )
