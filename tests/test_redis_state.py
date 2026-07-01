import asyncio
from uuid import UUID

from assistant_service.services.cache_keys import (
    build_session_summary_key,
    build_task_status_key,
)
from assistant_service.services.redis_state import AnswerLock, RedisStateStore


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        if nx and key in self.values:
            return False

        self.values[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.values:
                deleted += 1
                del self.values[key]
        return deleted

    async def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        token: str,
    ) -> int:
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0

    async def aclose(self) -> None:
        return None


def _store(client: FakeRedisClient) -> RedisStateStore:
    return RedisStateStore(
        client=client,  # type: ignore[arg-type]
        answer_cache_ttl_seconds=900,
        intent_cache_ttl_seconds=900,
        task_status_ttl_seconds=3600,
        session_summary_ttl_seconds=21600,
        answer_lock_ttl_seconds=90,
    )


def test_answer_cache_handles_set_get_empty_and_broken_json() -> None:
    client = FakeRedisClient()
    store = _store(client)
    key = "answer:v1:test"

    asyncio.run(store.set_answer(key, "  Cached answer  "))
    assert asyncio.run(store.get_answer(key)) == "Cached answer"

    asyncio.run(store.set_answer("answer:v1:empty", "   "))
    assert "answer:v1:empty" not in client.values

    client.values[key] = "{broken-json"
    assert asyncio.run(store.get_answer(key)) is None


def test_locks_and_delete_cleanup_preserve_answer_cache() -> None:
    client = FakeRedisClient()
    store = _store(client)
    lock_key = "lock:answer:test"

    first_lock = asyncio.run(store.acquire_answer_lock(lock_key=lock_key))
    second_lock = asyncio.run(store.acquire_answer_lock(lock_key=lock_key))

    assert first_lock is not None
    assert second_lock is None
    asyncio.run(
        store.release_answer_lock(AnswerLock(key=lock_key, token="wrong-token"))
    )
    assert lock_key in client.values
    asyncio.run(store.release_answer_lock(first_lock))
    assert lock_key not in client.values

    asyncio.run(store.set_answer("answer:v1:cached", "Answer"))
    asyncio.run(store.set_task_status(user_id=USER_ID, status="processing"))
    asyncio.run(store.set_session_summary(user_id=USER_ID, summary="Summary"))
    asyncio.run(store.clear_user_state(user_id=USER_ID))

    assert "answer:v1:cached" in client.values
    assert build_task_status_key(user_id=USER_ID) not in client.values
    assert build_session_summary_key(user_id=USER_ID) not in client.values
