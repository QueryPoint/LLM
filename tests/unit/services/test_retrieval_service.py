import asyncio

from redis.exceptions import RedisError

from assistant_service.agents.intent_agent import IntentDecision, IntentTaskType
from assistant_service.services.elasticsearch_client import SearchResult
from assistant_service.services.retrieval_cache import (
    build_retrieval_cache_key,
    serialize_search_results,
)
from assistant_service.services.retrieval_service import RetrievalService
from assistant_service.services.redis_state import RedisStateStore

USER_ID = "user-123"
OTHER_USER_ID = "user-456"


class FakeRedisClient:
    def __init__(
        self,
        *,
        fail_get: bool = False,
        fail_set: bool = False,
        fail_delete: bool = False,
    ) -> None:
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.fail_delete = fail_delete
        self.values: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[dict[str, object]] = []
        self.delete_calls: list[tuple[str, ...]] = []

    async def get(self, key: str) -> str | None:
        self.get_calls.append(key)
        if self.fail_get:
            raise RedisError("get failed")
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        self.set_calls.append({"key": key, "value": value, "ex": ex, "nx": nx})
        if self.fail_set:
            raise RedisError("set failed")
        self.values[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        self.delete_calls.append(keys)
        if self.fail_delete:
            raise RedisError("delete failed")
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
        return 0

    async def aclose(self) -> None:
        return None


class FakeSearchClient:
    index_name = "documents"

    def __init__(
        self,
        *,
        results: tuple[SearchResult, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._results = results
        self._error = error

    async def search(
        self,
        *,
        query_text: str,
        user_id: str,
        document_id: str | None,
    ) -> tuple[SearchResult, ...]:
        self.calls.append(
            {
                "query_text": query_text,
                "user_id": user_id,
                "document_id": document_id,
            }
        )
        if self._error is not None:
            raise self._error
        return self._results

    async def get_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> object | None:
        return None


def _result() -> SearchResult:
    return SearchResult(
        doc_id="document-1",
        chunk_id="chunk-1",
        file_name="lecture.pdf",
        page_number=3,
        text="Нормализация уменьшает избыточность.",
        score=12.5,
        highlights=("<em>Нормализация</em>",),
    )


def _store(client: FakeRedisClient) -> RedisStateStore:
    return RedisStateStore(
        client=client,  # type: ignore[arg-type]
        answer_cache_ttl_seconds=900,
        intent_cache_ttl_seconds=900,
        task_status_ttl_seconds=3600,
        session_summary_ttl_seconds=21600,
        answer_lock_ttl_seconds=90,
        retrieval_cache_ttl_seconds=900,
    )


def _intent_decision() -> IntentDecision:
    return IntentDecision(
        task_type=IntentTaskType.EXPLAIN_TOPIC,
        requires_retrieval=True,
        requires_full_document=False,
        keywords=["  Нормализация  ", "БАЗА ДАННЫХ", "нормализация"],
        source="rule_based",
    )


def test_retrieval_cache_hit_returns_deserialized_results_without_es() -> None:
    fake_redis = FakeRedisClient()
    cache_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("нормализация", "база данных"),
        user_id=USER_ID,
        document_id=None,
    )
    fake_redis.values[cache_key] = serialize_search_results((_result(),))
    search_client = FakeSearchClient()
    service = RetrievalService(search_client, _store(fake_redis))

    results = asyncio.run(
        service.search(
            intent_decision=_intent_decision(),
            user_id=USER_ID,
            document_id=None,
        )
    )

    assert results == (_result(),)
    assert search_client.calls == []


def test_retrieval_cache_miss_calls_es_and_stores_results() -> None:
    fake_redis = FakeRedisClient()
    search_result = (_result(),)
    search_client = FakeSearchClient(results=search_result)
    service = RetrievalService(search_client, _store(fake_redis))

    results = asyncio.run(
        service.search(
            intent_decision=_intent_decision(),
            user_id=USER_ID,
            document_id="document-id-123",
        )
    )

    assert results == search_result
    assert len(search_client.calls) == 1
    assert search_client.calls[0]["user_id"] == USER_ID
    assert fake_redis.set_calls[0]["ex"] == 900


def test_retrieval_cache_fail_open_on_redis_error() -> None:
    fake_redis = FakeRedisClient(fail_get=True, fail_set=True)
    search_result = (_result(),)
    search_client = FakeSearchClient(results=search_result)
    service = RetrievalService(search_client, _store(fake_redis))

    results = asyncio.run(
        service.search(
            intent_decision=_intent_decision(),
            user_id=USER_ID,
            document_id=None,
        )
    )

    assert results == search_result
    assert len(search_client.calls) == 1


def test_invalid_cached_payload_is_deleted_and_replaced() -> None:
    fake_redis = FakeRedisClient()
    cache_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("нормализация", "база данных"),
        user_id=USER_ID,
        document_id=None,
    )
    fake_redis.values[cache_key] = "{broken-json"
    search_result = (_result(),)
    search_client = FakeSearchClient(results=search_result)
    service = RetrievalService(search_client, _store(fake_redis))

    results = asyncio.run(
        service.search(
            intent_decision=_intent_decision(),
            user_id=USER_ID,
            document_id=None,
        )
    )

    assert results == search_result
    assert fake_redis.delete_calls == [(cache_key,)]
    assert fake_redis.set_calls[-1]["ex"] == 900


def test_empty_results_do_not_write_cache_and_keys_change_with_inputs() -> None:
    fake_redis = FakeRedisClient()
    search_client = FakeSearchClient(results=())
    service = RetrievalService(search_client, _store(fake_redis))

    results = asyncio.run(
        service.search(
            intent_decision=_intent_decision(),
            user_id=USER_ID,
            document_id=None,
        )
    )

    assert results == ()
    assert fake_redis.set_calls == []

    base_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("нормализация", "база данных"),
        user_id=USER_ID,
        document_id=None,
    )
    assert base_key != build_retrieval_cache_key(
        index_name="other-index",
        keywords=("нормализация", "база данных"),
        user_id=USER_ID,
        document_id=None,
    )
    assert base_key != build_retrieval_cache_key(
        index_name="documents",
        keywords=("другие", "слова"),
        user_id=USER_ID,
        document_id=None,
    )
    assert base_key != build_retrieval_cache_key(
        index_name="documents",
        keywords=("нормализация", "база данных"),
        user_id=USER_ID,
        document_id="document-id-123",
    )
    assert base_key != build_retrieval_cache_key(
        index_name="documents",
        keywords=("нормализация", "база данных"),
        user_id=OTHER_USER_ID,
        document_id=None,
    )
