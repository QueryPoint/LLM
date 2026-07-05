import asyncio
import os
from uuid import uuid4

import pytest
import redis.asyncio as redis
from redis.exceptions import RedisError

from assistant_service.services.retrieval_cache import (
    deserialize_search_results,
    serialize_search_results,
)
from tests.fixtures.builders import build_search_result


@pytest.mark.integration
@pytest.mark.requires_docker
def test_redis_retrieval_cache_roundtrip_uses_test_only_key_prefix() -> None:
    if os.getenv("RUN_INTEGRATION_TESTS", "").strip().lower() != "true":
        pytest.skip("integration tests are disabled")

    settings_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = redis.from_url(settings_redis_url, encoding="utf-8", decode_responses=True)
    cache_key = f"integration:retrieval:{uuid4().hex}"
    result = build_search_result()
    payload = serialize_search_results((result,))

    try:
        asyncio.run(client.set(cache_key, payload, ex=3))
        raw_payload = asyncio.run(client.get(cache_key))
    except RedisError as exc:
        pytest.skip(f"Redis integration unavailable: {type(exc).__name__}")
    finally:
        try:
            asyncio.run(client.delete(cache_key))
        except RedisError:
            pass
        asyncio.run(client.aclose())

    assert cache_key.startswith("integration:retrieval:")
    assert raw_payload == payload
    assert deserialize_search_results(raw_payload or "") == (result,)
