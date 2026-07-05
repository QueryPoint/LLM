import logging
from typing import Protocol

from assistant_service.agents.intent_agent import IntentDecision
from assistant_service.services.elasticsearch_client import DocumentMetadata, SearchResult
from assistant_service.services.retrieval_cache import (
    build_retrieval_cache_key,
    deserialize_search_results,
    normalize_retrieval_keywords,
    serialize_search_results,
)
from assistant_service.services.redis_state import RedisStateStore

logger = logging.getLogger(__name__)


class SearchClient(Protocol):
    @property
    def index_name(self) -> str:
        ...

    async def search(
        self,
        *,
        query_text: str,
        user_id: str,
        document_id: str | None,
    ) -> tuple[SearchResult, ...]:
        ...

    async def get_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> DocumentMetadata | None:
        ...


class RetrievalService:
    def __init__(
        self,
        search_client: SearchClient,
        redis_state: RedisStateStore,
    ) -> None:
        self._search_client = search_client
        self._redis_state = redis_state

    async def search(
        self,
        *,
        intent_decision: IntentDecision,
        user_id: str,
        document_id: str | None,
    ) -> tuple[SearchResult, ...]:
        normalized_keywords = normalize_retrieval_keywords(intent_decision.keywords)
        query_text = " ".join(normalized_keywords).strip()
        normalized_user_id = user_id.strip()
        if query_text == "" or normalized_user_id == "":
            return ()

        cache_key = build_retrieval_cache_key(
            index_name=self._search_client.index_name,
            keywords=normalized_keywords,
            user_id=normalized_user_id,
            document_id=document_id,
        )
        cached_results = await self._get_cached_results(cache_key)
        if cached_results is not None:
            logger.info("Retrieval cache lookup: cache_hit=%s", True)
            return cached_results

        logger.info("Retrieval cache lookup: cache_hit=%s", False)
        results = await self._search_client.search(
            query_text=query_text,
            user_id=normalized_user_id,
            document_id=document_id,
        )
        if results:
            await self.cache_results(
                intent_decision=intent_decision,
                user_id=normalized_user_id,
                document_id=document_id,
                results=results,
            )
        return results

    async def get_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> DocumentMetadata | None:
        if user_id.strip() == "" or document_id.strip() == "":
            return None

        return await self._search_client.get_document_metadata(
            user_id=user_id,
            document_id=document_id,
        )

    async def _get_cached_results(self, cache_key: str) -> tuple[SearchResult, ...] | None:
        raw_payload = await self._redis_state.get_retrieval_cache_payload(key=cache_key)
        if raw_payload is None:
            return None

        results = deserialize_search_results(raw_payload)
        if results is not None:
            return results

        await self._redis_state.delete_retrieval_cache_payload(key=cache_key)
        return None

    async def cache_results(
        self,
        *,
        intent_decision: IntentDecision,
        user_id: str,
        document_id: str | None,
        results: tuple[SearchResult, ...],
    ) -> None:
        if not results:
            return

        normalized_keywords = normalize_retrieval_keywords(intent_decision.keywords)
        cache_key = build_retrieval_cache_key(
            index_name=self._search_client.index_name,
            keywords=normalized_keywords,
            user_id=user_id,
            document_id=document_id,
        )
        await self._redis_state.set_retrieval_cache_payload(
            key=cache_key,
            payload=serialize_search_results(results),
        )
