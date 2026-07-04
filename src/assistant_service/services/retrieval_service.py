from typing import Protocol

from assistant_service.agents.intent_agent import IntentDecision
from assistant_service.services.elasticsearch_client import SearchResult


class SearchClient(Protocol):
    async def search(
        self,
        *,
        query_text: str,
        uid: str | None,
    ) -> tuple[SearchResult, ...]:
        ...


class RetrievalService:
    def __init__(self, search_client: SearchClient) -> None:
        self._search_client = search_client

    async def search(
        self,
        *,
        intent_decision: IntentDecision,
        uid: str | None,
    ) -> tuple[SearchResult, ...]:
        query_text = " ".join(intent_decision.keywords).strip()
        if query_text == "":
            return ()

        return await self._search_client.search(query_text=query_text, uid=uid)
