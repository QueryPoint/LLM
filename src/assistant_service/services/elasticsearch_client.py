import logging
from collections.abc import Mapping
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import (
    ApiError,
    ConnectionError,
    ConnectionTimeout,
    NotFoundError,
    SerializationError,
    TransportError,
)

from assistant_service.core.config import Settings

logger = logging.getLogger(__name__)

SOURCE_FIELDS = ("chunk_id", "doc_id", "file_name", "page_number", "text")
METADATA_SOURCE_FIELDS = ("doc_id", "file_name")


class ElasticsearchClientError(RuntimeError):
    """Raised when Elasticsearch retrieval cannot be completed safely."""


class ElasticsearchUnavailableError(ElasticsearchClientError):
    """Elasticsearch is temporarily unavailable."""


class ElasticsearchIndexNotFoundError(ElasticsearchClientError):
    """Configured Elasticsearch index does not exist."""


class ElasticsearchResponseError(ElasticsearchClientError):
    """Elasticsearch returned an unexpected response."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: str
    chunk_id: str
    file_name: str
    page_number: int
    text: str
    score: float
    highlights: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    doc_id: str
    file_name: str


class ElasticsearchClient:
    def __init__(
        self,
        client: AsyncElasticsearch,
        *,
        index_name: str,
        max_results: int,
    ) -> None:
        normalized_index_name = index_name.strip()
        if normalized_index_name == "":
            raise ValueError("index_name must not be empty")
        if max_results <= 0:
            raise ValueError("max_results must be positive")

        self._client = client
        self._index_name = normalized_index_name
        self._max_results = max_results

    @property
    def index_name(self) -> str:
        return self._index_name

    async def search(
        self,
        *,
        query_text: str,
        user_id: str,
        document_id: str | None,
    ) -> tuple[SearchResult, ...]:
        normalized_query_text = query_text.strip()
        normalized_user_id = user_id.strip()
        if normalized_query_text == "" or normalized_user_id == "":
            return ()

        query = self._build_query(
            query_text=normalized_query_text,
            user_id=normalized_user_id,
            document_id=document_id,
        )
        try:
            response = await self._client.search(index=self._index_name, body=query)
        except NotFoundError as exc:
            self._log_expected_error("search", exc)
            raise ElasticsearchIndexNotFoundError(
                "Elasticsearch index is not available."
            ) from exc
        except (ConnectionError, ConnectionTimeout) as exc:
            self._log_expected_error("search", exc)
            raise ElasticsearchUnavailableError(
                "Elasticsearch is not available."
            ) from exc
        except (ApiError, SerializationError, TransportError) as exc:
            self._log_expected_error("search", exc)
            raise ElasticsearchResponseError(
                "Elasticsearch returned an invalid response."
            ) from exc

        return self._normalize_response(response)

    async def get_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> DocumentMetadata | None:
        normalized_user_id = user_id.strip()
        normalized_document_id = document_id.strip()
        if normalized_user_id == "" or normalized_document_id == "":
            return None

        query = {
            "size": 1,
            "_source": list(METADATA_SOURCE_FIELDS),
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"user_id": normalized_user_id}},
                        {"term": {"doc_id": normalized_document_id}},
                    ]
                }
            },
        }
        try:
            response = await self._client.search(index=self._index_name, body=query)
        except NotFoundError as exc:
            self._log_expected_error("metadata_lookup", exc)
            raise ElasticsearchIndexNotFoundError(
                "Elasticsearch index is not available."
            ) from exc
        except (ConnectionError, ConnectionTimeout) as exc:
            self._log_expected_error("metadata_lookup", exc)
            raise ElasticsearchUnavailableError(
                "Elasticsearch is not available."
            ) from exc
        except (ApiError, SerializationError, TransportError) as exc:
            self._log_expected_error("metadata_lookup", exc)
            raise ElasticsearchResponseError(
                "Elasticsearch returned an invalid response."
            ) from exc

        return self._normalize_metadata_response(response)

    async def aclose(self) -> None:
        await self._client.close()

    def _build_query(
        self,
        *,
        query_text: str,
        user_id: str,
        document_id: str | None,
    ) -> dict[str, object]:
        normalized_user_id = user_id.strip()
        normalized_document_id = (
            document_id.strip() if isinstance(document_id, str) else None
        )
        base_query: dict[str, object] = {
            "size": self._max_results,
            "_source": list(SOURCE_FIELDS),
            "highlight": {"fields": {"text": {}}},
        }

        match_query = {"match": {"text": {"query": query_text}}}
        filters: list[dict[str, object]] = [{"term": {"user_id": normalized_user_id}}]
        if normalized_document_id:
            filters.append({"term": {"doc_id": normalized_document_id}})

        base_query["query"] = {
            "bool": {
                "filter": filters,
                "must": [match_query],
            }
        }
        return base_query

    def _normalize_response(self, response: object) -> tuple[SearchResult, ...]:
        response_body = getattr(response, "body", response)
        if not isinstance(response_body, Mapping):
            raise ElasticsearchResponseError(
                "Elasticsearch response has unexpected format."
            )

        hits_container = response_body.get("hits")
        if not isinstance(hits_container, Mapping):
            raise ElasticsearchResponseError(
                "Elasticsearch response has unexpected hits format."
            )

        raw_hits = hits_container.get("hits", [])
        if not isinstance(raw_hits, list):
            raise ElasticsearchResponseError(
                "Elasticsearch response has unexpected hit list format."
            )

        results = [
            result
            for hit in raw_hits
            if (result := self._normalize_hit(hit)) is not None
        ]
        return tuple(results)

    def _normalize_metadata_response(self, response: object) -> DocumentMetadata | None:
        response_body = getattr(response, "body", response)
        if not isinstance(response_body, Mapping):
            raise ElasticsearchResponseError(
                "Elasticsearch response has unexpected format."
            )

        hits_container = response_body.get("hits")
        if not isinstance(hits_container, Mapping):
            raise ElasticsearchResponseError(
                "Elasticsearch response has unexpected hits format."
            )

        raw_hits = hits_container.get("hits", [])
        if not isinstance(raw_hits, list):
            raise ElasticsearchResponseError(
                "Elasticsearch response has unexpected hit list format."
            )

        for hit in raw_hits:
            metadata = self._normalize_metadata_hit(hit)
            if metadata is not None:
                return metadata
        return None

    @staticmethod
    def _normalize_hit(hit: object) -> SearchResult | None:
        if not isinstance(hit, Mapping):
            return None

        source = hit.get("_source")
        if not isinstance(source, Mapping):
            return None

        doc_id = ElasticsearchClient._required_string(source.get("doc_id"))
        chunk_id = ElasticsearchClient._required_string(source.get("chunk_id"))
        file_name = ElasticsearchClient._required_string(source.get("file_name"))
        text = ElasticsearchClient._required_string(source.get("text"))
        page_number = source.get("page_number")

        if (
            doc_id is None
            or chunk_id is None
            or file_name is None
            or text is None
            or not isinstance(page_number, int)
            or isinstance(page_number, bool)
        ):
            return None

        score = hit.get("_score")
        normalized_score = (
            float(score)
            if isinstance(score, int | float) and not isinstance(score, bool)
            else 0.0
        )
        highlights = ElasticsearchClient._normalize_highlights(hit.get("highlight"))

        return SearchResult(
            doc_id=doc_id,
            chunk_id=chunk_id,
            file_name=file_name,
            page_number=page_number,
            text=text,
            score=normalized_score,
            highlights=highlights,
        )

    @staticmethod
    def _normalize_metadata_hit(hit: object) -> DocumentMetadata | None:
        if not isinstance(hit, Mapping):
            return None

        source = hit.get("_source")
        if not isinstance(source, Mapping):
            return None

        doc_id = ElasticsearchClient._required_string(source.get("doc_id"))
        file_name = ElasticsearchClient._required_string(source.get("file_name"))
        if doc_id is None or file_name is None:
            return None

        return DocumentMetadata(doc_id=doc_id, file_name=file_name)

    @staticmethod
    def _required_string(value: object) -> str | None:
        if not isinstance(value, str):
            return None

        stripped_value = value.strip()
        return stripped_value or None

    @staticmethod
    def _normalize_highlights(value: object) -> tuple[str, ...]:
        if not isinstance(value, Mapping):
            return ()

        raw_text_highlights = value.get("text")
        if not isinstance(raw_text_highlights, list):
            return ()

        return tuple(
            stripped_highlight
            for highlight in raw_text_highlights
            if isinstance(highlight, str)
            and (stripped_highlight := highlight.strip()) != ""
        )

    @staticmethod
    def _log_expected_error(operation: str, exc: Exception) -> None:
        logger.warning(
            "Elasticsearch operation failed: operation=%s error_type=%s",
            operation,
            type(exc).__name__,
        )


def create_elasticsearch_client_from_settings(
    settings: Settings,
) -> ElasticsearchClient:
    client = AsyncElasticsearch(
        hosts=[settings.elasticsearch_url],
        request_timeout=settings.elasticsearch_timeout_seconds,
    )
    return ElasticsearchClient(
        client,
        index_name=settings.elasticsearch_index,
        max_results=settings.elasticsearch_max_results,
    )
