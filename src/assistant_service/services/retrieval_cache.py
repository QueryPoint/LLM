import hashlib
import json
from collections.abc import Sequence

from assistant_service.services.elasticsearch_client import SearchResult

RETRIEVAL_CACHE_PREFIX = "retrieval:v1:"
RETRIEVAL_CACHE_SCHEMA_VERSION = 1


def build_retrieval_cache_key(
    *,
    index_name: str,
    keywords: Sequence[str],
    document_id: str | None,
) -> str:
    normalized_index_name = index_name.strip()
    if normalized_index_name == "":
        raise ValueError("index_name must not be empty")

    normalized_keywords = _normalize_keywords(keywords)
    normalized_document_id = _normalize_document_id(document_id)
    fingerprint = "|".join(
        (
            normalized_index_name,
            " ".join(normalized_keywords),
            normalized_document_id,
        )
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return f"{RETRIEVAL_CACHE_PREFIX}{digest}"


def normalize_retrieval_keywords(keywords: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalize_keywords(keywords))


def serialize_search_results(results: Sequence[SearchResult]) -> str:
    payload = {
        "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
        "results": [
            {
                "doc_id": result.doc_id,
                "chunk_id": result.chunk_id,
                "file_name": result.file_name,
                "page_number": result.page_number,
                "text": result.text,
                "score": result.score,
                "highlights": list(result.highlights),
            }
            for result in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deserialize_search_results(payload: str) -> tuple[SearchResult, ...] | None:
    try:
        raw_payload = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(raw_payload, dict):
        return None

    if raw_payload.get("schema_version") != RETRIEVAL_CACHE_SCHEMA_VERSION:
        return None

    raw_results = raw_payload.get("results")
    if not isinstance(raw_results, list):
        return None

    results: list[SearchResult] = []
    for raw_result in raw_results:
        result = _deserialize_search_result(raw_result)
        if result is None:
            return None
        results.append(result)

    return tuple(results)


def _normalize_keywords(keywords: Sequence[str]) -> list[str]:
    normalized_keywords: list[str] = []
    seen_keywords: set[str] = set()

    for keyword in keywords:
        normalized_keyword = " ".join(keyword.strip().lower().split())
        if normalized_keyword == "":
            continue

        if normalized_keyword in seen_keywords:
            continue

        seen_keywords.add(normalized_keyword)
        normalized_keywords.append(normalized_keyword)

    return normalized_keywords


def _normalize_document_id(document_id: str | None) -> str:
    if document_id is None:
        return "global"

    normalized_document_id = document_id.strip()
    return normalized_document_id or "global"


def _deserialize_search_result(raw_result: object) -> SearchResult | None:
    if not isinstance(raw_result, dict):
        return None

    doc_id = _required_string(raw_result.get("doc_id"))
    chunk_id = _required_string(raw_result.get("chunk_id"))
    file_name = _required_string(raw_result.get("file_name"))
    text = _required_string(raw_result.get("text"))
    page_number = raw_result.get("page_number")
    score = raw_result.get("score")
    highlights = raw_result.get("highlights")

    if (
        doc_id is None
        or chunk_id is None
        or file_name is None
        or text is None
        or not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or not isinstance(score, int | float)
        or isinstance(score, bool)
    ):
        return None

    normalized_highlights = _normalize_highlights(highlights)
    if normalized_highlights is None:
        return None

    return SearchResult(
        doc_id=doc_id,
        chunk_id=chunk_id,
        file_name=file_name,
        page_number=page_number,
        text=text,
        score=float(score),
        highlights=normalized_highlights,
    )


def _normalize_highlights(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None

    normalized_highlights: list[str] = []
    for highlight in value:
        if not isinstance(highlight, str):
            return None

        stripped_highlight = highlight.strip()
        if stripped_highlight == "":
            return None

        normalized_highlights.append(stripped_highlight)

    return tuple(normalized_highlights)


def _required_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    stripped_value = value.strip()
    return stripped_value or None
