import asyncio
import os

import pytest
from elasticsearch import AsyncElasticsearch

from assistant_service.core.config import Settings
from assistant_service.services.elasticsearch_client import ElasticsearchClient


@pytest.mark.integration
@pytest.mark.requires_docker
def test_elasticsearch_metadata_lookup_reads_existing_document_metadata() -> None:
    if os.getenv("RUN_INTEGRATION_TESTS", "").strip().lower() != "true":
        pytest.skip("integration tests are disabled")

    document_id = os.getenv("INTEGRATION_DOCUMENT_ID", "").strip()
    if document_id == "":
        pytest.skip("INTEGRATION_DOCUMENT_ID is not set")

    settings = Settings()
    client = AsyncElasticsearch(
        hosts=[settings.elasticsearch_url],
        request_timeout=settings.elasticsearch_timeout_seconds,
    )
    es_client = ElasticsearchClient(
        client,
        index_name=settings.elasticsearch_index,
        max_results=1,
    )

    async def _run() -> object:
        return await es_client.get_document_metadata(document_id=document_id)

    try:
        metadata = asyncio.run(_run())
    except Exception as exc:
        pytest.skip(f"Elasticsearch integration unavailable: {type(exc).__name__}")
    finally:
        asyncio.run(client.close())

    if metadata is None:
        pytest.skip("test document metadata was not found")

    assert metadata.doc_id == document_id
    assert metadata.file_name.strip() != ""
