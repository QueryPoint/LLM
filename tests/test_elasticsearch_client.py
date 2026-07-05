import asyncio

from assistant_service.services.elasticsearch_client import ElasticsearchClient


class FakeAsyncElasticsearch:
    def __init__(self, response: dict[str, object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response

    async def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
        self.calls.append({"index": index, "body": body})
        return self._response

    async def close(self) -> None:
        return None


def _response() -> dict[str, object]:
    return {
        "hits": {
            "hits": [
                {
                    "_score": 12.5,
                    "_source": {
                        "chunk_id": "chunk-1",
                        "doc_id": "document-1",
                        "file_name": "lecture.pdf",
                        "page_number": 3,
                        "text": "  Нормализация уменьшает избыточность. ",
                    },
                    "highlight": {"text": ["<em>Нормализация</em>"]},
                }
            ]
        }
    }


def test_global_search_queries_text_without_doc_filter_and_normalizes_hit() -> None:
    fake_client = FakeAsyncElasticsearch(response=_response())
    client = ElasticsearchClient(fake_client, index_name="documents", max_results=8)

    results = asyncio.run(client.search(query_text="нормализация", uid=None))

    assert len(results) == 1
    assert results[0].doc_id == "document-1"
    assert results[0].chunk_id == "chunk-1"
    assert results[0].file_name == "lecture.pdf"
    assert results[0].page_number == 3
    assert results[0].text == "Нормализация уменьшает избыточность."
    assert results[0].score == 12.5
    assert results[0].highlights == ("<em>Нормализация</em>",)

    body = fake_client.calls[0]["body"]
    assert body["query"] == {"match": {"text": {"query": "нормализация"}}}
    assert "doc_id" not in str(body["query"])
    assert "user_id" not in str(body)


def test_document_scoped_search_filters_by_doc_id_without_global_fallback() -> None:
    fake_client = FakeAsyncElasticsearch(response={"hits": {"hits": []}})
    client = ElasticsearchClient(fake_client, index_name="documents", max_results=5)

    results = asyncio.run(
        client.search(query_text="нормальные формы", uid="document-uid-123")
    )

    assert results == ()
    assert len(fake_client.calls) == 1
    body = fake_client.calls[0]["body"]
    assert body["size"] == 5
    assert body["query"] == {
        "bool": {
            "filter": [{"term": {"doc_id": "document-uid-123"}}],
            "must": [
                {"match": {"text": {"query": "нормальные формы"}}},
            ],
        }
    }
    assert "user_id" not in str(body)


def test_document_metadata_lookup_filters_by_doc_id_and_reads_file_name() -> None:
    fake_client = FakeAsyncElasticsearch(
        response={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "doc_id": "document-uid-123",
                            "file_name": " lecture.PDF ",
                        }
                    }
                ]
            }
        }
    )
    client = ElasticsearchClient(fake_client, index_name="documents", max_results=8)

    metadata = asyncio.run(client.get_document_metadata(uid="document-uid-123"))

    assert metadata is not None
    assert metadata.doc_id == "document-uid-123"
    assert metadata.file_name == "lecture.PDF"
    body = fake_client.calls[0]["body"]
    assert body == {
        "size": 1,
        "_source": ["doc_id", "file_name"],
        "query": {"term": {"doc_id": "document-uid-123"}},
    }
    assert "user_id" not in str(body)
