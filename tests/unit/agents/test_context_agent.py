from assistant_service.agents.context_agent import ContextAgent
from assistant_service.core.enums import RetrievalStatus
from assistant_service.services.elasticsearch_client import SearchResult

DOCUMENT_ID = "document-uid-4"
CHUNK_ID_1 = "chunk-101"
CHUNK_ID_2 = "chunk-102"
CHUNK_ID_3 = "chunk-103"
CHUNK_ID_4 = "chunk-104"


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        doc_id=DOCUMENT_ID,
        chunk_id=chunk_id,
        file_name="lecture.pdf",
        page_number=1,
        text=text,
        score=score,
        highlights=(),
    )


def test_missing_document_context_returns_not_found() -> None:
    decision = ContextAgent().prepare(None)

    assert decision.status == RetrievalStatus.NOT_FOUND
    assert decision.chunks == ()
    assert decision.sources == ()
    assert decision.total_chars == 0


def test_duplicates_keep_first_retrieval_result_by_chunk_id_and_text() -> None:
    decision = ContextAgent().prepare(
        [
            _result(CHUNK_ID_1, "Same chunk id wins by relevance order", 4.0),
            _result(CHUNK_ID_1, "Same chunk id loses", 9.0),
            _result(CHUNK_ID_3, "Repeated text", 2.0),
            _result(CHUNK_ID_4, " repeated   TEXT ", 5.0),
        ]
    )

    assert [chunk.chunk_id for chunk in decision.chunks] == [CHUNK_ID_1, CHUNK_ID_3]
    assert [chunk.score for chunk in decision.chunks] == [4.0, 2.0]
    assert [source.chunk_id for source in decision.sources] == [CHUNK_ID_1, CHUNK_ID_3]


def test_budget_and_max_chunks_select_whole_chunks_only() -> None:
    decision = ContextAgent(max_context_chars=8, max_chunks=2).prepare(
        [
            _result(CHUNK_ID_1, "12345", 10.0),
            _result(CHUNK_ID_2, "abcdef", 9.0),
            _result(CHUNK_ID_3, "xy", 8.0),
            _result(CHUNK_ID_4, "z", 7.0),
        ]
    )

    assert [chunk.chunk_id for chunk in decision.chunks] == [CHUNK_ID_1, CHUNK_ID_3]
    assert decision.total_chars == 7


def test_without_selected_chunks_becomes_not_found() -> None:
    decision = ContextAgent(max_context_chars=3).prepare(
        [_result(CHUNK_ID_1, "too long", 1.0)]
    )

    assert decision.status == RetrievalStatus.NOT_FOUND
    assert decision.chunks == ()
