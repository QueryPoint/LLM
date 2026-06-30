from uuid import UUID

from assistant_service.agents.context_agent import ContextAgent
from assistant_service.core.enums import RetrievalStatus
from assistant_service.messaging.contracts import DocumentContext, RetrievedChunk

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000004")
CHUNK_ID_1 = UUID("00000000-0000-0000-0000-000000000101")
CHUNK_ID_2 = UUID("00000000-0000-0000-0000-000000000102")
CHUNK_ID_3 = UUID("00000000-0000-0000-0000-000000000103")
CHUNK_ID_4 = UUID("00000000-0000-0000-0000-000000000104")


def _chunk(chunk_id: UUID, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        file_name="lecture.pdf",
        page=1,
        text=text,
        score=score,
    )


def test_missing_document_context_returns_not_found() -> None:
    decision = ContextAgent().prepare(None)

    assert decision.status == RetrievalStatus.NOT_FOUND
    assert decision.chunks == ()
    assert decision.sources == ()
    assert decision.total_chars == 0


def test_not_found_context_ignores_chunks() -> None:
    context = DocumentContext(
        retrieval_status=RetrievalStatus.NOT_FOUND,
        chunks=[_chunk(CHUNK_ID_1, "Unused text", 9.0)],
    )

    decision = ContextAgent().prepare(context)

    assert decision.status == RetrievalStatus.NOT_FOUND
    assert decision.chunks == ()
    assert decision.sources == ()


def test_duplicates_keep_higher_score_by_chunk_id_and_text() -> None:
    context = DocumentContext(
        retrieval_status=RetrievalStatus.FOUND,
        chunks=[
            _chunk(CHUNK_ID_1, "Same chunk id loses", 1.0),
            _chunk(CHUNK_ID_1, "Same chunk id wins", 4.0),
            _chunk(CHUNK_ID_3, "Repeated text", 2.0),
            _chunk(CHUNK_ID_4, " repeated   TEXT ", 5.0),
        ],
    )

    decision = ContextAgent().prepare(context)

    assert [chunk.chunk_id for chunk in decision.chunks] == [CHUNK_ID_4, CHUNK_ID_1]
    assert [chunk.score for chunk in decision.chunks] == [5.0, 4.0]
    assert [source.chunk_id for source in decision.sources] == [CHUNK_ID_4, CHUNK_ID_1]


def test_budget_and_max_chunks_select_whole_chunks_only() -> None:
    context = DocumentContext(
        retrieval_status=RetrievalStatus.FOUND,
        chunks=[
            _chunk(CHUNK_ID_1, "12345", 10.0),
            _chunk(CHUNK_ID_2, "abcdef", 9.0),
            _chunk(CHUNK_ID_3, "xy", 8.0),
            _chunk(CHUNK_ID_4, "z", 7.0),
        ],
    )

    decision = ContextAgent(max_context_chars=8, max_chunks=2).prepare(context)

    assert [chunk.chunk_id for chunk in decision.chunks] == [CHUNK_ID_1, CHUNK_ID_3]
    assert decision.total_chars == 7


def test_found_without_selected_chunks_becomes_insufficient() -> None:
    context = DocumentContext(
        retrieval_status=RetrievalStatus.FOUND,
        chunks=[_chunk(CHUNK_ID_1, "too long", 1.0)],
    )

    decision = ContextAgent(max_context_chars=3).prepare(context)

    assert decision.status == RetrievalStatus.INSUFFICIENT
    assert decision.chunks == ()
