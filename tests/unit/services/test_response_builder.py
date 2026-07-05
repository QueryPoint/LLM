from uuid import UUID

from assistant_service.agents.context_agent import ContextDecision
from assistant_service.core.enums import AssistantMode, RetrievalStatus
from assistant_service.messaging.contracts import RetrievedChunk
from assistant_service.services.response_builder import (
    CONTEXT_NOT_FOUND_RESPONSE,
    DOCUMENT_SEARCH_INSUFFICIENT_RESPONSE,
    DOCUMENT_SEARCH_NOT_FOUND_RESPONSE,
    TEMPORARY_CONTEXT_READY_RESPONSE,
    build_response_text,
)

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000004")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000101")


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        file_name="lecture.pdf",
        page=3,
        text="RabbitMQ — брокер сообщений.",
        score=9.42,
    )


def _decision(
    status: RetrievalStatus,
    chunks: tuple[RetrievedChunk, ...] = (),
) -> ContextDecision:
    return ContextDecision(status=status, chunks=chunks, sources=(), total_chars=0)


def test_document_search_found_returns_numbered_fragments_without_internal_fields() -> None:
    response = build_response_text(
        mode=AssistantMode.DOCUMENT_SEARCH,
        context_decision=_decision(RetrievalStatus.FOUND, (_chunk(),)),
    )

    assert response == (
        "Нашёл подходящие фрагменты:\n\n"
        "1. lecture.pdf, стр. 3\n"
        "RabbitMQ — брокер сообщений."
    )
    assert str(CHUNK_ID) not in response
    assert "9.42" not in response


def test_document_search_not_found_returns_fallback() -> None:
    response = build_response_text(
        mode=AssistantMode.DOCUMENT_SEARCH,
        context_decision=_decision(RetrievalStatus.NOT_FOUND),
    )

    assert response == DOCUMENT_SEARCH_NOT_FOUND_RESPONSE


def test_document_search_insufficient_returns_fallback() -> None:
    response = build_response_text(
        mode=AssistantMode.DOCUMENT_SEARCH,
        context_decision=_decision(RetrievalStatus.INSUFFICIENT),
    )

    assert response == DOCUMENT_SEARCH_INSUFFICIENT_RESPONSE


def test_answer_question_not_found_returns_fallback() -> None:
    response = build_response_text(
        mode=AssistantMode.ANSWER_QUESTION,
        context_decision=_decision(RetrievalStatus.NOT_FOUND),
    )

    assert response == CONTEXT_NOT_FOUND_RESPONSE


def test_explain_topic_found_returns_temporary_response() -> None:
    response = build_response_text(
        mode=AssistantMode.EXPLAIN_TOPIC,
        context_decision=_decision(RetrievalStatus.FOUND, (_chunk(),)),
    )

    assert response == TEMPORARY_CONTEXT_READY_RESPONSE
