from uuid import UUID

from assistant_service.agents.context_agent import ContextDecision
from assistant_service.core.enums import AssistantMode, RetrievalStatus
from assistant_service.messaging.contracts import RetrievedChunk
from assistant_service.services.cache_keys import (
    build_answer_cache_key,
    build_intent_cache_key,
)
from assistant_service.services.retrieval_cache import build_retrieval_cache_key


USER_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_USER_ID = UUID("00000000-0000-0000-0000-000000000002")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000003")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000004")


def _context_decision(text: str = "Redis cache context") -> ContextDecision:
    chunk = RetrievedChunk(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        file_name="lecture.pdf",
        page=1,
        text=text,
        score=9.9,
    )
    return ContextDecision(
        status=RetrievalStatus.FOUND,
        chunks=(chunk,),
        sources=(),
        total_chars=len(chunk.text),
    )


def test_cache_keys_are_deterministic_and_do_not_expose_plaintext() -> None:
    prompt = " Explain Redis cache "
    first_key = build_answer_cache_key(
        user_id=USER_ID,
        mode=AssistantMode.ANSWER_QUESTION,
        user_prompt=prompt,
        context_decision=_context_decision(),
    )
    second_key = build_answer_cache_key(
        user_id=USER_ID,
        mode=AssistantMode.ANSWER_QUESTION,
        user_prompt=prompt,
        context_decision=_context_decision(),
    )
    changed_chunk_key = build_answer_cache_key(
        user_id=USER_ID,
        mode=AssistantMode.ANSWER_QUESTION,
        user_prompt=prompt,
        context_decision=_context_decision("Changed context"),
    )
    changed_user_key = build_answer_cache_key(
        user_id=OTHER_USER_ID,
        mode=AssistantMode.ANSWER_QUESTION,
        user_prompt=prompt,
        context_decision=_context_decision(),
    )

    assert first_key == second_key
    assert first_key != changed_chunk_key
    assert first_key != changed_user_key
    assert first_key.startswith("answer:v1:")
    assert "Explain Redis cache" not in first_key
    assert str(USER_ID) not in first_key
    assert str(CHUNK_ID) not in first_key

    intent_key = build_intent_cache_key(
        prompt=" Explain Redis cache ",
        document_id="document-id-123",
    )
    same_intent_key = build_intent_cache_key(
        prompt="explain   redis cache",
        document_id="document-id-123",
    )
    changed_document_id_key = build_intent_cache_key(
        prompt="explain   redis cache",
        document_id="another-document",
    )

    assert intent_key == same_intent_key
    assert intent_key != changed_document_id_key
    assert intent_key.startswith("intent:v1:")

    retrieval_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("Redis", "cache"),
        user_id=str(USER_ID),
        document_id="document-id-123",
    )
    same_retrieval_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("Redis", "cache"),
        user_id=str(USER_ID),
        document_id="document-id-123",
    )
    different_user_retrieval_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("Redis", "cache"),
        user_id=str(OTHER_USER_ID),
        document_id="document-id-123",
    )
    different_document_retrieval_key = build_retrieval_cache_key(
        index_name="documents",
        keywords=("Redis", "cache"),
        user_id=str(USER_ID),
        document_id="another-document",
    )

    assert retrieval_key == same_retrieval_key
    assert retrieval_key != different_user_retrieval_key
    assert retrieval_key != different_document_retrieval_key
    assert retrieval_key.startswith("retrieval:v1:")
    assert str(USER_ID) not in retrieval_key
    assert str(OTHER_USER_ID) not in retrieval_key
