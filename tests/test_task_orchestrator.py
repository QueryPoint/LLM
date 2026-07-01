import asyncio
from uuid import UUID

import pytest

from assistant_service.agents.context_agent import ContextDecision
from assistant_service.agents.intent_agent import IntentDecision
from assistant_service.core.enums import (
    AssistantMode,
    LLMStatus,
    OutgoingEventType,
    RetrievalStatus,
)
from assistant_service.messaging.contracts import (
    IncomingMessage,
    OutgoingEvent,
    ResponseEvent,
    ThinkEvent,
    RetrievedChunk,
    incoming_message_adapter,
)
from assistant_service.services.task_orchestrator import (
    INTENT_DETECTED_TEXT,
    SAFE_ERROR_RESPONSE,
    STATUS_MESSAGES,
    TaskOrchestrator,
)
from assistant_service.services.gemini_client import GeminiRateLimitError


USER_ID = "00000000-0000-0000-0000-000000000002"
DOC_ID = "00000000-0000-0000-0000-000000000004"
CHUNK_ID = "00000000-0000-0000-0000-000000000101"
MAX_USER_PROMPT_CHARS = 4_000
MAX_CHUNK_CHARS = 12_000
MAX_CHUNKS_PER_REQUEST = 32


class FakePublisher:
    def __init__(
        self,
        fail_on_think: bool = False,
        fail_on_response: bool = False,
    ) -> None:
        self.events: list[OutgoingEvent] = []
        self._fail_on_think = fail_on_think
        self._fail_on_response = fail_on_response

    async def publish_event(self, event: OutgoingEvent) -> None:
        if self._fail_on_think and isinstance(event, ThinkEvent):
            raise RuntimeError("think publish failed")

        if self._fail_on_response and isinstance(event, ResponseEvent):
            raise RuntimeError("response publish failed")

        self.events.append(event)


class FakeIntentAgent:
    def __init__(
        self,
        fail: bool = False,
        mode: AssistantMode = AssistantMode.ANSWER_QUESTION,
    ) -> None:
        self.calls: list[tuple[str | None, UUID | None, AssistantMode | None]] = []
        self._fail = fail
        self._mode = mode

    def detect(
        self,
        prompt: str | None,
        document_id: UUID | None,
        requested_mode: AssistantMode | None,
    ) -> IntentDecision:
        self.calls.append((prompt, document_id, requested_mode))
        if self._fail:
            raise RuntimeError("intent detection failed")
        return IntentDecision(
            mode=self._mode,
            source="rule_based",
        )


class FakeContextAgent:
    def __init__(
        self,
        status: RetrievalStatus = RetrievalStatus.FOUND,
        chunks: tuple[RetrievedChunk, ...] = (),
    ) -> None:
        self.calls: list[tuple[object, AssistantMode | None]] = []
        self._status = status
        self._chunks = chunks

    def prepare(
        self,
        document_context: object,
        mode: AssistantMode | None = None,
    ) -> ContextDecision:
        self.calls.append((document_context, mode))
        return ContextDecision(
            status=self._status,
            chunks=self._chunks,
            sources=(),
            total_chars=sum(len(chunk.text) for chunk in self._chunks),
        )


class FakeAnswerAgent:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[AssistantMode, str | None, ContextDecision]] = []
        self._error = error

    async def answer(
        self,
        *,
        mode: AssistantMode,
        user_prompt: str | None,
        context_decision: ContextDecision,
    ) -> str:
        self.calls.append((mode, user_prompt, context_decision))
        if self._error is not None:
            raise self._error
        return "Generated answer"


class FakeDocumentSummaryAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, ContextDecision]] = []

    async def summarize(
        self,
        *,
        user_prompt: str | None,
        context_decision: ContextDecision,
    ) -> str:
        self.calls.append((user_prompt, context_decision))
        return "Document summary"


class FakeRedisState:
    def __init__(self, cached_answer: str | None = None) -> None:
        self.cached_answer = cached_answer
        self.answer_get_keys: list[str] = []
        self.answer_set_calls: list[tuple[str, str]] = []
        self.intent_get_keys: list[str] = []
        self.intent_set_calls: list[tuple[str, AssistantMode]] = []
        self.task_statuses: list[tuple[UUID, str]] = []
        self.lock_keys: list[str] = []
        self.release_calls: list[object] = []
        self.clear_user_state_calls: list[UUID] = []

    async def get_answer(self, key: str) -> str | None:
        self.answer_get_keys.append(key)
        return self.cached_answer

    async def set_answer(self, key: str, answer: str) -> None:
        self.answer_set_calls.append((key, answer))

    async def acquire_answer_lock(self, *, lock_key: str) -> None:
        self.lock_keys.append(lock_key)
        return None

    async def release_answer_lock(self, lock: object) -> None:
        self.release_calls.append(lock)

    async def get_intent_mode(self, key: str) -> AssistantMode | None:
        self.intent_get_keys.append(key)
        return None

    async def set_intent_mode(self, key: str, mode: AssistantMode) -> None:
        self.intent_set_calls.append((key, mode))

    async def set_task_status(self, *, user_id: UUID, status: str) -> None:
        self.task_statuses.append((user_id, status))

    async def clear_user_state(self, *, user_id: UUID) -> None:
        self.clear_user_state_calls.append(user_id)


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(CHUNK_ID),
        document_id=UUID(DOC_ID),
        file_name="lecture.pdf",
        page=3,
        text="RabbitMQ — брокер сообщений.",
        score=9.42,
    )


def _prompt_message() -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Explain normalization",
            "doc": DOC_ID,
            "mode": "summarize_document",
            "document_context": {
                "retrieval_status": "found",
                "chunks": [
                    {
                        "chunk_id": "00000000-0000-0000-0000-000000000101",
                        "document_id": DOC_ID,
                        "file_name": "lecture.pdf",
                        "page": 1,
                        "text": "Context text",
                        "score": 1.0,
                    }
                ],
            },
        }
    )


def _delete_message() -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "delete",
            "user_id": USER_ID,
        }
    )


def test_prompt_publishes_think_statuses_then_response() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(mode=AssistantMode.DOCUMENT_SEARCH)
    context_agent = FakeContextAgent(chunks=(_chunk(),))
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        redis_state=FakeRedisState(),
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    message = _prompt_message()
    asyncio.run(orchestrator.handle(message))

    assert [event.type for event in publisher.events] == [
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.RESPONSE,
    ]
    assert [event.data for event in publisher.events[:-1]] == [
        STATUS_MESSAGES[LLMStatus.PROCESSING],
        STATUS_MESSAGES[LLMStatus.THINKING],
        INTENT_DETECTED_TEXT,
        STATUS_MESSAGES[LLMStatus.SEARCHING],
        STATUS_MESSAGES[LLMStatus.FOUND],
        STATUS_MESSAGES[LLMStatus.GENERATING],
    ]
    assert publisher.events[-1].warning == 0
    assert publisher.events[-1].data == (
        "Нашёл подходящие фрагменты:\n\n"
        "1. lecture.pdf, стр. 3\n"
        "RabbitMQ — брокер сообщений."
    )
    assert intent_agent.calls == [
        (
            "Explain normalization",
            UUID(DOC_ID),
            AssistantMode.SUMMARIZE_DOCUMENT,
        )
    ]
    assert context_agent.calls == [
        (message.document_context, AssistantMode.DOCUMENT_SEARCH)
    ]
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []


def test_answer_question_found_context_publishes_single_buffered_response() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(mode=AssistantMode.ANSWER_QUESTION)
    context_agent = FakeContextAgent(chunks=(_chunk(),))
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        redis_state=FakeRedisState(),
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert len(answer_agent.calls) == 1
    assert answer_agent.calls[0][0] == AssistantMode.ANSWER_QUESTION
    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == "Generated answer"
    assert [event.type for event in publisher.events[:-1]] == [
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
    ]
    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert {event.type.value for event in publisher.events} == {"think", "response"}
    assert document_summary_agent.calls == []


def test_answer_question_found_context_uses_cached_answer_without_agent() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(mode=AssistantMode.ANSWER_QUESTION)
    context_agent = FakeContextAgent(chunks=(_chunk(),))
    answer_agent = FakeAnswerAgent()
    redis_state = FakeRedisState(cached_answer="Cached answer")
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=FakeDocumentSummaryAgent(),
        redis_state=redis_state,
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == "Cached answer"
    assert answer_agent.calls == []
    assert redis_state.answer_get_keys
    assert redis_state.answer_set_calls == []
    assert redis_state.lock_keys == []
    assert {event.type.value for event in publisher.events} == {"think", "response"}
    assert "processing" in [status for _, status in redis_state.task_statuses]
    assert redis_state.task_statuses[-1] == (UUID(USER_ID), "completed")


def test_gemini_rate_limit_error_publishes_safe_response() -> None:
    publisher = FakePublisher()
    redis_state = FakeRedisState()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(mode=AssistantMode.ANSWER_QUESTION),
        context_agent=FakeContextAgent(chunks=(_chunk(),)),
        answer_agent=FakeAnswerAgent(error=GeminiRateLimitError("rate limit")),
        document_summary_agent=FakeDocumentSummaryAgent(),
        redis_state=redis_state,
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == (
        "Сервис временно перегружен. Подождите немного и повторите запрос."
    )
    assert redis_state.task_statuses[-1] == (UUID(USER_ID), "completed")


def test_oversized_user_prompt_returns_response_without_agents() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(mode=AssistantMode.ANSWER_QUESTION)
    context_agent = FakeContextAgent(chunks=(_chunk(),))
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    redis_state = FakeRedisState()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        redis_state=redis_state,
        max_user_prompt_chars=3,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.RESPONSE
    assert publisher.events[0].data == (
        "Запрос слишком большой. Сократите его и попробуйте ещё раз."
    )
    assert intent_agent.calls == []
    assert context_agent.calls == []
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []
    assert redis_state.task_statuses[-1] == (UUID(USER_ID), "completed")


def test_summarize_document_found_context_uses_document_summary_agent() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(mode=AssistantMode.SUMMARIZE_DOCUMENT)
    context_agent = FakeContextAgent(chunks=(_chunk(),))
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        redis_state=FakeRedisState(),
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert answer_agent.calls == []
    assert len(document_summary_agent.calls) == 1
    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == "Document summary"
    assert {event.type.value for event in publisher.events} == {"think", "response"}


def test_delete_publishes_only_think_confirmation() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent()
    context_agent = FakeContextAgent()
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    redis_state = FakeRedisState()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        redis_state=redis_state,
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_delete_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.THINK
    assert publisher.events[0].data == "История диалога очищена."
    assert intent_agent.calls == []
    assert context_agent.calls == []
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []
    assert redis_state.clear_user_state_calls == [UUID(USER_ID)]


def test_handled_processing_error_publishes_safe_response() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(fail=True),
        context_agent=FakeContextAgent(),
        answer_agent=FakeAnswerAgent(),
        document_summary_agent=FakeDocumentSummaryAgent(),
        redis_state=FakeRedisState(),
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == SAFE_ERROR_RESPONSE


def test_safe_response_publish_failure_is_reraised() -> None:
    publisher = FakePublisher(fail_on_response=True)
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(fail=True),
        context_agent=FakeContextAgent(),
        answer_agent=FakeAnswerAgent(),
        document_summary_agent=FakeDocumentSummaryAgent(),
        redis_state=FakeRedisState(),
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.handle(_prompt_message()))
