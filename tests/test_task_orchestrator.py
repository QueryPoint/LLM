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


USER_ID = "00000000-0000-0000-0000-000000000002"
DOC_ID = "00000000-0000-0000-0000-000000000004"
CHUNK_ID = "00000000-0000-0000-0000-000000000101"


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
    def __init__(self) -> None:
        self.calls: list[tuple[AssistantMode, str | None, ContextDecision]] = []

    async def answer(
        self,
        *,
        mode: AssistantMode,
        user_prompt: str | None,
        context_decision: ContextDecision,
    ) -> str:
        self.calls.append((mode, user_prompt, context_decision))
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
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
    )

    asyncio.run(orchestrator.handle(_delete_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.THINK
    assert publisher.events[0].data == "История диалога очищена."
    assert intent_agent.calls == []
    assert context_agent.calls == []
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []


def test_handled_processing_error_publishes_safe_response() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(fail=True),
        context_agent=FakeContextAgent(),
        answer_agent=FakeAnswerAgent(),
        document_summary_agent=FakeDocumentSummaryAgent(),
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
    )

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.handle(_prompt_message()))
