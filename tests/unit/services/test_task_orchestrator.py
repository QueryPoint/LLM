import asyncio
import logging

import pytest

from assistant_service.agents.intent_agent import IntentDecision, IntentTaskType
from assistant_service.agents.answer_agent import AnswerGenerationRequest
from assistant_service.agents.context_agent import ContextAgent
from assistant_service.core.enums import AssistantMode, OutgoingEventType
from assistant_service.messaging.contracts import (
    IncomingMessage,
    OutgoingEvent,
    ResponseEvent,
    ThinkEvent,
    incoming_message_adapter,
)
from assistant_service.services.response_builder import (
    build_context_too_large_response,
    build_document_summary_unavailable_response,
    build_materials_not_found_response,
    build_pdf_only_summary_response,
    build_request_too_large_response,
    build_retrieval_unavailable_response,
    build_summary_requires_complete_document_response,
    build_unsupported_request_response,
)
from assistant_service.services.elasticsearch_client import (
    DocumentMetadata,
    ElasticsearchUnavailableError,
    SearchResult,
)
from assistant_service.services.task_orchestrator import (
    DETERMINE_INTENT_TEXT,
    GENERATE_ANSWER_TEXT,
    GENERATE_SUMMARY_TEXT,
    PREPARE_DOCUMENT_TEXT,
    SAFE_ERROR_RESPONSE,
    SEARCH_MATERIALS_TEXT,
    TaskOrchestrator,
)


USER_ID = "user-123"
MAX_USER_PROMPT_CHARS = 4_000
MAX_CHUNK_CHARS = 12_000
MAX_CHUNKS_PER_REQUEST = 32
MAX_PROMPT_CHARS = 16_000
DOCUMENT_ID = "document-id-123"


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
        task_type: IntentTaskType = IntentTaskType.EXPLAIN_TOPIC,
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._fail = fail
        self._task_type = task_type

    async def detect(self, prompt: str, document_id: str | None) -> IntentDecision:
        self.calls.append((prompt, document_id))
        if self._fail:
            raise RuntimeError("intent detection failed")
        return IntentDecision(
            task_type=self._task_type,
            requires_retrieval=self._task_type != IntentTaskType.UNSUPPORTED,
            requires_full_document=self._task_type == IntentTaskType.SUMMARIZE_DOCUMENT,
            keywords=[] if self._task_type == IntentTaskType.UNSUPPORTED else ["sql"],
            source="rule_based",
        )


class RecordingContextAgent(ContextAgent):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, AssistantMode | None]] = []

    def prepare(
        self,
        search_results: object,
        mode: AssistantMode | None = None,
    ) -> object:
        self.calls.append((search_results, mode))
        return super().prepare(search_results, mode=mode)


class FakeAnswerAgent:
    def __init__(self, response: str = "Ответ по найденным материалам") -> None:
        self.calls: list[object] = []
        self.build_calls: list[object] = []
        self._response = response

    def build_generation_request(
        self,
        *,
        mode: AssistantMode,
        user_prompt: str | None,
        context_decision: object,
    ) -> AnswerGenerationRequest:
        self.build_calls.append(
            {
                "mode": mode,
                "user_prompt": user_prompt,
                "context_decision": context_decision,
            }
        )
        return AnswerGenerationRequest(
            system_instruction="system",
            prompt=f"prompt {user_prompt}",
            max_output_tokens=1024,
            temperature=0.2,
        )

    async def answer(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self._response


class FakeDocumentSummaryAgent:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def summarize(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        raise AssertionError("DocumentSummaryAgent must not be called without retrieval")


class FakeRetrievalService:
    def __init__(
        self,
        results: tuple[SearchResult, ...] = (),
        error: Exception | None = None,
        metadata: DocumentMetadata | None = None,
        metadata_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[IntentDecision, str | None]] = []
        self.metadata_calls: list[str] = []
        self._results = results
        self._error = error
        self._metadata = metadata
        self._metadata_error = metadata_error

    async def search(
        self,
        *,
        intent_decision: IntentDecision,
        document_id: str | None,
    ) -> tuple[SearchResult, ...]:
        self.calls.append((intent_decision, document_id))
        if self._error is not None:
            raise self._error
        return self._results

    async def get_document_metadata(
        self,
        *,
        document_id: str,
    ) -> DocumentMetadata | None:
        self.metadata_calls.append(document_id)
        if self._metadata_error is not None:
            raise self._metadata_error
        return self._metadata


class FakeDocumentStorage:
    def __init__(
        self,
        data: bytes = b"%PDF-1.4",
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._data = data
        self._error = error

    async def download_pdf_for_summary(self, *, user_id: str, document_id: str) -> bytes:
        self.calls.append((user_id, document_id))
        if self._error is not None:
            raise self._error
        return self._data


class FakePdfSummaryService:
    def __init__(
        self,
        summary: str = "Краткое изложение PDF",
        error: Exception | None = None,
    ) -> None:
        self.calls: list[bytes] = []
        self._summary = summary
        self._error = error

    async def summarize_pdf(self, *, pdf_bytes: bytes) -> str:
        self.calls.append(pdf_bytes)
        if self._error is not None:
            raise self._error
        return self._summary


class FakeRedisState:
    def __init__(self) -> None:
        self.task_statuses: list[tuple[str, str]] = []
        self.clear_user_state_calls: list[str] = []

    async def set_task_status(self, *, user_id: str, status: str) -> None:
        self.task_statuses.append((user_id, status))

    async def clear_user_state(self, *, user_id: str) -> None:
        self.clear_user_state_calls.append(user_id)


def _prompt_message(
    *,
    prompt: str = "Объясни нормализацию баз данных",
    doc: str | None = None,
) -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": prompt,
            "doc": doc,
        }
    )


def _delete_message() -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "delete",
            "user_id": USER_ID,
        }
    )


def _orchestrator(
    *,
    publisher: FakePublisher | None = None,
    intent_agent: FakeIntentAgent | None = None,
    context_agent: RecordingContextAgent | None = None,
    answer_agent: FakeAnswerAgent | None = None,
    document_summary_agent: FakeDocumentSummaryAgent | None = None,
    retrieval_service: FakeRetrievalService | None = None,
    document_storage: FakeDocumentStorage | None = None,
    pdf_summary_service: FakePdfSummaryService | None = None,
    redis_state: FakeRedisState | None = None,
    max_user_prompt_chars: int = MAX_USER_PROMPT_CHARS,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
) -> TaskOrchestrator:
    return TaskOrchestrator(
        publisher=publisher or FakePublisher(),
        intent_agent=intent_agent or FakeIntentAgent(),
        context_agent=context_agent or RecordingContextAgent(),
        answer_agent=answer_agent or FakeAnswerAgent(),
        document_summary_agent=document_summary_agent or FakeDocumentSummaryAgent(),
        retrieval_service=retrieval_service or FakeRetrievalService(),
        document_storage=document_storage or FakeDocumentStorage(),
        pdf_summary_service=pdf_summary_service or FakePdfSummaryService(),
        redis_state=redis_state or FakeRedisState(),
        max_user_prompt_chars=max_user_prompt_chars,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
        max_prompt_chars=max_prompt_chars,
    )


def _search_result(
    *,
    text: str = "Нормализация уменьшает избыточность данных.",
) -> SearchResult:
    return SearchResult(
        doc_id=DOCUMENT_ID,
        chunk_id="chunk-1",
        file_name="lecture.pdf",
        page_number=5,
        text=text,
        score=12.5,
        highlights=(),
    )


def test_empty_retrieval_publishes_materials_not_found_without_answer_agent() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent()
    context_agent = RecordingContextAgent()
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    retrieval_service = FakeRetrievalService(results=())
    redis_state = FakeRedisState()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        retrieval_service=retrieval_service,
        redis_state=redis_state,
        max_prompt_chars=100,
    )

    asyncio.run(orchestrator.handle(_prompt_message(doc=DOCUMENT_ID)))

    assert [event.type for event in publisher.events] == [
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.RESPONSE,
    ]
    assert publisher.events[0].data == DETERMINE_INTENT_TEXT
    assert publisher.events[1].data == SEARCH_MATERIALS_TEXT
    assert isinstance(publisher.events[0].data, str)
    assert publisher.events[2].data == build_materials_not_found_response()
    assert publisher.events[2].warning > 0
    assert intent_agent.calls == [("Объясни нормализацию баз данных", DOCUMENT_ID)]
    assert len(retrieval_service.calls) == 1
    assert retrieval_service.calls[0][1] == DOCUMENT_ID
    assert len(context_agent.calls) == 1
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []
    assert redis_state.task_statuses[-1] == (USER_ID, "completed")
    assert {event.type.value for event in publisher.events} == {"think", "response"}


def test_legacy_mode_and_context_are_not_required_for_pipeline() -> None:
    publisher = FakePublisher()
    orchestrator = _orchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message()))

    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == build_materials_not_found_response()


def test_unsupported_intent_returns_deterministic_response_without_agents() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(task_type=IntentTaskType.UNSUPPORTED)
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    retrieval_service = FakeRetrievalService(results=(_search_result(),))
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        retrieval_service=retrieval_service,
    )

    asyncio.run(orchestrator.handle(_prompt_message(prompt="Создай сайт")))

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == build_unsupported_request_response()
    assert retrieval_service.calls == []
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []


def test_found_context_invokes_answer_agent_once_with_single_response() -> None:
    publisher = FakePublisher()
    context_agent = RecordingContextAgent()
    answer_agent = FakeAnswerAgent(response="Grounded answer")
    retrieval_service = FakeRetrievalService(results=(_search_result(),))
    orchestrator = _orchestrator(
        publisher=publisher,
        context_agent=context_agent,
        answer_agent=answer_agent,
        retrieval_service=retrieval_service,
    )

    asyncio.run(orchestrator.handle(_prompt_message(doc=DOCUMENT_ID)))

    assert [event.data for event in publisher.events if event.type == OutgoingEventType.THINK] == [
        DETERMINE_INTENT_TEXT,
        SEARCH_MATERIALS_TEXT,
        GENERATE_ANSWER_TEXT,
    ]
    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == "Grounded answer"
    assert len(context_agent.calls) == 1
    assert context_agent.calls[0][0] == (_search_result(),)
    assert context_agent.calls[0][1] == AssistantMode.EXPLAIN_TOPIC
    assert len(answer_agent.build_calls) == 1
    assert len(answer_agent.calls) == 1


def test_summary_intent_returns_full_document_fallback_without_summary_agent() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(task_type=IntentTaskType.SUMMARIZE_DOCUMENT)
    document_summary_agent = FakeDocumentSummaryAgent()
    retrieval_service = FakeRetrievalService(results=(_search_result(),))
    document_storage = FakeDocumentStorage()
    pdf_summary_service = FakePdfSummaryService()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        document_summary_agent=document_summary_agent,
        retrieval_service=retrieval_service,
        document_storage=document_storage,
        pdf_summary_service=pdf_summary_service,
    )

    asyncio.run(orchestrator.handle(_prompt_message(prompt="Сделай summary")))

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == build_summary_requires_complete_document_response()
    assert retrieval_service.calls == []
    assert retrieval_service.metadata_calls == []
    assert document_storage.calls == []
    assert pdf_summary_service.calls == []
    assert document_summary_agent.calls == []


def test_summary_missing_metadata_returns_controlled_fallback_without_minio_or_gemini() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(task_type=IntentTaskType.SUMMARIZE_DOCUMENT)
    retrieval_service = FakeRetrievalService(metadata=None)
    document_storage = FakeDocumentStorage()
    pdf_summary_service = FakePdfSummaryService()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        retrieval_service=retrieval_service,
        document_storage=document_storage,
        pdf_summary_service=pdf_summary_service,
    )

    asyncio.run(
        orchestrator.handle(_prompt_message(prompt="Сделай summary", doc=DOCUMENT_ID))
    )

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == build_document_summary_unavailable_response()
    assert retrieval_service.metadata_calls == [DOCUMENT_ID]
    assert document_storage.calls == []
    assert pdf_summary_service.calls == []


def test_summary_non_pdf_metadata_returns_pdf_only_fallback_without_minio_or_gemini() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(task_type=IntentTaskType.SUMMARIZE_DOCUMENT)
    retrieval_service = FakeRetrievalService(
        metadata=DocumentMetadata(doc_id=DOCUMENT_ID, file_name="lecture.docx")
    )
    document_storage = FakeDocumentStorage()
    pdf_summary_service = FakePdfSummaryService()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        retrieval_service=retrieval_service,
        document_storage=document_storage,
        pdf_summary_service=pdf_summary_service,
    )

    asyncio.run(
        orchestrator.handle(_prompt_message(prompt="Сделай summary", doc=DOCUMENT_ID))
    )

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == build_pdf_only_summary_response()
    assert retrieval_service.metadata_calls == [DOCUMENT_ID]
    assert document_storage.calls == []
    assert pdf_summary_service.calls == []


def test_summary_pdf_flow_returns_gemini_summary_with_single_response() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(task_type=IntentTaskType.SUMMARIZE_DOCUMENT)
    retrieval_service = FakeRetrievalService(
        metadata=DocumentMetadata(doc_id=DOCUMENT_ID, file_name="lecture.PDF")
    )
    document_storage = FakeDocumentStorage(data=b"%PDF-1.4 content")
    pdf_summary_service = FakePdfSummaryService(summary="Структурированное изложение")
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        retrieval_service=retrieval_service,
        document_storage=document_storage,
        pdf_summary_service=pdf_summary_service,
    )

    asyncio.run(
        orchestrator.handle(_prompt_message(prompt="Сделай summary", doc=DOCUMENT_ID))
    )

    assert [event.data for event in publisher.events if event.type == OutgoingEventType.THINK] == [
        DETERMINE_INTENT_TEXT,
        "Определяю выбранный документ.",
        PREPARE_DOCUMENT_TEXT,
        GENERATE_SUMMARY_TEXT,
    ]
    response_events = [
        event for event in publisher.events if event.type == OutgoingEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].data == "Структурированное изложение"
    assert retrieval_service.metadata_calls == [DOCUMENT_ID]
    assert document_storage.calls == [(USER_ID, DOCUMENT_ID)]
    assert pdf_summary_service.calls == [b"%PDF-1.4 content"]


def test_summary_minio_error_returns_controlled_fallback_without_requeue(caplog: pytest.LogCaptureFixture) -> None:
    from assistant_service.services.minio_storage import DocumentStorageUnavailableError

    publisher = FakePublisher()
    intent_agent = FakeIntentAgent(task_type=IntentTaskType.SUMMARIZE_DOCUMENT)
    retrieval_service = FakeRetrievalService(
        metadata=DocumentMetadata(doc_id=DOCUMENT_ID, file_name="lecture.pdf")
    )
    document_storage = FakeDocumentStorage(
        error=DocumentStorageUnavailableError("unavailable")
    )
    pdf_summary_service = FakePdfSummaryService()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        retrieval_service=retrieval_service,
        document_storage=document_storage,
        pdf_summary_service=pdf_summary_service,
    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            orchestrator.handle(_prompt_message(prompt="Сделай summary", doc=DOCUMENT_ID))
        )

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == build_document_summary_unavailable_response()
    assert document_storage.calls == [(USER_ID, DOCUMENT_ID)]
    assert pdf_summary_service.calls == []
    assert USER_ID not in caplog.text
    assert DOCUMENT_ID not in caplog.text
    assert "lecture.pdf" not in caplog.text
    assert "%PDF" not in caplog.text


def test_expected_elasticsearch_error_returns_retrieval_unavailable_response() -> None:
    publisher = FakePublisher()
    answer_agent = FakeAnswerAgent()
    retrieval_service = FakeRetrievalService(
        error=ElasticsearchUnavailableError("unavailable")
    )
    orchestrator = _orchestrator(
        publisher=publisher,
        answer_agent=answer_agent,
        retrieval_service=retrieval_service,
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == build_retrieval_unavailable_response()
    assert answer_agent.calls == []


def test_prompt_budget_boundary_returns_context_too_large_without_agents() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent()
    answer_agent = FakeAnswerAgent()
    prompt = "abc"
    max_prompt_chars = len("ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n\nКОНТЕКСТ:\n") + len(prompt)
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        answer_agent=answer_agent,
        max_user_prompt_chars=100,
        max_prompt_chars=max_prompt_chars,
    )

    asyncio.run(orchestrator.handle(_prompt_message(prompt=prompt)))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.RESPONSE
    assert publisher.events[0].data == build_context_too_large_response()
    assert publisher.events[0].warning == 100
    assert intent_agent.calls == []
    assert answer_agent.calls == []


def test_oversized_user_prompt_returns_response_without_agents() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent()
    answer_agent = FakeAnswerAgent()
    redis_state = FakeRedisState()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        answer_agent=answer_agent,
        redis_state=redis_state,
        max_user_prompt_chars=3,
        max_prompt_chars=100,
    )

    asyncio.run(orchestrator.handle(_prompt_message(prompt="слишком длинно")))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.RESPONSE
    assert publisher.events[0].data == build_request_too_large_response()
    assert publisher.events[0].warning > 0
    assert intent_agent.calls == []
    assert answer_agent.calls == []
    assert redis_state.task_statuses[-1] == (USER_ID, "completed")


def test_delete_publishes_only_think_confirmation_and_clears_user_state() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent()
    context_agent = RecordingContextAgent()
    answer_agent = FakeAnswerAgent()
    document_summary_agent = FakeDocumentSummaryAgent()
    redis_state = FakeRedisState()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=intent_agent,
        context_agent=context_agent,
        answer_agent=answer_agent,
        document_summary_agent=document_summary_agent,
        redis_state=redis_state,
    )

    asyncio.run(orchestrator.handle(_delete_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.THINK
    assert publisher.events[0].data == "История диалога очищена."
    assert intent_agent.calls == []
    assert context_agent.calls == []
    assert answer_agent.calls == []
    assert document_summary_agent.calls == []
    assert redis_state.clear_user_state_calls == [USER_ID]


def test_handled_processing_error_publishes_safe_response() -> None:
    publisher = FakePublisher()
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(fail=True),
    )

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == SAFE_ERROR_RESPONSE


def test_safe_response_publish_failure_is_reraised() -> None:
    publisher = FakePublisher(fail_on_response=True)
    orchestrator = _orchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(fail=True),
    )

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.handle(_prompt_message()))
