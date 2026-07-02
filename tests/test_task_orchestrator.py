import asyncio

import pytest

from assistant_service.agents.intent_agent import IntentDecision
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
    build_materials_not_found_response,
    build_request_too_large_response,
)
from assistant_service.services.task_orchestrator import (
    DETERMINE_INTENT_TEXT,
    SAFE_ERROR_RESPONSE,
    TaskOrchestrator,
)


USER_ID = "user-123"
MAX_USER_PROMPT_CHARS = 4_000
MAX_CHUNK_CHARS = 12_000
MAX_CHUNKS_PER_REQUEST = 32
MAX_PROMPT_CHARS = 16_000


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
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._fail = fail

    def detect(self, prompt: str, uid: str | None) -> IntentDecision:
        self.calls.append((prompt, uid))
        if self._fail:
            raise RuntimeError("intent detection failed")
        return IntentDecision(
            mode=AssistantMode.EXPLAIN_TOPIC,
            source="rule_based",
        )


class FakeContextAgent:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def prepare(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        raise AssertionError("ContextAgent must not be called without retrieval")


class FakeAnswerAgent:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def answer(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        raise AssertionError("AnswerAgent must not be called without retrieval")


class FakeDocumentSummaryAgent:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def summarize(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        raise AssertionError("DocumentSummaryAgent must not be called without retrieval")


class FakeRedisState:
    def __init__(self, cached_mode: AssistantMode | None = None) -> None:
        self.cached_mode = cached_mode
        self.intent_get_keys: list[str] = []
        self.intent_set_calls: list[tuple[str, AssistantMode]] = []
        self.task_statuses: list[tuple[str, str]] = []
        self.clear_user_state_calls: list[str] = []

    async def get_intent_mode(self, key: str) -> AssistantMode | None:
        self.intent_get_keys.append(key)
        return self.cached_mode

    async def set_intent_mode(self, key: str, mode: AssistantMode) -> None:
        self.intent_set_calls.append((key, mode))

    async def set_task_status(self, *, user_id: str, status: str) -> None:
        self.task_statuses.append((user_id, status))

    async def clear_user_state(self, *, user_id: str) -> None:
        self.clear_user_state_calls.append(user_id)


def _prompt_message(
    *,
    prompt: str = "Объясни нормализацию баз данных",
    uid: str | None = None,
) -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": prompt,
            "uid": uid,
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
    context_agent: FakeContextAgent | None = None,
    answer_agent: FakeAnswerAgent | None = None,
    document_summary_agent: FakeDocumentSummaryAgent | None = None,
    redis_state: FakeRedisState | None = None,
    max_user_prompt_chars: int = MAX_USER_PROMPT_CHARS,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
) -> TaskOrchestrator:
    return TaskOrchestrator(
        publisher=publisher or FakePublisher(),
        intent_agent=intent_agent or FakeIntentAgent(),
        context_agent=context_agent or FakeContextAgent(),
        answer_agent=answer_agent or FakeAnswerAgent(),
        document_summary_agent=document_summary_agent or FakeDocumentSummaryAgent(),
        redis_state=redis_state or FakeRedisState(),
        max_user_prompt_chars=max_user_prompt_chars,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
        max_prompt_chars=max_prompt_chars,
    )


def test_prompt_without_retrieval_publishes_think_then_controlled_fallback() -> None:
    publisher = FakePublisher()
    intent_agent = FakeIntentAgent()
    context_agent = FakeContextAgent()
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
        max_prompt_chars=100,
    )

    asyncio.run(orchestrator.handle(_prompt_message(uid="document-uid-123")))

    assert [event.type for event in publisher.events] == [
        OutgoingEventType.THINK,
        OutgoingEventType.RESPONSE,
    ]
    assert publisher.events[0].data == DETERMINE_INTENT_TEXT
    assert isinstance(publisher.events[0].data, str)
    assert publisher.events[1].data == build_materials_not_found_response()
    assert publisher.events[1].warning > 0
    assert intent_agent.calls == [
        ("Объясни нормализацию баз данных", "document-uid-123")
    ]
    assert context_agent.calls == []
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
    context_agent = FakeContextAgent()
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
