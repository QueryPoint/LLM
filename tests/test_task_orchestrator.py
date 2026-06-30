import asyncio

import pytest

from assistant_service.core.enums import LLMStatus, OutgoingEventType
from assistant_service.messaging.contracts import (
    IncomingMessage,
    OutgoingEvent,
    ResponseEvent,
    ThinkEvent,
    incoming_message_adapter,
)
from assistant_service.services.task_orchestrator import (
    SAFE_ERROR_RESPONSE,
    STATUS_MESSAGES,
    TaskOrchestrator,
)


USER_ID = "00000000-0000-0000-0000-000000000002"
DOC_ID = "00000000-0000-0000-0000-000000000004"


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


def _prompt_message() -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Explain normalization",
            "doc": DOC_ID,
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
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert [event.type for event in publisher.events] == [
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.RESPONSE,
    ]
    assert [event.data for event in publisher.events[:-1]] == [
        STATUS_MESSAGES[LLMStatus.PROCESSING],
        STATUS_MESSAGES[LLMStatus.THINKING],
        STATUS_MESSAGES[LLMStatus.SEARCHING],
        STATUS_MESSAGES[LLMStatus.GENERATING],
    ]
    assert publisher.events[-1].warning == 0


def test_delete_publishes_only_think_confirmation() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_delete_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.THINK
    assert publisher.events[0].data == "История диалога очищена."


def test_handled_processing_error_publishes_safe_response() -> None:
    publisher = FakePublisher(fail_on_think=True)
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.RESPONSE
    assert publisher.events[0].data == SAFE_ERROR_RESPONSE


def test_safe_response_publish_failure_is_reraised() -> None:
    publisher = FakePublisher(fail_on_think=True, fail_on_response=True)
    orchestrator = TaskOrchestrator(publisher=publisher)

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.handle(_prompt_message()))
