import asyncio

import pytest

from assistant_service.core.enums import OutgoingEventType
from assistant_service.messaging.contracts import (
    IncomingMessage,
    OutgoingEvent,
    ResponseEvent,
    SyncEvent,
    incoming_message_adapter,
)
from assistant_service.services.task_orchestrator import (
    SAFE_ERROR_RESPONSE,
    TaskOrchestrator,
)


USER_ID = "00000000-0000-0000-0000-000000000002"
DOC_ID = "00000000-0000-0000-0000-000000000004"


class FakePublisher:
    def __init__(
        self,
        fail_on_sync: bool = False,
        fail_on_response: bool = False,
    ) -> None:
        self.events: list[OutgoingEvent] = []
        self._fail_on_sync = fail_on_sync
        self._fail_on_response = fail_on_response

    async def publish_event(self, event: OutgoingEvent) -> None:
        if self._fail_on_sync and isinstance(event, SyncEvent):
            raise RuntimeError("sync publish failed")

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


def test_prompt_publishes_sync_then_response() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert [event.type for event in publisher.events] == [
        OutgoingEventType.SYNC,
        OutgoingEventType.RESPONSE,
    ]
    assert publisher.events[0].data == "Обрабатываем запрос..."
    assert publisher.events[1].warning == 0


def test_delete_publishes_only_sync_confirmation() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_delete_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.SYNC
    assert publisher.events[0].data == "История диалога очищена."


def test_handled_processing_error_publishes_safe_response() -> None:
    publisher = FakePublisher(fail_on_sync=True)
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.RESPONSE
    assert publisher.events[0].data == SAFE_ERROR_RESPONSE


def test_safe_response_publish_failure_is_reraised() -> None:
    publisher = FakePublisher(fail_on_sync=True, fail_on_response=True)
    orchestrator = TaskOrchestrator(publisher=publisher)

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.handle(_prompt_message()))
