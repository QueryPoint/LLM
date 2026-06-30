import asyncio
from uuid import UUID

import pytest

from assistant_service.core.enums import OutgoingEventType, ThinkStage
from assistant_service.messaging.contracts import (
    DoneEvent,
    IncomingMessage,
    OutgoingEvent,
    incoming_message_adapter,
)
from assistant_service.services.task_orchestrator import TaskOrchestrator


REQUEST_ID = "00000000-0000-0000-0000-000000000001"
USER_ID = "00000000-0000-0000-0000-000000000002"
SESSION_ID = "00000000-0000-0000-0000-000000000003"


class FakePublisher:
    def __init__(self, fail_on_done: bool = False) -> None:
        self.events: list[OutgoingEvent] = []
        self._fail_on_done = fail_on_done

    async def publish_event(self, event: OutgoingEvent) -> None:
        if self._fail_on_done and isinstance(event, DoneEvent):
            raise RuntimeError("publish failed")

        self.events.append(event)


def _prompt_message(request_id: str | None = REQUEST_ID) -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "request_id": request_id,
            "user_id": USER_ID,
            "session_id": SESSION_ID,
            "prompt": "Explain normalization",
            "doc": None,
        }
    )


def _delete_message() -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "delete",
            "request_id": REQUEST_ID,
            "user_id": USER_ID,
            "session_id": SESSION_ID,
        }
    )


def test_prompt_publishes_think_events_then_done() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message()))

    assert [event.type for event in publisher.events] == [
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.THINK,
        OutgoingEventType.DONE,
    ]
    assert [event.stage for event in publisher.events[:-1]] == [
        ThinkStage.RECEIVED,
        ThinkStage.VALIDATING_REQUEST,
        ThinkStage.DETECTING_INTENT,
        ThinkStage.CHECKING_CONTEXT,
    ]
    assert isinstance(publisher.events[-1], DoneEvent)
    assert publisher.events[-1].data.sources == []


def test_prompt_without_request_id_reuses_generated_id() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_prompt_message(request_id=None)))

    request_ids = {event.request_id for event in publisher.events}
    assert len(request_ids) == 1
    assert isinstance(next(iter(request_ids)), UUID)


def test_delete_publishes_clearing_session_only() -> None:
    publisher = FakePublisher()
    orchestrator = TaskOrchestrator(publisher=publisher)

    asyncio.run(orchestrator.handle(_delete_message()))

    assert len(publisher.events) == 1
    assert publisher.events[0].type == OutgoingEventType.THINK
    assert publisher.events[0].stage == ThinkStage.CLEARING_SESSION


def test_processing_error_publishes_error_and_reraises() -> None:
    publisher = FakePublisher(fail_on_done=True)
    orchestrator = TaskOrchestrator(publisher=publisher)

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.handle(_prompt_message()))

    assert publisher.events[-1].type == OutgoingEventType.ERROR
    assert publisher.events[-1].error_code == "TASK_PROCESSING_ERROR"
