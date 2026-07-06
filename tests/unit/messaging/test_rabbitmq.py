import asyncio
import json

from aio_pika import Message

from assistant_service.core.config import Settings
from assistant_service.messaging.contracts import IncomingMessage
from assistant_service.messaging.rabbitmq import (
    PROCESSING_ATTEMPT_HEADER,
    RabbitMQWorker,
)


USER_ID = "00000000-0000-0000-0000-000000000002"


class FakeDefaultExchange:
    def __init__(self) -> None:
        self.published: list[tuple[Message, str]] = []

    async def publish(self, message: Message, routing_key: str) -> None:
        self.published.append((message, routing_key))


class FakeChannel:
    def __init__(self) -> None:
        self.default_exchange = FakeDefaultExchange()


class FakeIncomingMessage:
    def __init__(self, headers: dict[str, object] | None = None) -> None:
        self.body = json.dumps({"type": "delete", "user_id": USER_ID}).encode("utf-8")
        self.headers = headers
        self.content_type = "application/json"
        self.ack_calls = 0
        self.nack_calls: list[bool] = []
        self.reject_calls: list[bool] = []

    async def ack(self) -> None:
        self.ack_calls += 1

    async def nack(self, *, requeue: bool) -> None:
        self.nack_calls.append(requeue)

    async def reject(self, *, requeue: bool) -> None:
        self.reject_calls.append(requeue)


async def _failing_handler(message: IncomingMessage) -> None:
    raise RuntimeError("handler failed")


def test_worker_bounds_processing_retries_without_requeue_loop() -> None:
    settings = Settings()
    worker = RabbitMQWorker(
        settings=settings,
        message_handler=_failing_handler,
        max_processing_attempts=2,
    )
    worker._channel = FakeChannel()

    first_message = FakeIncomingMessage()
    asyncio.run(worker._handle_message(first_message))

    assert first_message.ack_calls == 1
    assert first_message.nack_calls == []
    assert first_message.reject_calls == []
    published_message, routing_key = worker._channel.default_exchange.published[0]
    assert routing_key == settings.rabbitmq_back_to_llm_queue
    assert published_message.body == first_message.body
    assert published_message.headers[PROCESSING_ATTEMPT_HEADER] == 2

    second_message = FakeIncomingMessage(headers={PROCESSING_ATTEMPT_HEADER: 2})
    asyncio.run(worker._handle_message(second_message))

    assert second_message.ack_calls == 0
    assert second_message.nack_calls == []
    assert second_message.reject_calls == [False]
    assert len(worker._channel.default_exchange.published) == 1
