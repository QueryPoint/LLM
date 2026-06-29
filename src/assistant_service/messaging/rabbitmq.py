import asyncio
import json
import logging
from typing import Any

import aio_pika
from aio_pika import DeliveryMode, Message
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)
from pydantic import ValidationError

from assistant_service.core.config import Settings
from assistant_service.messaging.contracts import (
    IncomingMessage,
    OutgoingEvent,
    incoming_message_adapter,
    outgoing_event_adapter,
)

logger = logging.getLogger(__name__)


class RabbitMQPublisher:
    def __init__(
        self,
        channel: AbstractRobustChannel,
        default_routing_key: str,
    ) -> None:
        self._channel = channel
        self._default_routing_key = default_routing_key

    async def _publish_json(
        self,
        payload: dict[str, Any],
        routing_key: str | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        message = Message(
            body=body,
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
        )

        await self._channel.default_exchange.publish(
            message,
            routing_key=routing_key or self._default_routing_key,
        )

    async def publish_event(self, event: OutgoingEvent) -> None:
        payload = outgoing_event_adapter.dump_python(event, mode="json")
        await self._publish_json(payload)
        logger.info(
            "Outgoing event published: request_id=%s user_id=%s type=%s",
            event.request_id,
            event.user_id,
            event.type,
        )


class RabbitMQWorker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None
        self._queue: AbstractRobustQueue | None = None
        self._consumer_tag: str | None = None
        self.publisher: RabbitMQPublisher | None = None

    @property
    def is_ready(self) -> bool:
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
            and self._consumer_tag is not None
        )

    async def start(self) -> None:
        logger.info("RabbitMQ connection started")
        try:
            self._connection = await aio_pika.connect_robust(
                self._settings.rabbitmq_url,
            )

            logger.info("RabbitMQ connected")
            self._channel = await self._connection.channel()
            await self._channel.set_qos(
                prefetch_count=self._settings.rabbitmq_prefetch_count,
            )

            self._queue = await self._channel.declare_queue(
                self._settings.rabbitmq_back_to_llm_queue,
                durable=True,
            )
            await self._channel.declare_queue(
                self._settings.rabbitmq_llm_to_back_queue,
                durable=True,
            )

            self.publisher = RabbitMQPublisher(
                channel=self._channel,
                default_routing_key=self._settings.rabbitmq_llm_to_back_queue,
            )
            self._consumer_tag = await self._queue.consume(
                self._handle_message,
                no_ack=False,
            )
        except Exception:
            logger.exception("RabbitMQ startup failed")
            await self.stop()
            raise

        logger.info(
            "RabbitMQ consumer started: queue=%s",
            self._settings.rabbitmq_back_to_llm_queue,
        )

    async def stop(self) -> None:
        if self._queue is not None and self._consumer_tag is not None:
            await self._queue.cancel(self._consumer_tag)
            self._consumer_tag = None

        if self._channel is not None and not self._channel.is_closed:
            await self._channel.close()

        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()

        logger.info("RabbitMQ connection closed")

    async def _handle_message(self, message: AbstractIncomingMessage) -> None:
        logger.info("Incoming RabbitMQ message received")
        try:
            payload = json.loads(message.body.decode("utf-8"))
            request = incoming_message_adapter.validate_python(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            logger.exception("Invalid message rejected")
            await message.reject(requeue=False)
            return

        logger.info(
            "Incoming message accepted: request_id=%s user_id=%s type=%s",
            request.request_id,
            request.user_id,
            request.type,
        )

        try:
            await self._process_request(request)
        except Exception:
            logger.exception(
                "Unexpected handler error: request_id=%s user_id=%s",
                request.request_id,
                request.user_id,
            )
            await asyncio.sleep(self._settings.rabbitmq_requeue_delay_seconds)
            await message.nack(requeue=True)
            return

        await message.ack()

    async def _process_request(self, request: IncomingMessage) -> None:
        logger.info(
            "LLM request handled by stub: request_id=%s user_id=%s type=%s",
            request.request_id,
            request.user_id,
            request.type,
        )
