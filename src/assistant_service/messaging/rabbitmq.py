import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
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

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]
PROCESSING_ATTEMPT_HEADER = "x-assistant-processing-attempt"


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
            "Outgoing event published: type=%s",
            event.type,
        )


class RabbitMQWorker:
    def __init__(
        self,
        settings: Settings,
        message_handler: MessageHandler | None = None,
        max_processing_attempts: int | None = None,
    ) -> None:
        resolved_max_processing_attempts = (
            settings.rabbitmq_max_processing_attempts
            if max_processing_attempts is None
            else max_processing_attempts
        )
        if resolved_max_processing_attempts <= 0:
            raise ValueError("max_processing_attempts must be positive")

        self._settings = settings
        self._message_handler = message_handler
        self._max_processing_attempts = resolved_max_processing_attempts
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None
        self._queue: AbstractRobustQueue | None = None
        self._consumer_tag: str | None = None
        self.publisher: RabbitMQPublisher | None = None

    def set_message_handler(self, message_handler: MessageHandler) -> None:
        self._message_handler = message_handler

    async def publish_event(self, event: OutgoingEvent) -> None:
        if self.publisher is None:
            raise RuntimeError("RabbitMQ publisher is not ready")

        await self.publisher.publish_event(event)

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
        if self._message_handler is None:
            raise RuntimeError("RabbitMQ message handler is not configured")

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
            "Incoming message accepted: type=%s",
            request.type,
        )

        try:
            await self._process_request(request)
        except Exception as exc:
            logger.exception("Unexpected handler error: type=%s", request.type)
            attempt = self._get_processing_attempt(message)
            if attempt < self._max_processing_attempts:
                try:
                    await self._republish_for_processing_retry(
                        message=message,
                        next_attempt=attempt + 1,
                    )
                    await message.ack()
                    return
                except Exception as republish_exc:
                    logger.exception(
                        "RabbitMQ retry republish failed: queue=%s attempt=%s "
                        "error_type=%s",
                        self._settings.rabbitmq_back_to_llm_queue,
                        attempt,
                        type(republish_exc).__name__,
                    )
                    await asyncio.sleep(self._settings.rabbitmq_requeue_delay_seconds)
                    await message.nack(requeue=True)
                    return

            logger.error(
                "RabbitMQ message rejected after processing attempts: queue=%s "
                "attempt=%s error_type=%s",
                self._settings.rabbitmq_back_to_llm_queue,
                attempt,
                type(exc).__name__,
            )
            await message.reject(requeue=False)
            return

        await message.ack()

    async def _process_request(self, request: IncomingMessage) -> None:
        if self._message_handler is None:
            raise RuntimeError("RabbitMQ message handler is not configured")

        await self._message_handler(request)

    async def _republish_for_processing_retry(
        self,
        *,
        message: AbstractIncomingMessage,
        next_attempt: int,
    ) -> None:
        if self._channel is None:
            raise RuntimeError("RabbitMQ channel is not ready")

        headers = dict(message.headers or {})
        headers[PROCESSING_ATTEMPT_HEADER] = next_attempt
        retry_message = Message(
            body=message.body,
            content_type=message.content_type,
            delivery_mode=DeliveryMode.PERSISTENT,
            headers=headers,
        )
        await self._channel.default_exchange.publish(
            retry_message,
            routing_key=self._settings.rabbitmq_back_to_llm_queue,
        )

    @staticmethod
    def _get_processing_attempt(message: AbstractIncomingMessage) -> int:
        headers = message.headers or {}
        raw_attempt = headers.get(PROCESSING_ATTEMPT_HEADER)
        if isinstance(raw_attempt, int) and raw_attempt > 0:
            return raw_attempt
        if isinstance(raw_attempt, str):
            try:
                attempt = int(raw_attempt)
            except ValueError:
                return 1
            return attempt if attempt > 0 else 1
        return 1
