import logging
from typing import Protocol
from uuid import UUID

from assistant_service.core.enums import OutgoingEventType
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    IncomingMessage,
    OutgoingEvent,
    PromptRequestMessage,
    ResponseEvent,
    SyncEvent,
)

logger = logging.getLogger(__name__)

PROMPT_SYNC_TEXT = "Обрабатываем запрос..."
DELETE_SYNC_TEXT = "История диалога очищена."
TEMPORARY_ANSWER = (
    "Запрос принят. Ответ по материалам базы знаний будет сформирован после "
    "подключения модулей анализа контекста."
)
SAFE_ERROR_RESPONSE = "Не удалось обработать запрос. Попробуйте ещё раз."


class EventPublisher(Protocol):
    async def publish_event(self, event: OutgoingEvent) -> None:
        ...


class TaskOrchestrator:
    def __init__(self, publisher: EventPublisher) -> None:
        self._publisher = publisher

    async def handle(self, message: IncomingMessage) -> None:
        try:
            if isinstance(message, PromptRequestMessage):
                await self._handle_prompt(message)
                return

            if isinstance(message, DeleteRequestMessage):
                await self._handle_delete(message)
                return

        except Exception as exc:
            logger.exception(
                "Task orchestrator processing failed: user_id=%s message_type=%s",
                message.user_id,
                message.type,
            )
            try:
                await self._publish_response(
                    user_id=message.user_id,
                    data=SAFE_ERROR_RESPONSE,
                )
                return
            except Exception:
                logger.exception(
                    "Task orchestrator failed to publish safe response: "
                    "user_id=%s message_type=%s",
                    message.user_id,
                    message.type,
                )
                raise

        raise ValueError(f"Unsupported incoming message type: {message.type}")

    async def _handle_prompt(self, message: PromptRequestMessage) -> None:
        logger.info(
            "Task orchestrator received prompt: user_id=%s",
            message.user_id,
        )
        await self._publish_sync(user_id=message.user_id, data=PROMPT_SYNC_TEXT)
        await self._publish_response(user_id=message.user_id, data=TEMPORARY_ANSWER)

    async def _handle_delete(self, message: DeleteRequestMessage) -> None:
        logger.info(
            "Task orchestrator received delete: user_id=%s",
            message.user_id,
        )
        await self._publish_sync(user_id=message.user_id, data=DELETE_SYNC_TEXT)

    async def _publish_sync(self, user_id: UUID, data: str) -> None:
        await self._publisher.publish_event(
            SyncEvent(
                type=OutgoingEventType.SYNC,
                user_id=user_id,
                data=data,
            )
        )

    async def _publish_response(self, user_id: UUID, data: str) -> None:
        await self._publisher.publish_event(
            ResponseEvent(
                type=OutgoingEventType.RESPONSE,
                user_id=user_id,
                data=data,
                warning=0,
            )
        )
