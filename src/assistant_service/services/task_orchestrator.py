import logging
from typing import Protocol
from uuid import UUID

from assistant_service.agents.intent_agent import IntentAgent
from assistant_service.core.enums import LLMStatus, OutgoingEventType
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    IncomingMessage,
    OutgoingEvent,
    PromptRequestMessage,
    ResponseEvent,
    ThinkEvent,
)

logger = logging.getLogger(__name__)

STATUS_MESSAGES = {
    LLMStatus.QUEUED: "Запрос добавлен в очередь...",
    LLMStatus.PROCESSING: "Приняли запрос в обработку...",
    LLMStatus.THINKING: "Анализируем запрос...",
    LLMStatus.SEARCHING: "Ищем релевантные материалы...",
    LLMStatus.FOUND: "Определили тип запроса...",
    LLMStatus.GENERATING: "Готовим дальнейшую обработку...",
}
DELETE_THINK_TEXT = "История диалога очищена."
TEMPORARY_ANSWER = (
    "Запрос принят. Ответ по материалам базы знаний будет сформирован после "
    "подключения модулей анализа контекста."
)
SAFE_ERROR_RESPONSE = "Не удалось обработать запрос. Попробуйте ещё раз."


class EventPublisher(Protocol):
    async def publish_event(self, event: OutgoingEvent) -> None:
        ...


class TaskOrchestrator:
    def __init__(self, publisher: EventPublisher, intent_agent: IntentAgent) -> None:
        self._publisher = publisher
        self._intent_agent = intent_agent

    async def handle(self, message: IncomingMessage) -> None:
        try:
            if isinstance(message, PromptRequestMessage):
                await self._handle_prompt(message)
                return

            if isinstance(message, DeleteRequestMessage):
                await self._handle_delete(message)
                return

        except Exception:
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
        await self._publish_prompt_status(message.user_id, LLMStatus.PROCESSING)
        await self._publish_prompt_status(message.user_id, LLMStatus.THINKING)

        intent_decision = self._intent_agent.detect(
            prompt=message.prompt,
            document_id=message.doc,
            requested_mode=message.mode,
        )
        logger.info(
            "Intent detected: user_id=%s selected_mode=%s "
            "selection_source=%s has_document=%s",
            message.user_id,
            intent_decision.mode.value,
            intent_decision.source,
            message.doc is not None,
        )

        await self._publish_prompt_status(message.user_id, LLMStatus.FOUND)
        await self._publish_prompt_status(message.user_id, LLMStatus.GENERATING)
        await self._publish_response(user_id=message.user_id, data=TEMPORARY_ANSWER)

    async def _handle_delete(self, message: DeleteRequestMessage) -> None:
        logger.info(
            "Task orchestrator received delete: user_id=%s",
            message.user_id,
        )
        await self._publish_think(user_id=message.user_id, data=DELETE_THINK_TEXT)

    async def _publish_think(
        self,
        user_id: UUID,
        data: str,
        status: LLMStatus | None = None,
    ) -> None:
        await self._publisher.publish_event(
            ThinkEvent(
                type=OutgoingEventType.THINK,
                user_id=user_id,
                data=data,
            )
        )
        logger.info(
            "Task orchestrator published think event: user_id=%s status=%s",
            user_id,
            status.value if status is not None else None,
        )

    async def _publish_prompt_status(self, user_id: UUID, status: LLMStatus) -> None:
        await self._publish_think(
            user_id=user_id,
            data=STATUS_MESSAGES[status],
            status=status,
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
