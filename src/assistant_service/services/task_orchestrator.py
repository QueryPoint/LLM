import logging
from typing import Protocol
from uuid import UUID, uuid4

from assistant_service.core.enums import (
    OutgoingEventType,
    ThinkStage,
)
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    DoneEvent,
    DoneEventData,
    ErrorEvent,
    IncomingMessage,
    OutgoingEvent,
    PromptRequestMessage,
    ThinkEvent,
)

logger = logging.getLogger(__name__)

PROMPT_THINK_STEPS = (
    (ThinkStage.RECEIVED, "Приняли запрос в обработку..."),
    (ThinkStage.VALIDATING_REQUEST, "Проверяем параметры запроса..."),
    (ThinkStage.DETECTING_INTENT, "Определяем тип запроса..."),
    (ThinkStage.CHECKING_CONTEXT, "Проверяем найденные материалы..."),
)

DELETE_THINK_TEXT = "Очищаем временный контекст..."
TEMPORARY_ANSWER = (
    "Запрос принят. Ответ по материалам базы знаний будет сформирован после "
    "подключения модулей анализа контекста."
)
TASK_PROCESSING_ERROR_TEXT = (
    "Не удалось обработать запрос ассистента. Попробуйте ещё раз."
)
TASK_PROCESSING_ERROR_CODE = "TASK_PROCESSING_ERROR"


class EventPublisher(Protocol):
    async def publish_event(self, event: OutgoingEvent) -> None:
        ...


class TaskOrchestrator:
    def __init__(self, publisher: EventPublisher) -> None:
        self._publisher = publisher

    async def handle(self, message: IncomingMessage) -> None:
        request_id = self._resolve_request_id(message)

        try:
            if isinstance(message, PromptRequestMessage):
                await self._handle_prompt(message=message, request_id=request_id)
                return

            if isinstance(message, DeleteRequestMessage):
                await self._handle_delete(message=message, request_id=request_id)
                return

        except Exception as exc:
            logger.exception(
                "Task orchestrator processing failed: request_id=%s user_id=%s "
                "message_type=%s",
                request_id,
                message.user_id,
                message.type,
            )
            try:
                await self._publish_error(
                    request_id=request_id,
                    user_id=message.user_id,
                )
            except Exception:
                logger.exception(
                    "Task orchestrator failed to publish error event: "
                    "request_id=%s user_id=%s message_type=%s",
                    request_id,
                    message.user_id,
                    message.type,
                )
            raise exc

        raise ValueError(f"Unsupported incoming message type: {message.type}")

    def _resolve_request_id(self, message: IncomingMessage) -> UUID:
        if message.request_id is not None:
            return message.request_id

        request_id = uuid4()
        logger.info(
            "Task orchestrator generated request id: request_id=%s user_id=%s "
            "message_type=%s",
            request_id,
            message.user_id,
            message.type,
        )
        return request_id

    async def _handle_prompt(
        self,
        message: PromptRequestMessage,
        request_id: UUID,
    ) -> None:
        logger.info(
            "Task orchestrator received prompt: request_id=%s user_id=%s "
            "session_id=%s",
            request_id,
            message.user_id,
            message.session_id,
        )

        for stage, text in PROMPT_THINK_STEPS:
            await self._publish_think(
                request_id=request_id,
                user_id=message.user_id,
                stage=stage,
                data=text,
            )

        await self._publisher.publish_event(
            DoneEvent(
                request_id=request_id,
                user_id=message.user_id,
                type=OutgoingEventType.DONE,
                data=DoneEventData(
                    answer=TEMPORARY_ANSWER,
                    sources=[],
                ),
                warning=0,
            )
        )
        logger.info(
            "Task orchestrator published temporary done event: request_id=%s "
            "user_id=%s",
            request_id,
            message.user_id,
        )

    async def _handle_delete(
        self,
        message: DeleteRequestMessage,
        request_id: UUID,
    ) -> None:
        logger.info(
            "Task orchestrator received delete: request_id=%s user_id=%s "
            "session_id=%s",
            request_id,
            message.user_id,
            message.session_id,
        )
        await self._publish_think(
            request_id=request_id,
            user_id=message.user_id,
            stage=ThinkStage.CLEARING_SESSION,
            data=DELETE_THINK_TEXT,
        )

    async def _publish_think(
        self,
        request_id: UUID,
        user_id: UUID,
        stage: ThinkStage,
        data: str,
    ) -> None:
        await self._publisher.publish_event(
            ThinkEvent(
                request_id=request_id,
                user_id=user_id,
                type=OutgoingEventType.THINK,
                stage=stage,
                data=data,
                warning=0,
            )
        )
        logger.info(
            "Task orchestrator published think event: request_id=%s user_id=%s "
            "stage=%s",
            request_id,
            user_id,
            stage.value,
        )

    async def _publish_error(self, request_id: UUID, user_id: UUID) -> None:
        await self._publisher.publish_event(
            ErrorEvent(
                request_id=request_id,
                user_id=user_id,
                type=OutgoingEventType.ERROR,
                data=TASK_PROCESSING_ERROR_TEXT,
                error_code=TASK_PROCESSING_ERROR_CODE,
                warning=0,
            )
        )
