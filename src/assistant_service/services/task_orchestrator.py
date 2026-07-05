import logging
from typing import Protocol

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextAgent
from assistant_service.agents.document_summary_agent import DocumentSummaryAgent
from assistant_service.agents.intent_agent import (
    IntentAgent,
    IntentDecision,
    IntentTaskType,
)
from assistant_service.core.enums import (
    AssistantMode,
    LLMStatus,
    OutgoingEventType,
    RetrievalStatus,
)
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    IncomingMessage,
    OutgoingEvent,
    PromptRequestMessage,
    ResponseEvent,
    ThinkEvent,
)
from assistant_service.services.prompt_budget import (
    calculate_prompt_budget_warning,
    prompt_budget_exceeded,
)
from assistant_service.services.response_builder import (
    build_context_too_large_response,
    build_document_summary_unavailable_response,
    build_gemini_rate_limit_response,
    build_gemini_unavailable_response,
    build_materials_not_found_response,
    build_pdf_only_summary_response,
    build_request_too_large_response,
    build_response_text,
    build_retrieval_unavailable_response,
    build_summary_requires_complete_document_response,
    build_unsupported_request_response,
)
from assistant_service.services.elasticsearch_client import (
    ElasticsearchIndexNotFoundError,
    ElasticsearchResponseError,
    ElasticsearchUnavailableError,
)
from assistant_service.services.gemini_client import (
    GeminiPermanentError,
    GeminiRateLimitError,
    GeminiRequestLimitError,
    GeminiTransientError,
)
from assistant_service.services.minio_storage import (
    DocumentStorageError,
    MinioDocumentStorageClient,
)
from assistant_service.services.pdf_document_summary import PdfDocumentSummaryService
from assistant_service.services.redis_state import RedisStateStore
from assistant_service.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

DETERMINE_INTENT_TEXT = "Определяю тип запроса..."
SEARCH_MATERIALS_TEXT = "Ищу подходящие материалы..."
GENERATE_ANSWER_TEXT = "Формирую ответ..."
DETERMINE_DOCUMENT_TEXT = "Определяю выбранный документ."
PREPARE_DOCUMENT_TEXT = "Подготавливаю документ к обработке."
GENERATE_SUMMARY_TEXT = "Формирую краткое изложение."
DELETE_THINK_TEXT = "История диалога очищена."
SAFE_ERROR_RESPONSE = "Не удалось обработать запрос. Попробуйте ещё раз."
NO_RETRIEVAL_PROMPT_MARKUP = "ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n\nКОНТЕКСТ:\n"
EXPECTED_RETRIEVAL_ERRORS = (
    ElasticsearchUnavailableError,
    ElasticsearchIndexNotFoundError,
    ElasticsearchResponseError,
)


class EventPublisher(Protocol):
    async def publish_event(self, event: OutgoingEvent) -> None:
        ...


class TaskOrchestrator:
    def __init__(
        self,
        publisher: EventPublisher,
        intent_agent: IntentAgent,
        context_agent: ContextAgent,
        answer_agent: AnswerAgent,
        document_summary_agent: DocumentSummaryAgent,
        retrieval_service: RetrievalService,
        document_storage: MinioDocumentStorageClient,
        pdf_summary_service: PdfDocumentSummaryService,
        redis_state: RedisStateStore,
        max_user_prompt_chars: int,
        max_chunk_chars: int,
        max_chunks_per_request: int,
        max_prompt_chars: int,
    ) -> None:
        if max_user_prompt_chars <= 0:
            raise ValueError("max_user_prompt_chars must be positive")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")
        if max_chunks_per_request <= 0:
            raise ValueError("max_chunks_per_request must be positive")
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")

        self._publisher = publisher
        self._intent_agent = intent_agent
        self._context_agent = context_agent
        self._answer_agent = answer_agent
        self._document_summary_agent = document_summary_agent
        self._retrieval_service = retrieval_service
        self._document_storage = document_storage
        self._pdf_summary_service = pdf_summary_service
        self._redis_state = redis_state
        self._max_user_prompt_chars = max_user_prompt_chars
        self._max_chunk_chars = max_chunk_chars
        self._max_chunks_per_request = max_chunks_per_request
        self._max_prompt_chars = max_prompt_chars

    async def handle(self, message: IncomingMessage) -> None:
        try:
            if isinstance(message, PromptRequestMessage):
                await self._handle_prompt(message)
                return

            if isinstance(message, DeleteRequestMessage):
                await self._handle_delete(message)
                return

        except Exception:
            logger.exception("Task orchestrator processing failed")
            await self._redis_state.set_task_status(
                user_id=message.user_id,
                status="failed",
            )
            try:
                await self._publish_response(
                    user_id=message.user_id,
                    data=SAFE_ERROR_RESPONSE,
                    mark_completed=False,
                )
                return
            except Exception:
                logger.exception("Task orchestrator failed to publish safe response")
                raise

        raise ValueError(f"Unsupported incoming message type: {message.type}")

    async def _handle_prompt(self, message: PromptRequestMessage) -> None:
        document_id = message.doc
        warning = self._calculate_warning(user_prompt=message.prompt)
        if self._is_user_prompt_too_large(message.prompt):
            logger.info("Request rejected by local limit")
            await self._publish_response(
                user_id=message.user_id,
                data=build_request_too_large_response(),
                warning=warning,
            )
            return

        if self._prompt_budget_exceeded(user_prompt=message.prompt):
            logger.info("Request rejected by budget limit")
            await self._publish_response(
                user_id=message.user_id,
                data=build_context_too_large_response(),
                warning=warning,
            )
            return

        logger.info(
            "Task orchestrator received request: has_document=%s",
            document_id is not None,
        )
        await self._redis_state.set_task_status(
            user_id=message.user_id,
            status=LLMStatus.PROCESSING.value.lower(),
        )
        await self._publish_think(
            user_id=message.user_id,
            data=DETERMINE_INTENT_TEXT,
            status=LLMStatus.THINKING,
        )

        intent_decision = await self._detect_intent(message)
        logger.info(
            "Intent detected: task_type=%s selection_source=%s has_document=%s",
            intent_decision.task_type.value,
            intent_decision.source,
            document_id is not None,
        )
        if intent_decision.task_type == IntentTaskType.UNSUPPORTED:
            await self._publish_response(
                user_id=message.user_id,
                data=build_unsupported_request_response(),
                warning=warning,
            )
            return

        if intent_decision.task_type == IntentTaskType.SUMMARIZE_DOCUMENT:
            await self._handle_summary_document(message=message, warning=warning)
            return

        if not intent_decision.keywords:
            await self._redis_state.set_task_status(
                user_id=message.user_id,
                status="not_found",
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_materials_not_found_response(),
                warning=warning,
            )
            return

        await self._publish_think(
            user_id=message.user_id,
            data=SEARCH_MATERIALS_TEXT,
            status=LLMStatus.SEARCHING,
        )
        try:
            search_results = await self._retrieval_service.search(
                intent_decision=intent_decision,
                document_id=document_id,
            )
        except EXPECTED_RETRIEVAL_ERRORS as exc:
            logger.warning(
                "Retrieval failed with expected error: task_type=%s error_type=%s",
                intent_decision.task_type.value,
                type(exc).__name__,
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_retrieval_unavailable_response(),
                warning=warning,
            )
            return

        assistant_mode = self._assistant_mode_for_task(intent_decision.task_type)
        context_decision = self._context_agent.prepare(
            search_results,
            mode=assistant_mode,
        )
        if context_decision.status == RetrievalStatus.FOUND:
            await self._redis_state.set_task_status(
                user_id=message.user_id,
                status=LLMStatus.FOUND.value.lower(),
            )

        if context_decision.status != RetrievalStatus.FOUND:
            await self._redis_state.set_task_status(
                user_id=message.user_id,
                status=context_decision.status.value,
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_materials_not_found_response(),
                warning=warning,
            )
            return

        if self._context_exceeds_limits(context_decision):
            await self._publish_response(
                user_id=message.user_id,
                data=build_context_too_large_response(),
                warning=warning,
            )
            return

        if assistant_mode == AssistantMode.DOCUMENT_SEARCH:
            await self._publish_response(
                user_id=message.user_id,
                data=build_response_text(assistant_mode, context_decision),
                warning=warning,
            )
            return

        generation_request = self._answer_agent.build_generation_request(
            mode=assistant_mode,
            user_prompt=message.prompt,
            context_decision=context_decision,
        )
        generation_warning = self._calculate_generation_warning(
            system_instruction=generation_request.system_instruction,
            prompt=generation_request.prompt,
        )
        if self._generation_prompt_budget_exceeded(
            system_instruction=generation_request.system_instruction,
            prompt=generation_request.prompt,
        ):
            await self._publish_response(
                user_id=message.user_id,
                data=build_context_too_large_response(),
                warning=generation_warning,
            )
            return

        await self._publish_think(
            user_id=message.user_id,
            data=GENERATE_ANSWER_TEXT,
            status=LLMStatus.GENERATING,
        )
        try:
            answer_text = await self._answer_agent.answer(
                mode=assistant_mode,
                user_prompt=message.prompt,
                context_decision=context_decision,
            )
        except GeminiRequestLimitError:
            await self._publish_response(
                user_id=message.user_id,
                data=build_context_too_large_response(),
                warning=generation_warning,
            )
            return
        except GeminiRateLimitError:
            await self._publish_response(
                user_id=message.user_id,
                data=build_gemini_rate_limit_response(),
                warning=generation_warning,
            )
            return
        except (GeminiTransientError, GeminiPermanentError):
            await self._publish_response(
                user_id=message.user_id,
                data=build_gemini_unavailable_response(),
                warning=generation_warning,
            )
            return

        await self._publish_response(
            user_id=message.user_id,
            data=answer_text,
            warning=generation_warning,
        )

    async def _handle_summary_document(
        self,
        *,
        message: PromptRequestMessage,
        warning: int,
    ) -> None:
        document_id = message.doc
        if document_id is None:
            await self._publish_response(
                user_id=message.user_id,
                data=build_summary_requires_complete_document_response(),
                warning=warning,
            )
            return

        await self._publish_think(
            user_id=message.user_id,
            data=DETERMINE_DOCUMENT_TEXT,
            status=LLMStatus.SEARCHING,
        )
        try:
            metadata = await self._retrieval_service.get_document_metadata(
                document_id=document_id,
            )
        except EXPECTED_RETRIEVAL_ERRORS as exc:
            logger.warning(
                "Summary metadata lookup failed: task_type=%s error_type=%s",
                IntentTaskType.SUMMARIZE_DOCUMENT.value,
                type(exc).__name__,
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_document_summary_unavailable_response(),
                warning=warning,
            )
            return

        if metadata is None or metadata.file_name.strip() == "":
            await self._publish_response(
                user_id=message.user_id,
                data=build_document_summary_unavailable_response(),
                warning=warning,
            )
            return

        if self._document_extension(metadata.file_name) != "pdf":
            await self._publish_response(
                user_id=message.user_id,
                data=build_pdf_only_summary_response(),
                warning=warning,
            )
            return

        await self._publish_think(
            user_id=message.user_id,
            data=PREPARE_DOCUMENT_TEXT,
            status=LLMStatus.SEARCHING,
        )
        try:
            pdf_bytes = await self._document_storage.download_pdf_for_summary(
                user_id=message.user_id,
                document_id=document_id,
            )
        except DocumentStorageError as exc:
            logger.warning(
                "Summary document download failed: operation=%s format=%s error_type=%s",
                "download",
                "pdf",
                type(exc).__name__,
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_document_summary_unavailable_response(),
                warning=warning,
            )
            return

        await self._publish_think(
            user_id=message.user_id,
            data=GENERATE_SUMMARY_TEXT,
            status=LLMStatus.GENERATING,
        )
        try:
            summary_text = await self._pdf_summary_service.summarize_pdf(
                pdf_bytes=pdf_bytes,
            )
        except GeminiRateLimitError:
            await self._publish_response(
                user_id=message.user_id,
                data=build_gemini_rate_limit_response(),
                warning=warning,
            )
            return
        except (GeminiRequestLimitError, GeminiTransientError, GeminiPermanentError) as exc:
            logger.warning(
                "Summary generation failed: operation=%s format=%s error_type=%s",
                "files_api_summary",
                "pdf",
                type(exc).__name__,
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_document_summary_unavailable_response(),
                warning=warning,
            )
            return

        await self._publish_response(
            user_id=message.user_id,
            data=summary_text,
            warning=warning,
        )

    async def _handle_delete(self, message: DeleteRequestMessage) -> None:
        logger.info(
            "Task orchestrator received delete",
        )
        await self._redis_state.clear_user_state(user_id=message.user_id)
        await self._publish_think(user_id=message.user_id, data=DELETE_THINK_TEXT)

    async def _detect_intent(self, message: PromptRequestMessage) -> IntentDecision:
        return await self._intent_agent.detect(
            prompt=message.prompt,
            document_id=message.doc,
        )

    async def _publish_think(
        self,
        user_id: str,
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
            "Task orchestrator published think event: status=%s",
            status.value if status is not None else None,
        )
        if status is not None:
            await self._redis_state.set_task_status(
                user_id=user_id,
                status=status.value.lower(),
            )

    def _is_user_prompt_too_large(self, prompt: str) -> bool:
        return len(prompt) > self._max_user_prompt_chars

    def _calculate_warning(self, *, user_prompt: str) -> int:
        return calculate_prompt_budget_warning(
            max_prompt_chars=self._max_prompt_chars,
            user_prompt=user_prompt,
            prompt_markup=NO_RETRIEVAL_PROMPT_MARKUP,
        )

    def _prompt_budget_exceeded(self, *, user_prompt: str) -> bool:
        return prompt_budget_exceeded(
            max_prompt_chars=self._max_prompt_chars,
            user_prompt=user_prompt,
            prompt_markup=NO_RETRIEVAL_PROMPT_MARKUP,
        )

    def _calculate_generation_warning(
        self,
        *,
        system_instruction: str,
        prompt: str,
    ) -> int:
        return calculate_prompt_budget_warning(
            max_prompt_chars=self._max_prompt_chars,
            system_instruction=system_instruction,
            prompt_markup=prompt,
        )

    def _generation_prompt_budget_exceeded(
        self,
        *,
        system_instruction: str,
        prompt: str,
    ) -> bool:
        return prompt_budget_exceeded(
            max_prompt_chars=self._max_prompt_chars,
            system_instruction=system_instruction,
            prompt_markup=prompt,
        )

    def _context_exceeds_limits(self, context_decision: object) -> bool:
        chunks = getattr(context_decision, "chunks", ())
        if len(chunks) > self._max_chunks_per_request:
            return True

        return any(len(chunk.text) > self._max_chunk_chars for chunk in chunks)

    @staticmethod
    def _document_extension(file_name: str) -> str:
        stripped_name = file_name.strip()
        if "." not in stripped_name:
            return ""
        return stripped_name.rsplit(".", 1)[-1].lower()

    @staticmethod
    def _assistant_mode_for_task(task_type: IntentTaskType) -> AssistantMode:
        if task_type == IntentTaskType.ANSWER_QUESTION:
            return AssistantMode.ANSWER_QUESTION
        if task_type == IntentTaskType.EXPLAIN_TOPIC:
            return AssistantMode.EXPLAIN_TOPIC
        if task_type == IntentTaskType.DOCUMENT_SEARCH:
            return AssistantMode.DOCUMENT_SEARCH

        raise ValueError(f"Unsupported retrieval task type: {task_type.value}")

    async def _publish_response(
        self,
        user_id: str,
        data: str,
        *,
        warning: int = 0,
        mark_completed: bool = True,
    ) -> None:
        await self._publisher.publish_event(
            ResponseEvent(
                type=OutgoingEventType.RESPONSE,
                user_id=user_id,
                data=data,
                warning=warning,
            )
        )
        if mark_completed:
            await self._redis_state.set_task_status(
                user_id=user_id,
                status="completed",
            )
