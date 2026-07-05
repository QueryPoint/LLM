from assistant_service.agents.intent_agent import IntentDecision, IntentTaskType
from assistant_service.core.enums import IncomingMessageType, OutgoingEventType
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    PromptRequestMessage,
    ResponseEvent,
    ThinkEvent,
)
from assistant_service.services.elasticsearch_client import DocumentMetadata, SearchResult


def build_prompt_message(
    *,
    user_id: str = "user-123",
    prompt: str = "Объясни нормализацию",
    doc: str | None = None,
) -> PromptRequestMessage:
    return PromptRequestMessage(
        type=IncomingMessageType.PROMPT,
        user_id=user_id,
        prompt=prompt,
        doc=doc,
    )


def build_delete_message(*, user_id: str = "user-123") -> DeleteRequestMessage:
    return DeleteRequestMessage(type=IncomingMessageType.DELETE, user_id=user_id)


def build_think_event(
    *,
    user_id: str = "user-123",
    data: str = "Определяю тип запроса...",
) -> ThinkEvent:
    return ThinkEvent(type=OutgoingEventType.THINK, user_id=user_id, data=data)


def build_response_event(
    *,
    user_id: str = "user-123",
    data: str = "Ответ готов.",
    warning: int = 0,
) -> ResponseEvent:
    return ResponseEvent(
        type=OutgoingEventType.RESPONSE,
        user_id=user_id,
        data=data,
        warning=warning,
    )


def build_search_result(
    *,
    doc_id: str = "document-1",
    chunk_id: str = "chunk-1",
    file_name: str = "lecture.pdf",
    page_number: int = 3,
    text: str = "Нормализация уменьшает избыточность.",
    score: float = 12.5,
) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_id=chunk_id,
        file_name=file_name,
        page_number=page_number,
        text=text,
        score=score,
        highlights=(),
    )


def build_document_metadata(
    *,
    doc_id: str = "document-uid-123",
    file_name: str = "lecture.pdf",
) -> DocumentMetadata:
    return DocumentMetadata(doc_id=doc_id, file_name=file_name)


def build_intent_decision(
    *,
    task_type: IntentTaskType = IntentTaskType.EXPLAIN_TOPIC,
    keywords: list[str] | None = None,
    source: str = "rule_based",
) -> IntentDecision:
    normalized_keywords = keywords if keywords is not None else ["нормализация"]
    return IntentDecision(
        task_type=task_type,
        requires_retrieval=task_type != IntentTaskType.UNSUPPORTED,
        requires_full_document=task_type == IntentTaskType.SUMMARIZE_DOCUMENT,
        keywords=normalized_keywords,
        source=source,
    )
