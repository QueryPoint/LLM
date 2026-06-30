from assistant_service.agents.context_agent import ContextDecision
from assistant_service.core.enums import AssistantMode, RetrievalStatus

TEMPORARY_CONTEXT_READY_RESPONSE = (
    "Запрос принят. Ответ по материалам базы знаний будет сформирован "
    "после подключения модулей анализа контекста."
)
DOCUMENT_SEARCH_NOT_FOUND_RESPONSE = (
    "По вашему запросу подходящие материалы в базе знаний не найдены."
)
DOCUMENT_SEARCH_INSUFFICIENT_RESPONSE = (
    "Нашлись отдельные материалы, но их недостаточно для надёжного результата. "
    "Попробуйте уточнить запрос."
)
CONTEXT_NOT_FOUND_RESPONSE = (
    "Не удалось найти подходящие материалы в базе знаний для ответа на этот запрос."
)
CONTEXT_INSUFFICIENT_RESPONSE = (
    "Найденных материалов недостаточно, чтобы дать надёжный ответ. "
    "Попробуйте уточнить запрос или выбрать другой документ."
)
SUMMARY_REQUIRES_COMPLETE_DOCUMENT_RESPONSE = (
    "Для подготовки краткого изложения нужен полный текст выбранного документа."
)


def build_response_text(mode: AssistantMode, context_decision: ContextDecision) -> str:
    if mode == AssistantMode.DOCUMENT_SEARCH:
        return _build_document_search_response(context_decision)

    if context_decision.status == RetrievalStatus.NOT_FOUND:
        return CONTEXT_NOT_FOUND_RESPONSE

    if context_decision.status == RetrievalStatus.INSUFFICIENT:
        return CONTEXT_INSUFFICIENT_RESPONSE

    return TEMPORARY_CONTEXT_READY_RESPONSE


def get_response_kind(mode: AssistantMode, context_decision: ContextDecision) -> str:
    if mode == AssistantMode.DOCUMENT_SEARCH:
        return "document_search"

    if context_decision.status == RetrievalStatus.NOT_FOUND:
        return "not_found_fallback"

    if context_decision.status == RetrievalStatus.INSUFFICIENT:
        return "insufficient_fallback"

    return "temporary_context_ready"


def build_summary_requires_complete_document_response() -> str:
    return SUMMARY_REQUIRES_COMPLETE_DOCUMENT_RESPONSE


def _build_document_search_response(context_decision: ContextDecision) -> str:
    if context_decision.status == RetrievalStatus.NOT_FOUND:
        return DOCUMENT_SEARCH_NOT_FOUND_RESPONSE

    if context_decision.status == RetrievalStatus.INSUFFICIENT:
        return DOCUMENT_SEARCH_INSUFFICIENT_RESPONSE

    fragments = [
        f"{index}. {chunk.file_name}, стр. {chunk.page}\n{chunk.text}"
        for index, chunk in enumerate(context_decision.chunks, start=1)
    ]
    return "Нашёл подходящие фрагменты:\n\n" + "\n\n".join(fragments)
