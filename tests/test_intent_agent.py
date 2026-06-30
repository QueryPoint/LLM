from uuid import UUID

from assistant_service.agents.intent_agent import IntentAgent
from assistant_service.core.enums import AssistantMode

DOC_ID = UUID("00000000-0000-0000-0000-000000000004")


def test_backend_mode_has_priority_over_prompt_text() -> None:
    decision = IntentAgent().detect(
        prompt="Объясни, что такое RabbitMQ",
        document_id=None,
        requested_mode=AssistantMode.SUMMARIZE_DOCUMENT,
    )

    assert decision.mode == AssistantMode.SUMMARIZE_DOCUMENT
    assert decision.source == "backend"


def test_empty_prompt_with_document_is_summarize_document() -> None:
    decision = IntentAgent().detect(
        prompt=None,
        document_id=DOC_ID,
        requested_mode=None,
    )

    assert decision.mode == AssistantMode.SUMMARIZE_DOCUMENT
    assert decision.source == "rule_based"


def test_document_search_request_is_document_search() -> None:
    decision = IntentAgent().detect(
        prompt="В каком файле есть нормализация?",
        document_id=None,
        requested_mode=None,
    )

    assert decision.mode == AssistantMode.DOCUMENT_SEARCH


def test_explain_request_is_explain_topic() -> None:
    decision = IntentAgent().detect(
        prompt="Объясни нормализацию",
        document_id=None,
        requested_mode=None,
    )

    assert decision.mode == AssistantMode.EXPLAIN_TOPIC


def test_neutral_request_without_document_is_answer_question() -> None:
    decision = IntentAgent().detect(
        prompt="Когда проходит экзамен?",
        document_id=None,
        requested_mode=None,
    )

    assert decision.mode == AssistantMode.ANSWER_QUESTION
