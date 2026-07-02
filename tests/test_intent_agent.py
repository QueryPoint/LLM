from assistant_service.agents.intent_agent import IntentAgent
from assistant_service.core.enums import AssistantMode

DOCUMENT_UID = "document-uid-123"


def test_empty_prompt_with_uid_is_summarize_document() -> None:
    decision = IntentAgent().detect(
        prompt="",
        uid=DOCUMENT_UID,
    )

    assert decision.mode == AssistantMode.SUMMARIZE_DOCUMENT
    assert decision.source == "rule_based"


def test_document_search_request_is_document_search() -> None:
    decision = IntentAgent().detect(
        prompt="В каком файле есть нормализация?",
        uid=None,
    )

    assert decision.mode == AssistantMode.DOCUMENT_SEARCH


def test_explain_request_is_explain_topic() -> None:
    decision = IntentAgent().detect(
        prompt="Объясни нормализацию",
        uid=None,
    )

    assert decision.mode == AssistantMode.EXPLAIN_TOPIC


def test_neutral_request_without_document_is_answer_question() -> None:
    decision = IntentAgent().detect(
        prompt="Когда проходит экзамен?",
        uid=None,
    )

    assert decision.mode == AssistantMode.ANSWER_QUESTION
