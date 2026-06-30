import pytest
from pydantic import ValidationError

from assistant_service.core.enums import AssistantMode
from assistant_service.messaging.contracts import (
    incoming_message_adapter,
    outgoing_event_adapter,
)


USER_ID = "00000000-0000-0000-0000-000000000002"
DOC_ID = "00000000-0000-0000-0000-000000000004"


def test_prompt_is_valid() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Explain database normalization",
            "doc": DOC_ID,
            "mode": "explain_topic",
        }
    )

    assert message.type == "prompt"
    assert str(message.user_id) == USER_ID
    assert message.prompt == "Explain database normalization"
    assert str(message.doc) == DOC_ID
    assert message.mode == AssistantMode.EXPLAIN_TOPIC


def test_delete_is_valid() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "delete",
            "user_id": USER_ID,
        }
    )

    assert message.type == "delete"
    assert str(message.user_id) == USER_ID


def test_think_is_valid() -> None:
    event = outgoing_event_adapter.validate_python(
        {
            "type": "think",
            "user_id": USER_ID,
            "data": "Ищем релевантные материалы...",
        }
    )

    assert event.type == "think"
    assert event.data == "Ищем релевантные материалы..."


def test_response_is_valid() -> None:
    event = outgoing_event_adapter.validate_python(
        {
            "type": "response",
            "user_id": USER_ID,
            "data": "Ответ готов.",
        }
    )

    assert event.type == "response"
    assert event.warning == 0


def test_unknown_field_fails_validation() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "prompt",
                "user_id": USER_ID,
                "prompt": "Explain normalization",
                "doc": None,
                "unexpected": "value",
            }
        )
