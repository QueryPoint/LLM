import pytest
from pydantic import ValidationError

from assistant_service.messaging.contracts import (
    incoming_message_adapter,
    outgoing_event_adapter,
)


USER_ID = "user-123"


def test_prompt_accepts_minimal_uid_contract() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Объясни нормализацию баз данных",
            "uid": " document-uid-123 ",
        }
    )

    assert message.type == "prompt"
    assert message.user_id == USER_ID
    assert message.prompt == "Объясни нормализацию баз данных"
    assert message.uid == "document-uid-123"


@pytest.mark.parametrize("field_name", ["mode", "document_context", "doc_uid", "doc"])
def test_old_prompt_fields_are_not_accepted(field_name: str) -> None:
    payload: dict[str, object] = {
        "type": "prompt",
        "user_id": USER_ID,
        "prompt": "Объясни нормализацию",
        "uid": None,
        field_name: "legacy",
    }

    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(payload)


def test_delete_is_valid_without_prompt_or_uid() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "delete",
            "user_id": USER_ID,
        }
    )

    assert message.type == "delete"
    assert message.user_id == USER_ID


def test_think_event_keeps_human_readable_string_data() -> None:
    event = outgoing_event_adapter.validate_python(
        {
            "type": "think",
            "user_id": USER_ID,
            "data": "Определяю тип запроса...",
        }
    )

    assert event.type == "think"
    assert isinstance(event.data, str)
    assert event.data == "Определяю тип запроса..."


def test_response_warning_remains_numeric_and_can_exceed_100() -> None:
    event = outgoing_event_adapter.validate_python(
        {
            "type": "response",
            "user_id": USER_ID,
            "data": "Ответ готов.",
            "warning": 125,
        }
    )

    assert event.type == "response"
    assert event.warning == 125
