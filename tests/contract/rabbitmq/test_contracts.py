import pytest
from pydantic import ValidationError

from assistant_service.messaging.contracts import (
    incoming_message_adapter,
    outgoing_event_adapter,
)
from tests.fixtures.builders import build_delete_message, build_prompt_message


def test_prompt_accepts_minimal_uid_contract() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": "user-123",
            "prompt": "Объясни нормализацию баз данных",
            "uid": " document-uid-123 ",
        }
    )

    expected = build_prompt_message(
        prompt="Объясни нормализацию баз данных",
        uid="document-uid-123",
    )
    assert message == expected


@pytest.mark.parametrize("field_name", ["mode", "document_context", "doc_uid", "request_id", "session_id"])
def test_old_prompt_fields_are_not_accepted(field_name: str) -> None:
    payload: dict[str, object] = {
        "type": "prompt",
        "user_id": "user-123",
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
            "user_id": "user-123",
        }
    )

    assert message == build_delete_message()


def test_outgoing_events_keep_string_payloads_and_no_extra_fields() -> None:
    think = outgoing_event_adapter.validate_python(
        {
            "type": "think",
            "user_id": "user-123",
            "data": "Определяю тип запроса...",
        }
    )
    response = outgoing_event_adapter.validate_python(
        {
            "type": "response",
            "user_id": "user-123",
            "data": "Ответ готов.",
            "warning": 125,
        }
    )

    assert isinstance(think.data, str)
    assert think.data == "Определяю тип запроса..."
    assert isinstance(response.data, str)
    assert response.data == "Ответ готов."
    assert response.warning == 125
    assert set(think.model_dump().keys()) == {"type", "user_id", "data"}
    assert set(response.model_dump().keys()) == {"type", "user_id", "data", "warning"}
