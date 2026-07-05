import pytest
from pydantic import ValidationError

from assistant_service.messaging.contracts import (
    incoming_message_adapter,
    outgoing_event_adapter,
)
from tests.fixtures.builders import build_delete_message, build_prompt_message


def test_prompt_accepts_doc_contract() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": "user-123",
            "prompt": "Объясни тему",
            "doc": " test-document-id ",
        }
    )

    expected = build_prompt_message(
        prompt="Объясни тему",
        doc="test-document-id",
    )
    assert message == expected


def test_prompt_accepts_null_doc() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": "user-123",
            "prompt": "Объясни нормализацию",
            "doc": None,
        }
    )

    assert message == build_prompt_message(prompt="Объясни нормализацию", doc=None)


@pytest.mark.parametrize(
    "field_name",
    ["mode", "document_context", "doc_uid", "request_id", "session_id"],
)
def test_old_prompt_fields_are_not_accepted(field_name: str) -> None:
    payload: dict[str, object] = {
        "type": "prompt",
        "user_id": "user-123",
        "prompt": "Объясни нормализацию",
        "doc": None,
        field_name: "legacy",
    }

    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(payload)


def test_prompt_rejects_uid_field_only() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "prompt",
                "user_id": "user-123",
                "prompt": "Объясни нормализацию",
                "uid": "legacy-document-id",
            }
        )


def test_prompt_rejects_doc_and_uid_together() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "prompt",
                "user_id": "user-123",
                "prompt": "Объясни нормализацию",
                "doc": "document-id-123",
                "uid": "legacy-document-id",
            }
        )


def test_delete_is_valid_without_prompt_or_doc() -> None:
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
    assert "doc" not in think.model_dump()
    assert "doc" not in response.model_dump()
