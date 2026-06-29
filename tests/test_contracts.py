import pytest
from pydantic import ValidationError

from assistant_service.core.enums import AssistantMode, ThinkStage
from assistant_service.messaging.contracts import (
    incoming_message_adapter,
    outgoing_event_adapter,
)


REQUEST_ID = "00000000-0000-0000-0000-000000000001"
USER_ID = "00000000-0000-0000-0000-000000000002"
SESSION_ID = "00000000-0000-0000-0000-000000000003"
DOC_ID = "00000000-0000-0000-0000-000000000004"
CHUNK_ID = "00000000-0000-0000-0000-000000000005"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000006"


def test_minimal_prompt_is_valid() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Explain database normalization",
            "doc": None,
        }
    )

    assert message.type == "prompt"
    assert message.request_id is None
    assert message.session_id is None
    assert message.document_context is None


def test_extended_prompt_with_mode_and_document_context_is_valid() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "request_id": REQUEST_ID,
            "user_id": USER_ID,
            "session_id": SESSION_ID,
            "prompt": "Explain database normalization simply",
            "doc": DOC_ID,
            "mode": "explain_topic",
            "document_context": {
                "retrieval_status": "found",
                "chunks": [
                    {
                        "chunk_id": CHUNK_ID,
                        "document_id": DOCUMENT_ID,
                        "file_name": "lecture_database.pdf",
                        "page": 5,
                        "text": "Database normalization organizes data.",
                        "score": 12.45,
                    }
                ],
            },
        }
    )

    assert message.type == "prompt"
    assert message.mode == "explain_topic"
    assert message.document_context is not None
    assert message.document_context.chunks[0].page == 5


def test_delete_is_valid() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "delete",
            "request_id": REQUEST_ID,
            "user_id": USER_ID,
            "session_id": SESSION_ID,
        }
    )

    assert message.type == "delete"


def test_unknown_type_fails_validation() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "unknown",
                "user_id": USER_ID,
            }
        )


def test_invalid_uuid_fails_validation() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "prompt",
                "user_id": "not-a-uuid",
                "prompt": "Explain normalization",
                "doc": None,
            }
        )


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


def test_think_event_with_stage_is_valid() -> None:
    event = outgoing_event_adapter.validate_python(
        {
            "request_id": REQUEST_ID,
            "user_id": USER_ID,
            "type": "think",
            "stage": "PREPARING_CONTEXT",
            "data": "Preparing context",
            "warning": 0,
        }
    )

    assert event.type == "think"
    assert event.stage == "PREPARING_CONTEXT"


@pytest.mark.parametrize("warning", [-1, 101])
def test_invalid_warning_fails_validation(warning: int) -> None:
    with pytest.raises(ValidationError):
        outgoing_event_adapter.validate_python(
            {
                "request_id": REQUEST_ID,
                "user_id": USER_ID,
                "type": "think",
                "stage": "PREPARING_CONTEXT",
                "data": "Preparing context",
                "warning": warning,
            }
        )


def test_done_event_with_typed_source_is_valid() -> None:
    event = outgoing_event_adapter.validate_python(
        {
            "request_id": REQUEST_ID,
            "user_id": USER_ID,
            "type": "done",
            "data": {
                "answer": "Normalization reduces duplication.",
                "sources": [
                    {
                        "document_id": DOCUMENT_ID,
                        "file_name": "lecture_database.pdf",
                        "page": 5,
                        "chunk_id": CHUNK_ID,
                        "text": "Database normalization organizes data.",
                        "score": 12.45,
                    }
                ],
            },
            "warning": 0,
        }
    )

    assert event.type == "done"
    assert event.data.sources[0].file_name == "lecture_database.pdf"


def test_invalid_retrieval_status_fails_validation() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "prompt",
                "user_id": USER_ID,
                "prompt": "Explain normalization",
                "doc": None,
                "document_context": {
                    "retrieval_status": "unknown",
                    "chunks": [],
                },
            }
        )


def test_invalid_mode_fails_validation() -> None:
    with pytest.raises(ValidationError):
        incoming_message_adapter.validate_python(
            {
                "type": "prompt",
                "user_id": USER_ID,
                "prompt": "Explain normalization",
                "doc": None,
                "mode": "free_chat",
            }
        )


def test_think_stage_serializes_to_json_value() -> None:
    assert ThinkStage.PREPARING_CONTEXT.value == "PREPARING_CONTEXT"


def test_assistant_mode_validates_from_json_value() -> None:
    message = incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Explain normalization",
            "doc": None,
            "mode": "explain_topic",
        }
    )

    assert message.mode is AssistantMode.EXPLAIN_TOPIC
