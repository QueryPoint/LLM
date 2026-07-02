from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from assistant_service.core.enums import (
    IncomingMessageType,
    OutgoingEventType,
    RetrievalStatus,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievedChunk(StrictBaseModel):
    chunk_id: UUID
    document_id: UUID
    file_name: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str = Field(min_length=1)
    score: float

    @field_validator("file_name", "text", mode="before")
    @classmethod
    def _strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class DocumentContext(StrictBaseModel):
    retrieval_status: RetrievalStatus
    is_complete_document: bool = False
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class PromptRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.PROMPT]
    user_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    uid: str | None = None

    @field_validator("user_id", "prompt", mode="before")
    @classmethod
    def _strip_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("uid", mode="before")
    @classmethod
    def _strip_optional_uid(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value


class DeleteRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.DELETE]
    user_id: str = Field(min_length=1)

    @field_validator("user_id", mode="before")
    @classmethod
    def _strip_user_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


IncomingMessage = Annotated[
    PromptRequestMessage | DeleteRequestMessage,
    Field(discriminator="type"),
]


class ThinkEvent(StrictBaseModel):
    type: Literal[OutgoingEventType.THINK]
    user_id: str = Field(min_length=1)
    data: str = Field(min_length=1)


class ResponseEvent(StrictBaseModel):
    type: Literal[OutgoingEventType.RESPONSE]
    user_id: str = Field(min_length=1)
    data: str = Field(min_length=1)
    warning: int = Field(default=0, ge=0)


OutgoingEvent = Annotated[
    ThinkEvent | ResponseEvent,
    Field(discriminator="type"),
]


incoming_message_adapter = TypeAdapter(IncomingMessage)
outgoing_event_adapter = TypeAdapter(OutgoingEvent)
