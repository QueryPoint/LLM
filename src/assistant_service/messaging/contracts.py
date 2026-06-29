from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentChunk(StrictBaseModel):
    chunk_id: UUID
    page_number: int
    text: str


class DocumentContext(StrictBaseModel):
    file_name: str
    chunks: list[DocumentChunk] = Field(default_factory=list)


class PromptRequestMessage(StrictBaseModel):
    type: Literal["prompt"]
    request_id: UUID
    user_id: UUID
    session_id: UUID
    prompt: str
    doc: UUID | None = None
    document_context: DocumentContext | None = None


class DeleteRequestMessage(StrictBaseModel):
    type: Literal["delete"]
    request_id: UUID
    user_id: UUID
    session_id: UUID


IncomingMessage = Annotated[
    PromptRequestMessage | DeleteRequestMessage,
    Field(discriminator="type"),
]


class DoneEventData(StrictBaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


class ThinkEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal["think"]
    data: str
    warning: int = Field(default=0, ge=0, le=100)


class TokenEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal["token"]
    data: str
    warning: int = Field(default=0, ge=0, le=100)


class DoneEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal["done"]
    data: DoneEventData
    warning: int = Field(default=0, ge=0, le=100)


class ErrorEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal["error"]
    data: str
    warning: int = Field(default=0, ge=0, le=100)


OutgoingEvent = Annotated[
    ThinkEvent | TokenEvent | DoneEvent | ErrorEvent,
    Field(discriminator="type"),
]


incoming_message_adapter = TypeAdapter(IncomingMessage)
outgoing_event_adapter = TypeAdapter(OutgoingEvent)
