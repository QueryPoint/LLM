from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from assistant_service.core.enums import (
    AssistantMode,
    IncomingMessageType,
    OutgoingEventType,
    RetrievalStatus,
    ThinkStage,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentChunk(StrictBaseModel):
    chunk_id: UUID
    document_id: UUID
    file_name: str
    page: int | None = None
    text: str
    score: float | None = None


class DocumentContext(StrictBaseModel):
    retrieval_status: RetrievalStatus
    chunks: list[DocumentChunk] = Field(default_factory=list)


class PromptRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.PROMPT]
    request_id: UUID | None = None
    user_id: UUID
    session_id: UUID | None = None
    prompt: str
    doc: UUID | None = None
    mode: AssistantMode | None = None
    document_context: DocumentContext | None = None


class DeleteRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.DELETE]
    request_id: UUID | None = None
    user_id: UUID
    session_id: UUID | None = None


IncomingMessage = Annotated[
    PromptRequestMessage | DeleteRequestMessage,
    Field(discriminator="type"),
]


class Source(StrictBaseModel):
    document_id: UUID
    file_name: str
    page: int | None = None
    chunk_id: UUID
    text: str
    score: float | None = None


class DoneEventData(StrictBaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)


class ThinkEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal[OutgoingEventType.THINK]
    stage: ThinkStage
    data: str
    warning: int = Field(default=0, ge=0, le=100)


class DoneEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal[OutgoingEventType.DONE]
    data: DoneEventData
    warning: int = Field(default=0, ge=0, le=100)


class ErrorEvent(StrictBaseModel):
    request_id: UUID
    user_id: UUID
    type: Literal[OutgoingEventType.ERROR]
    data: str
    warning: int = Field(default=0, ge=0, le=100)


OutgoingEvent = Annotated[
    ThinkEvent | DoneEvent | ErrorEvent,
    Field(discriminator="type"),
]


incoming_message_adapter = TypeAdapter(IncomingMessage)
outgoing_event_adapter = TypeAdapter(OutgoingEvent)
