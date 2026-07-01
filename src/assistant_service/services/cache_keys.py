import hashlib
import json
from uuid import UUID

from assistant_service.agents.context_agent import ContextDecision
from assistant_service.core.enums import AssistantMode

ANSWER_CACHE_PREFIX = "answer:v1:"
INTENT_CACHE_PREFIX = "intent:v1:"


def build_answer_cache_key(
    *,
    user_id: UUID,
    mode: AssistantMode,
    user_prompt: str | None,
    context_decision: ContextDecision,
) -> str:
    fingerprint = {
        "version": "v1",
        "user_id": str(user_id),
        "mode": mode.value,
        "prompt": _normalize_text(user_prompt),
        "chunks": [
            {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "file_name": chunk.file_name,
                "page": chunk.page,
                "text": chunk.text,
            }
            for chunk in context_decision.chunks
        ],
    }
    return f"{ANSWER_CACHE_PREFIX}{_stable_hash(fingerprint)}"


def build_answer_lock_key(answer_cache_key: str) -> str:
    digest = answer_cache_key.removeprefix(ANSWER_CACHE_PREFIX)
    return f"lock:answer:{digest}"


def build_intent_cache_key(
    *,
    prompt: str | None,
    document_id: UUID | None,
    requested_mode: AssistantMode | None,
) -> str:
    fingerprint = {
        "version": "v1",
        "prompt": _normalize_text(prompt),
        "document_id": str(document_id) if document_id is not None else None,
        "requested_mode": requested_mode.value if requested_mode is not None else None,
    }
    return f"{INTENT_CACHE_PREFIX}{_stable_hash(fingerprint)}"


def build_task_status_key(*, user_id: UUID) -> str:
    return f"task:{user_id}:status"


def build_session_summary_key(*, user_id: UUID) -> str:
    return f"chat_summary:{user_id}"


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())


def _stable_hash(payload: object) -> str:
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
