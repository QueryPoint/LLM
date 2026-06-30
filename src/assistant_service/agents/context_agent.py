from dataclasses import dataclass
from uuid import UUID

from assistant_service.core.enums import AssistantMode, RetrievalStatus
from assistant_service.messaging.contracts import DocumentContext, RetrievedChunk


@dataclass(frozen=True, slots=True)
class ContextSource:
    chunk_id: UUID
    document_id: UUID
    file_name: str
    page: int


@dataclass(frozen=True, slots=True)
class ContextDecision:
    status: RetrievalStatus
    chunks: tuple[RetrievedChunk, ...]
    sources: tuple[ContextSource, ...]
    total_chars: int


@dataclass(frozen=True, slots=True)
class _RankedChunk:
    index: int
    chunk: RetrievedChunk


class ContextAgent:
    def __init__(self, max_context_chars: int = 12_000, max_chunks: int = 8) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        self._max_context_chars = max_context_chars
        self._max_chunks = max_chunks

    def prepare(
        self,
        document_context: DocumentContext | None,
        mode: AssistantMode | None = None,
    ) -> ContextDecision:
        if document_context is None:
            return self._empty_decision(RetrievalStatus.NOT_FOUND)

        if document_context.retrieval_status == RetrievalStatus.NOT_FOUND:
            return self._empty_decision(RetrievalStatus.NOT_FOUND)

        if mode == AssistantMode.SUMMARIZE_DOCUMENT:
            return self._prepare_complete_document(document_context)

        ranked_chunks = self._deduplicate(document_context.chunks)
        sorted_chunks = sorted(
            ranked_chunks,
            key=lambda ranked_chunk: (-ranked_chunk.chunk.score, ranked_chunk.index),
        )
        selected_chunks = self._select_chunks(sorted_chunks)
        sources = tuple(self._build_source(chunk) for chunk in selected_chunks)
        total_chars = sum(len(chunk.text) for chunk in selected_chunks)

        if document_context.retrieval_status == RetrievalStatus.INSUFFICIENT:
            status = RetrievalStatus.INSUFFICIENT
        elif selected_chunks:
            status = RetrievalStatus.FOUND
        else:
            status = RetrievalStatus.INSUFFICIENT

        return ContextDecision(
            status=status,
            chunks=tuple(selected_chunks),
            sources=sources,
            total_chars=total_chars,
        )

    def _prepare_complete_document(
        self,
        document_context: DocumentContext,
    ) -> ContextDecision:
        if not document_context.is_complete_document:
            return self._empty_decision(RetrievalStatus.INSUFFICIENT)

        if not document_context.chunks:
            return self._empty_decision(RetrievalStatus.INSUFFICIENT)

        selected_chunks = self._deduplicate_preserving_order(document_context.chunks)
        if not selected_chunks:
            return self._empty_decision(RetrievalStatus.INSUFFICIENT)

        sources = tuple(self._build_source(chunk) for chunk in selected_chunks)
        total_chars = sum(len(chunk.text) for chunk in selected_chunks)
        return ContextDecision(
            status=RetrievalStatus.FOUND,
            chunks=tuple(selected_chunks),
            sources=sources,
            total_chars=total_chars,
        )

    @staticmethod
    def _empty_decision(status: RetrievalStatus) -> ContextDecision:
        return ContextDecision(status=status, chunks=(), sources=(), total_chars=0)

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[_RankedChunk]:
        ranked_chunks: list[_RankedChunk] = []

        for index, chunk in enumerate(chunks):
            normalized_text = self._normalize_text(chunk.text)
            duplicate_indexes = [
                candidate_index
                for candidate_index, candidate in enumerate(ranked_chunks)
                if candidate.chunk.chunk_id == chunk.chunk_id
                or self._normalize_text(candidate.chunk.text) == normalized_text
            ]

            if not duplicate_indexes:
                ranked_chunks.append(_RankedChunk(index=index, chunk=chunk))
                continue

            duplicates = [
                ranked_chunks[candidate_index]
                for candidate_index in duplicate_indexes
            ]
            best_chunk = min(
                [*duplicates, _RankedChunk(index=index, chunk=chunk)],
                key=lambda candidate: (-candidate.chunk.score, candidate.index),
            )
            ranked_chunks = [
                candidate
                for candidate_index, candidate in enumerate(ranked_chunks)
                if candidate_index not in duplicate_indexes
            ]
            ranked_chunks.append(best_chunk)

        return ranked_chunks

    def _deduplicate_preserving_order(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        selected_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[UUID] = set()
        seen_texts: set[str] = set()

        for chunk in chunks:
            normalized_text = self._normalize_text(chunk.text)
            if chunk.chunk_id in seen_chunk_ids or normalized_text in seen_texts:
                continue

            selected_chunks.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            seen_texts.add(normalized_text)

        return selected_chunks

    def _select_chunks(self, ranked_chunks: list[_RankedChunk]) -> list[RetrievedChunk]:
        selected_chunks: list[RetrievedChunk] = []
        remaining_chars = self._max_context_chars

        for ranked_chunk in ranked_chunks:
            if len(selected_chunks) >= self._max_chunks:
                break

            chunk = ranked_chunk.chunk
            text_length = len(chunk.text)
            if text_length > remaining_chars:
                continue

            selected_chunks.append(chunk)
            remaining_chars -= text_length

        return selected_chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _build_source(chunk: RetrievedChunk) -> ContextSource:
        return ContextSource(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            page=chunk.page,
        )
