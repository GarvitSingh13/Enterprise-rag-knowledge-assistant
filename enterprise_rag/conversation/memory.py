"""Multi-Turn Conversation Memory Buffer for Enterprise RAG.

Manages conversational state, tracks query-response turns, enforces
sliding window constraints, and formats history for query rewriting and LLM context.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone


class ChatMessage(BaseModel):
    """A single chat message in a conversation thread."""
    role: str = Field(description="'user', 'assistant', or 'system'")
    content: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )


class ConversationTurn(BaseModel):
    """A complete interaction turn including raw input, rewritten query, and sources."""
    turn_index: int
    user_raw_query: str
    rewritten_query: Optional[str] = None
    assistant_response: str
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )


class ConversationMemory:
    """Session-based conversation buffer with sliding window management."""

    def __init__(self, session_id: Optional[str] = None, max_turns: int = 6):
        self.session_id = session_id or str(uuid.uuid4())
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.messages: List[ChatMessage] = []

    def add_turn(
        self,
        user_raw_query: str,
        assistant_response: str,
        rewritten_query: Optional[str] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> ConversationTurn:
        """Record a completed conversation turn."""
        turn_idx = len(self.turns) + 1
        turn = ConversationTurn(
            turn_index=turn_idx,
            user_raw_query=user_raw_query.strip(),
            rewritten_query=rewritten_query.strip() if rewritten_query else None,
            assistant_response=assistant_response.strip(),
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            sources=sources or [],
        )

        self.turns.append(turn)
        self.messages.append(ChatMessage(role="user", content=user_raw_query.strip()))
        self.messages.append(ChatMessage(role="assistant", content=assistant_response.strip()))

        # Enforce sliding window on turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

        return turn

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Return the most recent conversation turn."""
        return self.turns[-1] if self.turns else None

    def get_history_as_text(self, max_turns: Optional[int] = None) -> str:
        """
        Format recent conversation turns into a clean dialogue string
        suitable for feeding into the Query Rewriter or LLM prompt.
        """
        window_size = max_turns or self.max_turns
        recent_turns = self.turns[-window_size:]

        if not recent_turns:
            return ""

        formatted_lines = []
        for t in recent_turns:
            formatted_lines.append(f"User: {t.user_raw_query}")
            formatted_lines.append(f"Assistant: {t.assistant_response}")

        return "\n".join(formatted_lines)

    def get_last_topics(self) -> List[str]:
        """Extract key entities/terms discussed in recent turns."""
        topics = []
        for t in self.turns[-2:]:
            query = t.rewritten_query or t.user_raw_query
            topics.append(query)
        return topics

    def clear(self) -> None:
        """Reset conversation memory."""
        self.turns.clear()
        self.messages.clear()
