"""
Context engine type definitions.

Provides dataclasses for structured context representation throughout
the query understanding and retrieval pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    """Timezone-aware current time. All session/feedback timestamps use UTC."""
    return datetime.now(timezone.utc)


class Intent(Enum):
    """
    Query intent categories.

    Used to adapt retrieval strategy and prompt construction.
    """
    FACTUAL = "factual"           # Direct fact lookup: "What is X?"
    COMPARISON = "comparison"      # Compare entities: "What's the difference between X and Y?"
    HOW_TO = "how_to"             # Procedural: "How do I X?"
    SUMMARIZATION = "summarization"  # Summarize content: "Summarize X"
    CLARIFICATION = "clarification"  # Follow-up: "What do you mean by X?"
    EXPLORATORY = "exploratory"    # Open-ended: "Tell me about X"
    LIST = "list"                 # Enumeration: "List all X"
    DEFINITION = "definition"      # Define term: "Define X"
    UNKNOWN = "unknown"           # Fallback


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation.
    """
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=utcnow)
    intent: Optional[Intent] = None
    entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent.value if self.intent else None,
            "entities": self.entities,
        }


@dataclass
class SessionContext:
    """
    Full context for a user session.

    Tracks conversation history, extracted entities, and user preferences.
    """
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # entity -> canonical form
    topic_stack: List[str] = field(default_factory=list)    # current discussion topics
    created_at: datetime = field(default_factory=utcnow)
    last_active: datetime = field(default_factory=utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, role: str, content: str, intent: Optional[Intent] = None,
                 entities: Optional[List[str]] = None) -> ConversationTurn:
        """Add a conversation turn and update last_active."""
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            entities=entities or []
        )
        self.turns.append(turn)
        self.last_active = utcnow()

        # Update entity map
        for entity in (entities or []):
            if entity not in self.entities:
                self.entities[entity] = entity

        return turn

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        return self.turns[-n:] if self.turns else []

    def get_conversation_text(self, n: int = 5) -> str:
        """Get recent conversation as formatted text for LLM context."""
        recent = self.get_recent_turns(n)
        lines = []
        for turn in recent:
            prefix = "User:" if turn.role == "user" else "Assistant:"
            lines.append(f"{prefix} {turn.content}")
        return "\n".join(lines)

    def get_last_user_query(self) -> Optional[str]:
        """Get the most recent user query."""
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_current_topic(self) -> Optional[str]:
        """Get the current discussion topic."""
        return self.topic_stack[-1] if self.topic_stack else None

    def clear_topics(self) -> None:
        """Clear the topic stack (called when a topic shift is detected)."""
        self.topic_stack = []

    def push_topic(self, topic: str) -> None:
        """Push a new topic onto the stack."""
        if not self.topic_stack or self.topic_stack[-1] != topic:
            self.topic_stack.append(topic)
            # Keep stack bounded
            if len(self.topic_stack) > 10:
                self.topic_stack = self.topic_stack[-10:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "entities": self.entities,
            "topic_stack": self.topic_stack,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class QueryContext:
    """
    Enriched context for a single query.

    Produced by the context engine and consumed by the retrieval pipeline.
    """
    original_query: str
    rewritten_query: str
    intent: Intent
    entities: List[str]
    session: Optional[SessionContext]

    # Follow-up context
    is_followup: bool = False             # Query refers back to the conversation
    anchor_doc_ids: List[str] = field(default_factory=list)  # Docs cited in the previous answer

    # Retrieval hints
    require_recency: bool = False        # Weight recent documents higher
    require_comparison: bool = False      # Need multiple perspectives
    require_procedure: bool = False       # Need step-by-step content
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    # Search parameters (can be adjusted based on intent)
    top_k: int = 6
    include_neighbors: bool = True
    diversity_weight: float = 0.3        # MMR lambda

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "intent": self.intent.value,
            "entities": self.entities,
            "is_followup": self.is_followup,
            "anchor_doc_ids": self.anchor_doc_ids,
            "require_recency": self.require_recency,
            "require_comparison": self.require_comparison,
            "require_procedure": self.require_procedure,
            "metadata_filters": self.metadata_filters,
            "top_k": self.top_k,
            "include_neighbors": self.include_neighbors,
            "diversity_weight": self.diversity_weight,
        }
