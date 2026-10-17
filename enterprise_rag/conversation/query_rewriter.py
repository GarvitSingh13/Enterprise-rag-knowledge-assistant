"""Intelligent Query Rewriting and Expansion Engine for Enterprise RAG.

Transforms conversational, pronoun-heavy follow-up questions into complete,
self-contained search queries using dialogue context from ConversationMemory.
"""

from typing import List, Optional, Set
import re
from enterprise_rag.conversation.memory import ConversationMemory, ConversationTurn


class QueryRewriter:
    """
    De-contextualizes conversational queries by resolving anaphoric references
    (pronouns like 'that', 'it', 'those') and injecting previous turn context.
    """

    PRONOUNS_AND_REFERENTS: Set[str] = {
        "it", "that", "this", "these", "those", "they", "them",
        "the same", "too", "also", "as well", "what about", "how about",
        "and for", "can i", "is there any", "does it apply", "does that apply",
    }

    STOP_WORDS: Set[str] = {
        "what", "is", "the", "are", "for", "how", "much", "can", "do", "does",
        "i", "get", "a", "an", "to", "in", "on", "of", "and", "or", "our", "we",
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _contains_referent(self, query: str) -> bool:
        """Check if query contains ambiguous pronouns or shorthand follow-ups."""
        q_lower = query.lower()
        # Direct phrase match
        for phrase in ["does that apply", "what about", "how about", "and for", "does it", "can i get"]:
            if phrase in q_lower:
                return True

        # Token match for singular pronouns
        tokens = re.findall(r"\b\w+\b", q_lower)
        return any(t in self.PRONOUNS_AND_REFERENTS for t in tokens)

    def _extract_subject_from_turn(self, turn: ConversationTurn) -> str:
        """Extract the main substantive entities/topic from a previous turn."""
        base_query = turn.rewritten_query or turn.user_raw_query
        tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", base_query.lower())

        substantive = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 2]
        return " ".join(substantive)

    def rewrite(self, query: str, memory: Optional[ConversationMemory] = None) -> str:
        """
        Produce a standalone search query.

        If previous turns exist and current query is a conversational follow-up,
        resolves pronouns and synthesizes a fully self-contained query.
        """
        clean_query = query.strip()
        if not memory or not memory.turns:
            return clean_query

        # If LLM client is available, use prompt-based rewriting
        if self.llm_client:
            try:
                return self._llm_rewrite(clean_query, memory)
            except Exception:
                pass

        # Deterministic Contextual Rewriter
        last_turn = memory.get_last_turn()
        if not last_turn:
            return clean_query

        # Check if the query is an anaphoric follow-up
        if self._contains_referent(clean_query):
            prev_subject = self._extract_subject_from_turn(last_turn)

            # Case A: Pronoun replacement ("does that apply to international flights?")
            q_lower = clean_query.lower()
            if "that" in q_lower or "it" in q_lower:
                # Replace 'that' or 'it' with the previous subject
                rewritten = re.sub(r"\b(that|it)\b", prev_subject, clean_query, flags=re.IGNORECASE)
                return rewritten.strip()

            # Case B: Elliptical follow-up ("what about weekends?", "and international travel?")
            if q_lower.startswith(("what about", "how about", "and for", "and")):
                # Strip leading conversational marker and append to previous subject
                clean_tail = re.sub(r"^(what about|how about|and for|and)\s+", "", clean_query, flags=re.IGNORECASE)
                return f"{prev_subject} {clean_tail}".strip()

            # Case C: General contextual blend
            return f"{prev_subject} {clean_query}".strip()

        return clean_query

    def _llm_rewrite(self, query: str, memory: ConversationMemory) -> str:
        """Use LLM to perform de-contextualized query rewriting."""
        history = memory.get_history_as_text(max_turns=3)
        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rewrite the follow-up question into a standalone, complete search query. "
            "Do NOT answer the question. Only return the rewritten query.\n\n"
            f"Conversation History:\n{history}\n\n"
            f"Follow-up Question: {query}\n"
            "Standalone Query:"
        )
        return self.llm_client.generate(prompt).strip()

    def expand_query(self, query: str) -> List[str]:
        """
        Multi-Query Expansion: Generates 2 search variations
        to increase hybrid retrieval recall.
        """
        clean = query.strip()
        variations = [clean]

        # Synonym expansion rules for enterprise queries
        expansions = {
            "money": "allowance reimbursement stipend",
            "refund": "reimbursement claim policy",
            "remote": "work from home hybrid work",
            "flight": "airfare travel booking",
            "error": "incident troubleshooting guide",
            "bonus": "equity incentive compensation",
        }

        for keyword, synonyms in expansions.items():
            if keyword in clean.lower():
                variations.append(f"{clean} {synonyms}")
                break

        return variations
