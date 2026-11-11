from threading import RLock
from typing import Dict, List, Tuple
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


class ConversationMemory:

    def __init__(self, max_messages: int = 20):

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.max_messages = max_messages

        self._conversations: Dict[
            Tuple[str, str],
            List[BaseMessage],
        ] = {}

        self._lock = RLock()

    def get_history(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> List[BaseMessage]:

        key = (str(user_id), str(conversation_id))

        with self._lock:
            return list(self._conversations.get(key, []))

    def add_message(
        self,
        user_id: UUID,
        conversation_id: UUID,
        role: str,
        content: str,
    ) -> None:

        role = role.lower().strip()

        if not content or not content.strip():
            raise ValueError("Message content cannot be empty.")

        if role == "user":
            message = HumanMessage(content=content)

        elif role == "assistant":
            message = AIMessage(content=content)

        else:
            raise ValueError(
                "Role must be either 'user' or 'assistant'."
            )

        key = (str(user_id), str(conversation_id))

        with self._lock:

            history = self._conversations.setdefault(key, [])

            history.append(message)

            # Keep only the most recent messages.
            self._conversations[key] = history[
                -self.max_messages:
            ]

    def clear_history(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> bool:

        key = (str(user_id), str(conversation_id))

        with self._lock:
            existed = key in self._conversations

            self._conversations.pop(key, None)

            return existed