
import json
import uuid

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from W2_LangChain.models.model_factory import get_chat_model
from W2_LangChain.embeddings import get_embedding_model
from W2_LangChain.vector_stores.chroma_store import ChromaVectorStore
from W2_LangChain.retrievers import create_vectorstore_retriever


# ============================================================
# CONFIGURATION
# ============================================================

BUFFER_MAX_MESSAGES = 12

SUMMARY_TRIGGER_MESSAGES = 10
SUMMARY_KEEP_MESSAGES = 4

VECTOR_TOP_K = 3

CONVERSATION_COLLECTION = "w2_conversation_memory"


# ============================================================
# CONVERSATION PROMPT
# ============================================================

conversation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "Answer clearly and naturally. "
            "Use the supplied conversation history and memory "
            "to maintain context across questions."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ============================================================
# BASE MEMORY
# ============================================================

class BaseConversationMemory:

    def __init__(self):
        self.history = []

    def add(self, question, answer):
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

    def get_context(self, question):
        return list(self.history)

    def clear(self):
        self.history.clear()

    def get_status(self):
        return {
            "memory_type": self.__class__.__name__,
            "stored_messages": len(self.history),
        }


# ============================================================
# 1. BUFFER MEMORY
# ============================================================

class BufferMemory(BaseConversationMemory):

    def get_context(self, question):
        return self.history[-BUFFER_MAX_MESSAGES:]


# ============================================================
# 2. SUMMARY MEMORY
# ============================================================

class SummaryMemory(BaseConversationMemory):

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.summary = ""

    def add(self, question, answer):
        super().add(question, answer)

        if len(self.history) >= SUMMARY_TRIGGER_MESSAGES:
            self.update_summary()

    def update_summary(self):
        old_messages = self.history[:-SUMMARY_KEEP_MESSAGES]

        if not old_messages:
            return

        conversation_text = "\n".join(
            f"{message.type}: {message.content}"
            for message in old_messages
        )

        prompt = f"""
Update the existing conversation summary.

Existing summary:
{self.summary or "No previous summary."}

New conversation messages:
{conversation_text}

Preserve important facts, preferences, names and decisions.
Remove unnecessary repetition.
Return only the updated summary.
"""

        response = self.llm.invoke(prompt)
        self.summary = response.content.strip()

        self.history = self.history[-SUMMARY_KEEP_MESSAGES:]

    def get_context(self, question):
        context = []

        if self.summary:
            context.append(
                SystemMessage(
                    content=(
                        "Previous conversation summary:\n"
                        f"{self.summary}"
                    )
                )
            )

        context.extend(self.history)
        return context

    def clear(self):
        super().clear()
        self.summary = ""

    def get_status(self):
        return {
            "memory_type": "SummaryMemory",
            "summary_available": bool(self.summary),
            "summary": self.summary,
            "recent_messages": len(self.history),
        }


# ============================================================
# 3. VECTOR MEMORY - CHROMADB + RETRIEVER
# ============================================================

class VectorMemory:

    def __init__(self, embeddings, session_id):
        self.session_id = session_id

        # Use a separate collection for conversation memories.
        self.vector_store = ChromaVectorStore(
            embeddings=embeddings,
            collection_name=CONVERSATION_COLLECTION,
        )

        # Use the existing vector retriever with session filtering.
        self.retriever = create_vectorstore_retriever(
            vector_store=self.vector_store,
            k=VECTOR_TOP_K,
            filter={"session_id": self.session_id},
        )

        self.document_ids = []

    def add(self, question, answer):
        """Store one conversation turn in ChromaDB."""

        document_id = str(uuid.uuid4())

        document = Document(
            page_content=(
                f"User: {question}\n"
                f"Assistant: {answer}"
            ),
            metadata={
                "session_id": self.session_id,
                "memory_type": "conversation",
            },
        )

        self.vector_store.add_documents(
            documents=[document],
            ids=[document_id],
        )

        self.document_ids.append(document_id)

    def get_context(self, question):
        """Retrieve relevant previous conversation turns."""

        documents = self.retriever.invoke(question)

        if not documents:
            return []

        context_text = "\n\n".join(
            document.page_content
            for document in documents
        )

        return [
            SystemMessage(
                content=(
                    "Relevant information from previous conversations:\n"
                    f"{context_text}"
                )
            )
        ]

    def clear(self):
        """Delete only this session's stored memories."""

        if self.document_ids:
            self.vector_store.delete(self.document_ids)

        self.document_ids.clear()

    def get_status(self):
        return {
            "memory_type": "VectorMemory",
            "collection": CONVERSATION_COLLECTION,
            "session_id": self.session_id,
            "stored_conversation_entries": len(self.document_ids),
            "retrieval_limit": VECTOR_TOP_K,
        }


# ============================================================
# 4. ENTITY MEMORY
# ============================================================

class EntityMemory(BaseConversationMemory):

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.entities = {}

    def add(self, question, answer):
        super().add(question, answer)
        self.extract_entities(question, answer)

    def extract_entities(self, question, answer):
        prompt = f"""
Extract important entities and their relevant information
from this conversation.

Return ONLY a valid JSON object.

Example:
{{
    "person": "John",
    "company": "OpenAI",
    "location": "Pakistan"
}}

If there are no useful entities, return {{}}.
Do not invent information.

User: {question}
Assistant: {answer}
"""

        response = self.llm.invoke(prompt)

        try:
            content = response.content.strip()

            if content.startswith("```"):
                content = content.replace("```json", "")
                content = content.replace("```", "").strip()

            extracted = json.loads(content)

            if isinstance(extracted, dict):
                self.entities.update(extracted)

        except (json.JSONDecodeError, TypeError, AttributeError):
            print("Warning: Entity extraction returned invalid JSON.")

    def get_context(self, question):
        if not self.entities:
            return []

        entity_text = json.dumps(
            self.entities,
            indent=4,
            ensure_ascii=False,
        )

        return [
            SystemMessage(
                content=(
                    "Remember these entities from previous conversations:\n"
                    f"{entity_text}"
                )
            )
        ]

    def clear(self):
        super().clear()
        self.entities.clear()

    def get_status(self):
        return {
            "memory_type": "EntityMemory",
            "tracked_entities": self.entities,
        }


# ============================================================
# MEMORY FACTORY
# ============================================================

def create_memory(
    memory_type,
    llm,
    embeddings=None,
    session_id=None,
):
    memory_type = memory_type.lower().strip()

    if memory_type == "buffer":
        return BufferMemory()

    if memory_type == "summary":
        return SummaryMemory(llm)

    if memory_type == "vector":
        if embeddings is None:
            raise ValueError(
                "Vector memory requires an embedding model."
            )

        return VectorMemory(
            embeddings=embeddings,
            session_id=session_id or str(uuid.uuid4()),
        )

    if memory_type == "entity":
        return EntityMemory(llm)

    raise ValueError(
        f"Unsupported memory type: {memory_type}"
    )


# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:

    def __init__(self, memory_type="buffer"):
        self.llm = get_chat_model()
        self.embeddings = None

        self.session_id = str(uuid.uuid4())

        if memory_type.lower() == "vector":
            self.embeddings = get_embedding_model()

        self.memory = create_memory(
            memory_type=memory_type,
            llm=self.llm,
            embeddings=self.embeddings,
            session_id=self.session_id,
        )

        self.memory_type = memory_type.lower()

        self.chain = (
            conversation_prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question):
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve previous context.
        chat_history = self.memory.get_context(question)

        # 2. Generate an answer using the conversation chain.
        answer = self.chain.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            }
        )

        # 3. Save the new conversation turn.
        self.memory.add(question, answer)

        return answer

    def clear(self):
        self.memory.clear()

    def get_status(self):
        return self.memory.get_status()
