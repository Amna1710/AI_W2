
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from rag_system.models.model_factory import ModelFactory


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "query_rewrite_prompt.txt"
)


class QueryRewriter:

    def __init__(self):

        if not PROMPT_PATH.exists():
            raise FileNotFoundError(
                f"Query rewrite prompt not found: {PROMPT_PATH}"
            )

        self.system_prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

    def rewrite_query(
        self,
        question: str,
        chat_history: list,
        provider: str | None = None,
    ) -> str:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        # No history means no rewriting is necessary.
        if not chat_history:
            return question

        # Select the configured LLM.
        llm = ModelFactory.get_chat_model(
            provider=provider,
            temperature=0.0,
        )

        # Build the rewriting prompt.
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            ("human", "{question}"),
        ])

        # Build the LCEL chain.
        chain = prompt | llm | StrOutputParser()

        # Generate standalone search query.
        rewritten_query = chain.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            },
            config={
                "run_name": "query_rewriting",
                "tags": ["rag", "query-rewriting"],
                "metadata": {
                    "component": "query_rewriter",
                },
            },
        )

        rewritten_query = rewritten_query.strip()

        return rewritten_query or question
