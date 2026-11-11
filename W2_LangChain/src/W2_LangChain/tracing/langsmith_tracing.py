
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables before importing model configuration.
load_dotenv()

from W2_LangChain.models.model_factory import get_chat_model


@traceable(name="W2 LangChain Example")
def run_traced_chain(question: str) -> str:
    """Execute a LangChain chain and record its trace."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Answer clearly and concisely.",
            ),
            ("human", "{question}"),
        ]
    )

    llm = get_chat_model()

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"question": question})
