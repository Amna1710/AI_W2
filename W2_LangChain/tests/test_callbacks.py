
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from W2_LangChain.models.model_factory import get_chat_model
from W2_LangChain.callbacks import LoggingCallbackHandler


def test_custom_callback():

    print("\n" + "=" * 65)
    print("CUSTOM CALLBACK TEST")
    print("=" * 65)

    # Initialize model.
    llm = get_chat_model()

    # Create prompt.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Answer concisely.",
            ),
            ("human", "{question}"),
        ]
    )

    # Create chain.
    chain = prompt | llm | StrOutputParser()

    # Initialize callback.
    callback = LoggingCallbackHandler()

    # Execute chain with callback.
    response = chain.invoke(
        {
            "question": "What is LangChain?"
        },
        config={
            "callbacks": [callback]
        },
    )

    print("\nAssistant Response:")
    print(response)

    print("\nCustom callback test completed!")
    print("Check the logs/langchain_callbacks.log file.")


if __name__ == "__main__":
    test_custom_callback()
