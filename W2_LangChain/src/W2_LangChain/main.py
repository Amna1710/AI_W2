import json
from pathlib import Path
from pprint import pprint

from W2_LangChain.chains.qa_chain import create_qa_chain
from W2_LangChain.chains.structured_chain import (
    create_article_summary_chain,
    create_json_chain,
    create_list_chain,
)
from W2_LangChain.pipelines import process_file
from W2_LangChain.memory import ConversationManager
from W2_LangChain.models.model_factory import (
    set_active_provider,
    get_active_provider,
)

def print_separator():
    print("\n" + "=" * 60)


# ============================================================
# 1. NORMAL QUESTION ANSWERING
# ============================================================

def ask_question():
    """Run the normal question-answering chain with streaming."""

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nGenerating answer...\n")
    print("AI Response:\n")

    chain = create_qa_chain()

    try:
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)

        print("\n")

    except Exception as error:
        print(f"\nError generating response: {error}")


# ============================================================
# 2. ARTICLE SUMMARY
# ============================================================

def generate_article_summary():
    """Generate a structured article summary."""

    article = input("\nPaste your article text: ").strip()

    if not article:
        print("Please provide article text.")
        return

    print("\nGenerating article summary...")

    chain = create_article_summary_chain()
    response = chain.invoke({"article": article})

    print("\nArticle Summary:")
    pprint(response)


# ============================================================
# 3. JSON GENERATION
# ============================================================

def generate_json():
    """Generate a JSON response."""

    question = input("\nEnter your request: ").strip()

    if not question:
        print("Please enter a request.")
        return

    print("\nGenerating JSON...")

    chain = create_json_chain()
    response = chain.invoke({"question": question})

    print("\nJSON Response:")
    print(json.dumps(response, indent=4, ensure_ascii=False))


# ============================================================
# 4. LIST GENERATION
# ============================================================

def generate_list():
    """Generate a comma-separated list."""

    text = input("\nEnter a topic: ").strip()

    if not text:
        print("Please enter a topic.")
        return

    print("\nGenerating list...")

    chain = create_list_chain()
    response = chain.invoke({"text": text})

    print("\nGenerated List:")

    for index, item in enumerate(response, start=1):
        print(f"{index}. {item}")


# ============================================================
# 5. DOCUMENT PROCESSING
# ============================================================

def process_document():
    """Load, enrich, and split a document."""

    file_path = input("\nEnter document path: ").strip().strip('"')

    if not file_path:
        print("Please provide a document path.")
        return

    path = Path(file_path).expanduser()

    if not path.is_file():
        print(f"File not found: {path}")
        return

    splitter_type = input(
        "\nSelect splitter (recursive/character/token) [recursive]: "
    ).strip().lower()

    if not splitter_type:
        splitter_type = "recursive"

    try:
        chunk_size = int(
            input("Enter chunk size [1000]: ").strip() or "1000"
        )

        chunk_overlap = int(
            input("Enter chunk overlap [150]: ").strip() or "150"
        )

        print("\nProcessing document...")

        chunks = process_file(
            file_path=path,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        print("\nDocument processed successfully!")
        print(f"Total chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print_separator()
            print(f"CHUNK {index}")
            print_separator()

            print("\nContent:")
            print(chunk.page_content)

            print("\nMetadata:")
            pprint(chunk.metadata)

    except ValueError as error:
        print(f"Invalid input: {error}")


# ============================================================
# 6. CONVERSATION MEMORY
# ============================================================

def select_memory_type():
    """Allow the user to select a memory strategy."""

    print_separator()
    print("SELECT CONVERSATION MEMORY")
    print_separator()

    print("1. Buffer Memory")
    print("2. Summary Memory")
    print("3. Vector Memory")
    print("4. Entity Memory")

    choices = {
        "1": "buffer",
        "2": "summary",
        "3": "vector",
        "4": "entity",
    }

    while True:
        choice = input("\nSelect memory type (1-4): ").strip()

        if choice in choices:
            return choices[choice]

        print("Invalid choice. Please select 1, 2, 3, or 4.")


def show_conversation_status(manager):
    """Display information about the selected memory."""

    print_separator()
    print("CONVERSATION MEMORY STATUS")
    print_separator()

    pprint(manager.get_status(), sort_dicts=False)


def run_conversation():
    """Run a multi-turn conversation using the selected memory."""

    memory_type = select_memory_type()

    print(f"\nInitializing {memory_type.title()} Memory...")

    try:
        manager = ConversationManager(memory_type)

    except Exception as error:
        print(f"\nCould not initialize conversation memory: {error}")
        return

    print_separator()
    print(f"{memory_type.upper()} MEMORY CONVERSATION")
    print_separator()

    print("You can now start asking questions.")
    print("\nCommands:")
    print("  /status  - Show memory information")
    print("  /clear   - Clear conversation memory")
    print("  /exit    - End conversation")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        command = question.lower()

        if command == "/exit":
            print("\nConversation ended.")
            break

        elif command == "/status":
            show_conversation_status(manager)
            continue

        elif command == "/clear":
            manager.clear()
            print("\nConversation memory cleared.")
            continue

        try:
            print("\nAssistant: ", end="", flush=True)

            answer = manager.ask(question)

            print(answer)

        except Exception as error:
            print(f"\nError generating response: {error}")


def choose_llm_provider():
    """Ask the user which LLM provider to use."""

    providers = {
        "1": "openai",
        "2": "anthropic",
        "3": "ollama",
        "4": "gemini",
    }

    print("\n" + "=" * 55)
    print("SELECT YOUR LLM PROVIDER")
    print("=" * 55)

    print("1. OpenAI")
    print("2. Anthropic")
    print("3. Ollama")
    print("4. Google Gemini")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice in providers:
            provider = providers[choice]
            set_active_provider(provider)

            print(f"\nSelected LLM Provider: {provider.upper()}")
            return

        print("Invalid choice. Please select a number from 1 to 4.")



# ============================================================
# MAIN MENU
# ============================================================

def display_menu():
    """Display available project features."""

    print_separator()
    print("          W2 LANGCHAIN PROJECT")
    print_separator()

    print("1. Ask a Question (QA Chain)")
    print("2. Generate Article Summary")
    print("3. Generate JSON Response")
    print("4. Generate a List")
    print("5. Process a Document")
    print("6. Conversation Memory")
    print("0. Exit")

    print_separator()



def main():
    choose_llm_provider()

    """Main application entry point."""

    print("\nWelcome to W2 LangChain!")

    actions = {
        "1": ask_question,
        "2": generate_article_summary,
        "3": generate_json,
        "4": generate_list,
        "5": process_document,
        "6": run_conversation,
    }

    while True:
        display_menu()

        choice = input("Select an option: ").strip()

        if choice == "0":
            print("\nThank you for using W2 LangChain!")
            break

        action = actions.get(choice)

        if action is None:
            print("\nInvalid option. Please select a valid option.")
            continue

        try:
            action()

        except Exception as error:
            print(f"\nAn error occurred: {error}")

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
