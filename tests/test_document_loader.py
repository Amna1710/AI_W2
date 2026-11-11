from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfWriter

from rag_system.document_loaders import DocumentLoaderFactory


TEST_DIR = Path("data/processed/loader_tests")


def setup_test_files():

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # TXT file
    (TEST_DIR / "sample.txt").write_text(
        "This is a sample text document.",
        encoding="utf-8",
    )

    # Markdown file
    (TEST_DIR / "sample.md").write_text(
        "# RAG Project\n\nThis is a Markdown document.",
        encoding="utf-8",
    )

    # DOCX file
    doc = DocxDocument()
    doc.add_paragraph("This is a sample DOCX document.")
    doc.save(TEST_DIR / "sample.docx")

    # PDF file
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(TEST_DIR / "sample.pdf")


def test_document_loaders():

    setup_test_files()

    extensions = [".txt", ".md", ".docx", ".pdf"]

    for extension in extensions:

        file_path = TEST_DIR / f"sample{extension}"

        documents = DocumentLoaderFactory.load_document(file_path)

        print(f"\nTesting {extension}")
        print(f"Documents extracted: {len(documents)}")

        for document in documents:
            print("Metadata:", document.metadata)
            print("Content:", document.page_content[:200])

    print("\nDocument loader tests completed!")


if __name__ == "__main__":
    test_document_loaders()