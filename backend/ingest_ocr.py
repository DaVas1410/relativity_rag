import os
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = os.path.join("..", "data")
VECTOR_STORE_DIR = os.path.join("..", "vector_store_python")


def process_pdf_with_ocr(pdf_path):
    print(f"  -> Running OCR (Optical Character Recognition) on {pdf_path}...")
    pages = convert_from_path(pdf_path)
    text_content = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        text_content += text + "\n\n"
    return text_content


def process_pdf_native_text(pdf_path):
    print(f"  -> Extracting digital text directly from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text_content = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content += page_text + "\n\n"
    return text_content


def extract_text_smartly(pdf_path):
    """
    Decides whether to use native PDF text extraction (for books)
    or OCR (for scanned notes/whiteboards).
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    if num_pages == 0:
        return ""

    # Sample the first few pages to see if it has selectable digital text
    sample_pages = min(3, num_pages)
    total_chars = 0

    for i in range(sample_pages):
        page_text = reader.pages[i].extract_text()
        if page_text:
            total_chars += len(page_text.strip())

    # If the average characters per sampled page is very low, it's likely a scan/image
    avg_chars_per_page = total_chars / sample_pages

    if avg_chars_per_page < 50:
        print(f"Detected scanned document/images in {pdf_path}. Falling back to OCR.")
        return process_pdf_with_ocr(pdf_path)
    else:
        print(f"Detected digital text in {pdf_path}. Using fast text extraction.")
        return process_pdf_native_text(pdf_path)


def ingest():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return

    all_docs = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            print(f"\nProcessing {filename}...")

            raw_text = extract_text_smartly(pdf_path)

            if not raw_text.strip():
                print(f"Warning: Could not extract text from {filename}. Skipping.")
                continue

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(raw_text)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk, metadata={"source": filename, "chunk": i}
                )
                all_docs.append(doc)

    if not all_docs:
        print("No documents found to process.")
        return

    print(f"\nTotal chunks generated: {len(all_docs)}")
    print("Embedding documents into vector store...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store in ChromaDB
    Chroma.from_documents(
        documents=all_docs, embedding=embeddings, persist_directory=VECTOR_STORE_DIR
    )
    print("Ingestion complete. Vector store saved.")


if __name__ == "__main__":
    ingest()
