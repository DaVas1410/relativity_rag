# General Relativity AI RAG Agent

A local, privacy-first Retrieval-Augmented Generation (RAG) AI agent designed specifically for studying graduate-level General Relativity. This system utilizes local Ollama models to process complex mathematical formalisms, render LaTeX beautifully, and provide a 3D interactive knowledge graph of your study materials.

## Features

- **Local & Private:** Fully local LLM execution using [Ollama](https://ollama.ai/) (`mistral` for text generation, `nomic-embed-text` for vector embeddings). Your data never leaves your machine.
- **Smart Document Ingestion:** Intelligently switches between OCR (via Tesseract) for scanned whiteboard notes and native text extraction for digital textbooks.
- **Rigorous Mathematical Reasoning:** Prompts are engineered to demand formalisms (topology, tensors, manifolds) and reject purely qualitative analogies.
- **Premium Dark-Mode UI:** A visually stunning Next.js interface featuring Framer Motion animations, local storage persistence, and beautiful LaTeX rendering (`react-markdown`, `rehype-katex`).
- **Interactive 3D Vector Space:** Explore your document chunks in a 3D scatter plot directly in the UI, powered by PCA dimensionality reduction on the backend and `react-plotly.js`.
- **Live RAG Context:** Real-time side-by-side view of the retrieved document contexts as the LLM streams its response.

## Architecture

- **Frontend:** Next.js (App Router), React, Tailwind CSS, Framer Motion, Plotly.js, KaTeX.
- **Backend:** Python, FastAPI, Langchain, ChromaDB, PyTesseract, pdf2image.
- **AI/ML:** Ollama (Mistral, Nomic Embeddings), PCA (scikit-learn).

## Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- [Python](https://www.python.org/) (3.10+)
- [Ollama](https://ollama.ai/) installed and running.
- Tesseract OCR installed on your system (`apt-get install tesseract-ocr` or `brew install tesseract`).
- Poppler installed (for `pdf2image`, `apt-get install poppler-utils` or `brew install poppler`).

## Setup Instructions

### 1. Pull Local Models
Ensure Ollama is running, then pull the required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Backend Setup
Navigate to the `backend` directory and install the Python dependencies:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Place your PDFs and scanned notes into the `data/` directory (created at the root or within backend, based on your config).

Run the ingestion script to process your documents and build the ChromaDB vector store:
```bash
python ingest_ocr.py
```

Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
Open a new terminal, navigate to the project root, and install Node dependencies:
```bash
npm install
```

Start the Next.js development server:
```bash
npm run dev
```

Visit `http://localhost:3000` in your browser.

## Data Privacy
The `.gitignore` is comprehensively configured to block the `/data` directory, the `/vector_store_python` ChromaDB directory, and any generated `/metrics`. Your personal study materials will never be tracked or uploaded to version control.
