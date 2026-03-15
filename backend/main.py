from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio
from sklearn.decomposition import PCA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


VECTOR_STORE_DIR = os.path.join("..", "vector_store_python")

# Load embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Setup LLM - lower temperature for higher mathematical rigor
llm = ChatOllama(model="mistral", temperature=0.1)

# Build Prompt - Demanding extreme mathematical rigor
template = """You are an expert Professor of General Relativity, Differential Geometry, and Advanced Mathematics.
Answer the student's question based ONLY on the following context retrieved from their materials.
CRITICAL INSTRUCTIONS:
1. Provide HIGHLY RIGOROUS mathematical definitions. 
2. Use exact mathematical notation, formal definitions (e.g., topological spaces, coordinate charts, atlases, smooth manifolds, transition maps, tensors, metrics). 
3. DO NOT give purely qualitative or simplified analogies without the strict rigorous math behind it. Treat the user like a graduate physics student.
4. Use clear markdown and LaTeX formatting for all math equations. Inline math MUST use $ (e.g., $M$, $g_{{\\mu\\nu}}$). Block math MUST use $$ (e.g., $$ds^2 = g_{{\\mu\\nu}} dx^\\mu dx^\\nu$$).

Context:
{context}

Question:
{input}

Rigorous Mathematical Answer:"""
prompt = PromptTemplate.from_template(template)

# Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def generate():
        try:
            # We use astream to stream both the context retrieved and the generated text
            async for chunk in retrieval_chain.astream({"input": req.message}):
                if "context" in chunk:
                    # Send the retrieved documents to the frontend for the RAG visualization
                    docs = [
                        {
                            "content": d.page_content[:200] + "...",  # Send a snippet
                            "source": d.metadata.get("source", "Unknown"),
                        }
                        for d in chunk["context"]
                    ]
                    yield f"data: {json.dumps({'type': 'context', 'data': docs})}\n\n"

                if "answer" in chunk:
                    # Stream the LLM tokens
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk['answer']})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/vector-space")
async def get_vector_space():
    try:
        # Extract all documents, metadata, and embeddings from the Chroma DB collection
        data = vector_store._collection.get(
            include=["embeddings", "documents", "metadatas"]
        )

        embeddings = data.get("embeddings")
        documents = data.get("documents")
        metadatas = data.get("metadatas")

        if embeddings is None or len(embeddings) < 2:
            return {"points": []}

        # Use PCA to reduce high-dimensional embeddings down to 3D
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings)

        points = []
        for i, (x, y, z) in enumerate(reduced_embeddings):
            # Shorten the document text for the tooltip
            doc_text = documents[i] if documents and documents[i] else ""
            short_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text

            source = (
                metadatas[i].get("source", "Unknown")
                if metadatas and metadatas[i]
                else "Unknown"
            )

            points.append(
                {
                    "id": i,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "text": short_text,
                    "source": source,
                }
            )

        return {"points": points}
    except Exception as e:
        print(f"Error fetching vector space: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
