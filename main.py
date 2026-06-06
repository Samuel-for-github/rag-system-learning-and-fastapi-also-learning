import os
import uuid
import tempfile
from typing import List, Dict, Any
import time

import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="RAG API")

from langchain_openrouter import ChatOpenRouter

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

llm = ChatOpenRouter(
    api_key=openrouter_api_key,
    model="liquid/lfm-2.5-1.2b-instruct:free",
    temperature=0.1,
    max_tokens=1000,
)


# ==========================================================
# EMBEDDING MANAGER
# ==========================================================

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


embedding_manager = EmbeddingManager()


# ==========================================================
# RAG PIPELINE
# ==========================================================

class AdvancedRAGPipeline:
    def __init__(self, collection, embedding_manager, llm):
        self.collection = collection
        self.embedding_manager = embedding_manager
        self.llm = llm
        self.history: List[Dict[str, Any]] = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        query_embedding = self.embedding_manager.generate_embeddings([question])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # Filter by minimum similarity score
        filtered = [
            (doc, meta, round(1 - dist, 4))
            for doc, meta, dist in zip(docs, metas, distances)
            if (1 - dist) >= min_score
        ]

        if not filtered:
            answer = "No relevant context found."
            sources = []
        else:
            context = "\n\n".join(doc for doc, _, _ in filtered)
            sources = [
                {
                    "source": meta.get("source_file", meta.get("source", "unknown")),
                    "page": meta.get("page", "unknown"),
                    "score": score,
                    "preview": doc[:120] + "...",
                }
                for doc, meta, score in filtered
            ]

            prompt = (
                f"Use the following context to answer the question concisely.\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            # FIX: pass a HumanMessage, not a raw string
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        # answer_with_citations = (
        #     answer + "\n\nCitations:\n" + "\n".join(citations)
        #     if citations
        #     else answer
        # )

        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_resp.content

        self.history.append(
            {"question": question, "answer": answer, "sources": sources, "summary": summary}
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }


# ==========================================================
# CHROMA DB
# ==========================================================

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

collection = client.get_or_create_collection(name="pdf_documents")

# Wire up the pipeline now that collection is ready
rag_pipeline = AdvancedRAGPipeline(
    collection=collection,
    embedding_manager=embedding_manager,
    llm=llm,
)


# ==========================================================
# PDF PROCESSING
# ==========================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


# ==========================================================
# REQUEST MODELS
# ==========================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.2
    summarize: bool = False


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/")
def root():
    return {"message": "RAG API Running — go to /docs for Swagger UI"}


@app.get("/health")
def health():
    return {"status": "healthy", "documents": collection.count()}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = split_documents(docs)

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        ids = []
        metadatas = []
        for chunk in chunks:
            ids.append(str(uuid.uuid4()))
            metadata = dict(chunk.metadata)
            metadata["source_file"] = file.filename
            metadatas.append(metadata)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return {"message": "PDF indexed successfully", "chunks": len(texts)}

    finally:
        os.remove(pdf_path)


@app.post("/query")
def query_documents(request: QueryRequest):
    # FIX: route through the RAG pipeline instead of hitting ChromaDB directly
    return rag_pipeline.query(
        question=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        summarize=request.summarize,
    )