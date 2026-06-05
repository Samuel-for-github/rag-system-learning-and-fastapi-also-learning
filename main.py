# main.py

import os
import uuid
import tempfile
from pathlib import Path
from typing import List

import chromadb
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
load_dotenv()
app = FastAPI(title="RAG API")

# ==========================================================
# EMBEDDING MANAGER
# ==========================================================

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, texts):
        return self.model.encode(texts).tolist()


embedding_manager = EmbeddingManager()

# ==========================================================
# CHROMA DB
# ==========================================================

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

collection = client.get_or_create_collection(
    name="pdf_documents"
)

# ==========================================================
# PDF PROCESSING
# ==========================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


# ==========================================================
# REQUEST MODELS
# ==========================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/")
def root():
    return {"message": "RAG API Running go to /docs for Swagger UI"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents": collection.count()
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

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

        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))

            metadata = dict(chunk.metadata)
            metadata["source_file"] = file.filename

            metadatas.append(metadata)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return {
            "message": "PDF indexed successfully",
            "chunks": len(texts)
        }

    finally:
        os.remove(pdf_path)


@app.post("/query")
def query_documents(request: QueryRequest):

    query_embedding = embedding_manager.generate_embeddings(
        [request.query]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k
    )

    response = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(
        docs,
        metas,
        distances
    ):
        response.append({
            "content": doc,
            "metadata": meta,
            "similarity": round(1 - distance, 4)
        })

    return {
        "query": request.query,
        "results": response
    }