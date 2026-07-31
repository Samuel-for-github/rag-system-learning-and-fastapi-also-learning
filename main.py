import os
import uuid
import tempfile
from typing import List, Dict, Any, Optional
import time
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.messages import HumanMessage
# from sentence_transformers import SentenceTransformer
from charset_normalizer import from_path
import requests
load_dotenv()

app = FastAPI(title="RAG API")
from fastapi.middleware.cors import CORSMiddleware



origins = [
    "http://localhost:3000",  # Next.js
    "http://127.0.0.1:3000",
    "https://rag-project-frontend-indol.vercel.app",
    # Add your production frontend URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")


endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2,
    huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
)


llm = ChatHuggingFace(llm=endpoint)

# File types this API can ingest
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls"}

# The embedding dimension for sentence-transformers/all-MiniLM-L6-v2.
# Pinecone needs this up front to create the index.
EMBEDDING_DIMENSION = 384

# Pinecone has a hard 40KB metadata-per-vector limit, and the document text
# has to live in metadata (there's no separate "documents" field like Chroma).
# Truncate stored text defensively so a single huge chunk can't blow the limit.
MAX_METADATA_TEXT_CHARS = 30000


# ==========================================================
# EMBEDDING MANAGER
# ==========================================================

class EmbeddingManager:
    """Handles document embedding generation via the Hugging Face Inference API
    (no local model loaded — keeps the deployment lightweight)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.hf_token = os.getenv("HF_API_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_API_TOKEN environment variable is not set")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts via HF's hosted inference API.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} texts via HF Inference API...")

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HF Inference API error {response.status_code}: {response.text}"
            )

        embeddings = np.array(response.json())
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
embedding_manager = EmbeddingManager()
# ==========================================================
# VECTOR STORE
# ==========================================================
class VectorStore:
    """Manages document embeddings in a Pinecone vector store"""

    def __init__(self, index_name: str = "appbuddy"):
        """
        Initialize the vector store

        Args:
            index_name: Name of the Pinecone index. Pinecone index names must
                be lowercase alphanumeric + hyphens only (no underscores).
        """
        self.index_name = index_name

        self.client = None
        self.index = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize Pinecone client and index"""
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is not set")

            self.client = Pinecone(api_key=api_key)

            existing_indexes = [idx["name"] for idx in self.client.list_indexes()]

            if self.index_name not in existing_indexes:
                print(f"Index '{self.index_name}' not found, creating it...")
                self.client.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD", "aws"),
                        region=os.getenv("PINECONE_REGION", "us-east-1"),
                    ),
                )
                # Wait for the index to be ready before using it
                while not self.client.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)

            self.index = self.client.Index(self.index_name)

            print(f"Vector store initialized. Index: {self.index_name}")
            stats = self.index.describe_index_stats()
            print(f"Existing vectors in index: {stats.get('total_vector_count', 0)}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def count(self) -> int:
        """Return the current number of vectors stored in the index."""
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        vectors = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"

            # Prepare metadata. Pinecone metadata values must be strings,
            # numbers, booleans, or lists of strings — no nested dicts.
            metadata = {
                str(k): v for k, v in doc.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            # Pinecone has no separate "documents" store, so the text itself
            # has to be carried in metadata to be retrievable later.
            metadata["text"] = doc.page_content[:MAX_METADATA_TEXT_CHARS]

            vectors.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": metadata,
            })

        # Add to index. Pinecone recommends batching upserts (~100 per batch)
        # to stay comfortably under request size limits.
        try:
            batch_size = 100
            for start in range(0, len(vectors), batch_size):
                batch = vectors[start:start + batch_size]
                self.index.upsert(vectors=batch)

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total vectors in index: {self.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore=VectorStore()

# ==========================================================
# RAG PIPELINE
# ==========================================================


class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.4,
        stream: bool = False,
        summarize: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=min_score,
            metadata_filter=metadata_filter,
        )

        # FIX: debug visibility into exactly what's being handed to the LLM,
        # so it's obvious when weak/irrelevant chunks are the real cause of a
        # bad answer rather than the prompt itself.
        print(f"\n--- Retrieved {len(results)} chunk(s) passed to the LLM ---")
        for doc in results:
            print(f"  score={doc['similarity_score']:.4f} source={doc['metadata'].get('source_file', 'unknown')}")
            print(f"  preview: {doc['content'][:150]!r}")
        print("--- end retrieved chunks ---\n")

        if not results:
            answer = "I don't have information about this in the provided documents."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', doc['metadata'].get('row', 'unknown')),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]

            # FIX: strict grounding prompt — the old version only *suggested*
            # using the context, which left the LLM free to blend in outside
            # knowledge or invent details not actually retrieved.
            prompt = f"""You are a document assistant that answers questions using ONLY the context provided below.

Rules:
- Use ONLY the information in the context to answer. Do NOT use outside knowledge, even if you know the answer.
- Do NOT guess, infer, or add details that are not explicitly present in the context.
- If the context does not contain enough information to answer, respond exactly with:
  "I don't have information about this in the provided documents."
- Keep the answer concise and grounded strictly in the context.

Context:
{context}

Question: {question}

Answer:"""

            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()

            # FIX: previously this called prompt.format(context=context, question=question)
            # on a string that was already an f-string with those values filled in.
            # If any retrieved chunk contained a literal "{" or "}" (JSON, notes, etc.),
            # that second .format() call would raise or silently corrupt the prompt.
            response = self.llm.invoke([prompt])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'citations': citations,
            'summary': summary,
            'history': self.history
        }


# ==========================================================
# RAG RETRIEVER
# ==========================================================

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            metadata_filter: Optional Pinecone metadata filter, e.g.
                {"taluka": "Tiswadi"} or {"source_file": "goa_places.csv"} —
                lets a caller scope retrieval to a subset of ingested docs
                instead of always searching the whole index.

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}, Filter: {metadata_filter}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            query_kwargs = {
                "vector": query_embedding.tolist(),
                "top_k": top_k,
                "include_metadata": True,
            }
            # FIX: only attach a filter when one is actually supplied, so
            # existing callers that don't care about scoping are unaffected.
            if metadata_filter:
                query_kwargs["filter"] = metadata_filter

            results = self.vector_store.index.query(**query_kwargs)

            # Process results
            retrieved_docs = []

            matches = results.get("matches", [])

            if matches:
                for i, match in enumerate(matches):
                    metadata = dict(match.get("metadata", {}))
                    # Pinecone returns cosine *similarity* directly as `score`
                    # (unlike Chroma, which returns distance), so no 1 - x needed.
                    similarity_score = match.get("score", 0.0)
                    document_text = metadata.pop("text", "")

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': match.get("id"),
                            'content': document_text,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': 1 - similarity_score,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)

# Wire up the pipeline now that the index is ready
rag_pipeline = AdvancedRAGPipeline(rag_retriever,llm)


# ==========================================================
# FILE LOADING (PDF / CSV / XLSX)
# ==========================================================

def load_pdf(path: str, filename: str) -> List[Document]:
    """Load a PDF file into LangChain Documents (one per page)."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = filename
        doc.metadata["file_type"] = "pdf"
    return docs


def load_csv(path: str, filename: str) -> List[Document]:
    """Load a CSV file into LangChain Documents (one per row)."""
    # Detect the file's actual encoding instead of relying on the
    # OS default (cp1252 on Windows), which breaks on non-ASCII bytes.
    detected = from_path(path).best()
    encoding = detected.encoding if detected else "utf-8"

    try:
        loader = CSVLoader(file_path=path, encoding=encoding)
        docs = loader.load()
    except UnicodeDecodeError:
        # Last-resort fallback: latin-1 maps every byte 0-255,
        # so it will never raise, even if it's not a perfect guess.
        loader = CSVLoader(file_path=path, encoding="latin-1")
        docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata["source_file"] = filename
        doc.metadata["file_type"] = "csv"
        doc.metadata["row"] = i
    return docs


def load_excel(path: str, filename: str) -> List[Document]:
    """
    Load an Excel workbook into LangChain Documents.
    Every sheet is read, and each row becomes its own Document so that
    retrieval can point back to a specific sheet/row.
    """
    docs: List[Document] = []
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read Excel file '{filename}': {e}")

    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df = df.fillna("")
        columns = [str(c) for c in df.columns]

        for row_idx, row in df.iterrows():
            row_text = "\n".join(f"{col}: {row[col]}" for col in columns)
            content = f"Sheet: {sheet_name}\n{row_text}"
            metadata = {
                "source_file": filename,
                "file_type": "xlsx",
                "sheet": sheet_name,
                "row": int(row_idx),
            }
            docs.append(Document(page_content=content, metadata=metadata))

    return docs


def load_file(path: str, filename: str) -> List[Document]:
    """Dispatch to the correct loader based on file extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return load_pdf(path, filename)
    elif ext == ".csv":
        return load_csv(path, filename)
    elif ext in (".xlsx", ".xls"):
        return load_excel(path, filename)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed types: {sorted(SUPPORTED_EXTENSIONS)}"
        )


# FIX: chunk_size dropped from 10000 -> 800 (chars). At 10k chars a single
# chunk could span multiple unrelated topics (beaches + hotels + temples in
# one blob), which drags irrelevant text into every retrieval that happens
# to match any part of it. 800/100 keeps each chunk topically tight while
# still giving RecursiveCharacterTextSplitter enough room to break on
# paragraph/sentence boundaries first.
def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Split documents into smaller chunks for better RAG performance.

    Row-based documents (CSV/XLSX) are already small, so this is mainly
    relevant for PDF text, but it's safe to run on all document types —
    anything under chunk_size passes through untouched.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs


# ==========================================================
# REQUEST MODELS
# ==========================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    # FIX: raised from 0.2 -> 0.4. 0.2 cosine similarity is close to
    # "barely related" for MiniLM embeddings and let near-irrelevant chunks
    # into the context window. Tune this per your embedding model/data if
    # 0.4 is too strict or too loose for your corpus.
    min_score: float = 0.4
    summarize: bool = False
    # FIX: generic metadata filter (e.g. {"taluka": "Tiswadi"},
    # {"source_file": "goa_places.csv"}) so a query can be scoped to a
    # subset of the index instead of always searching everything.
    filter: Optional[Dict[str, Any]] = None


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/")
def root():
    return {"message": "RAG API Running — go to /docs for Swagger UI"}


@app.get("/health")
def health():
    return {"status": "healthy", "documents": vectorstore.count()}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name

    try:
        docs = load_file(file_path, file.filename)

        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the file")

        chunks = split_documents(docs)

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        vectorstore.add_documents(chunks, embeddings)

        return {
            "message": f"{ext.lstrip('.').upper()} file indexed successfully",
            "filename": file.filename,
            "chunks": len(texts),
        }

    finally:
        os.remove(file_path)


@app.post("/query")
def query_documents(request: QueryRequest):
    # Route through the RAG pipeline instead of hitting Pinecone directly
    return rag_pipeline.query(
        question=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        summarize=request.summarize,
        metadata_filter=request.filter,
    )
