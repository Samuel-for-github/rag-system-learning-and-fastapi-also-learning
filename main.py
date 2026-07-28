import os
import uuid
import tempfile
from typing import List, Dict, Any
import time
import numpy as np
import pandas as pd
import chromadb
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

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

llm = ChatOpenRouter(
    api_key=openrouter_api_key,
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature=0.1,
    max_tokens=1000,
)

# File types this API can ingest
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls"}


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
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents"):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name

        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            # os.makedirs(self.persist_directory, exist_ok=True)
            # self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_API_KEY"),
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG",
                    "hnsw:space": "cosine"
                }
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

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

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

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

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            print(context)
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', doc['metadata'].get('row', 'unknown')),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        # answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

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

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        # print(f"distance={distance}, similarity_score={similarity_score}")
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
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

# collection = client.get_or_create_collection(name="pdf_documents")

# Wire up the pipeline now that collection is ready
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


def split_documents(documents, chunk_size=10000, chunk_overlap=200):
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
    return {"status": "healthy", "documents":vectorstore.collection.count()}


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
    # FIX: route through the RAG pipeline instead of hitting ChromaDB directly
    return rag_pipeline.query(
        question=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        summarize=request.summarize,
    )