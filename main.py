import os
import re
import uuid
import tempfile
import hashlib
import json
import threading
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
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
from langchain_openai import ChatOpenAI

# --- LLM: vLLM served from a RunPod pod (OpenAI-compatible server) ---
# vLLM's `vllm/vllm-openai` image exposes /v1/chat/completions, so a plain
# ChatOpenAI client pointed at your pod's proxy URL works as a drop-in
# replacement for the old HuggingFaceEndpoint/ChatHuggingFace setup — no
# other code in this file (AdvancedRAGPipeline, etc.) needs to change since
# they only call llm.invoke(...).
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
VLLM_MODEL_NAME = os.getenv(
    "VLLM_MODEL_NAME",
    "Qwen/Qwen2.5-1.5B-Instruct"
)
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

if not VLLM_BASE_URL:
    raise ValueError("VLLM_BASE_URL environment variable is not set")

if not VLLM_API_KEY:
    raise ValueError("VLLM_API_KEY environment variable is not set")

llm = ChatOpenAI(
    base_url=VLLM_BASE_URL.rstrip("/"),
    api_key=VLLM_API_KEY,
    model=VLLM_MODEL_NAME,
    temperature=0.2,
    max_tokens=512,
)
# File types this API can ingest
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls"}

# The embedding dimension for sentence-transformers/all-MiniLM-L6-v2.
# Pinecone needs this up front to create the index.
EMBEDDING_DIMENSION = 384

# Pinecone has a hard 40KB metadata-per-vector limit, and the document text
# has to live in metadata (there's no separate "documents" field like Chroma).
# Truncate stored text defensively so a single huge chunk can't blow the limit.
MAX_METADATA_TEXT_CHARS = 30000

# Pinecone metadata values must be strings, numbers, or booleans (no nested
# dicts/lists of non-strings). Cap how long a string metadata value can be so
# one huge cell (e.g. a giant "description" column) can't blow the 40KB
# per-vector metadata limit on its own.
MAX_METADATA_FIELD_CHARS = 500

# Goa's 12 talukas — a fixed, known list, so matching against it directly is
# far more reliable than trying to generically extract a "location" from
# free text. Used to auto-detect a taluka mentioned in a user's question and
# scope retrieval to it without the caller having to build a filter by hand.
GOA_TALUKAS = [
    "tiswadi", "bardez", "salcete", "mormugao", "ponda", "bicholim",
    "sanguem", "quepem", "canacona", "pernem", "sattari", "dharbandora",
]

# Approximate center point (lat, lng) for each of Goa's 12 talukas. Used to
# reverse-geocode a user's live GPS coordinates to the nearest taluka via
# simple nearest-centroid matching (see nearest_taluka below) — no external
# geocoding API needed, and no taluka boundary/polygon data required.
#
# NOTE: these are approximate town/administrative-center coordinates, not
# precise taluka boundary centroids, so results near a taluka border can be
# coarse (e.g. a point right on the Bardez/Bicholim line might resolve to
# either neighbor). Good enough for a "which taluka am I probably in / near"
# signal; swap in real boundary polygons + point-in-polygon matching later
# if finer accuracy is needed.
GOA_TALUKA_CENTROIDS: Dict[str, tuple] = {
    "tiswadi": (15.4909, 73.8278),      # Panaji
    "bardez": (15.5937, 73.8142),       # Mapusa
    "salcete": (15.2832, 73.9862),      # Margao
    "mormugao": (15.3960, 73.8157),     # Vasco da Gama
    "ponda": (15.4027, 74.0078),        # Ponda town
    "bicholim": (15.5936, 73.9490),     # Bicholim town
    "sanguem": (15.2214, 74.1636),      # Sanguem town
    "quepem": (15.2141, 74.0797),       # Quepem town
    "canacona": (15.0100, 74.0450),     # Canacona/Chaudi
    "pernem": (15.7167, 73.7970),       # Pernem town
    "sattari": (15.5833, 74.1167),      # Valpoi
    "dharbandora": (15.3833, 74.1000),  # Dharbandora town
}


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two lat/lng points."""
    r = 6371.0  # Earth radius, km
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _nearest_taluka_by_centroid(lat: float, lng: float) -> Optional[str]:
    """
    Fallback reverse-geocode: nearest taluka by straight-line distance to
    each taluka's approximate center point (GOA_TALUKA_CENTROIDS). Coarser
    than real polygon matching (see taluka_from_point below) — used only
    when a real boundary isn't available (Dharbandora, see
    TALUKA_POLYGONS_MISSING) or as a small-radius "just outside the
    polygon" rescue for points that miss every real boundary narrowly
    (GPS error, coastline rounding, etc).
    """
    best_taluka = None
    best_distance = float("inf")
    for taluka, (t_lat, t_lng) in GOA_TALUKA_CENTROIDS.items():
        dist = _haversine_km(lat, lng, t_lat, t_lng)
        if dist < best_distance:
            best_distance = dist
            best_taluka = taluka

    if best_taluka is None or best_distance > 60:
        return None
    return best_taluka


# ---- Real taluka boundary polygons (point-in-polygon reverse geocoding) ----
#
# Loaded from goa_talukas.geojson, which was built from an all-India GADM
# taluk boundary file, filtered to Goa's 12 talukas, and normalized so each
# feature's "taluka" property matches the lowercase names already used in
# Pinecone metadata (see _row_metadata). One caveat carried over from that
# source data: Dharbandora taluka (carved out of Ponda/Sanguem in 1996) has
# no separate polygon in this dataset's boundary vintage, so it falls back
# to centroid-distance matching (_nearest_taluka_by_centroid) instead of a
# real polygon — see TALUKA_POLYGONS_MISSING below.
_GEOJSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goa_talukas.geojson")

TALUKA_POLYGONS: Dict[str, Any] = {}
TALUKA_POLYGONS_MISSING = {"dharbandora"}  # known gap — see comment above

try:
    from shapely.geometry import shape as _shapely_shape, Point as _shapely_Point
    from shapely.strtree import STRtree as _shapely_STRtree

    with open(_GEOJSON_PATH) as _f:
        _geojson = json.load(_f)

    TALUKA_POLYGONS = {
        feat["properties"]["taluka"]: _shapely_shape(feat["geometry"])
        for feat in _geojson["features"]
    }
    print(f"Loaded {len(TALUKA_POLYGONS)} taluka boundary polygons from {_GEOJSON_PATH}")
except FileNotFoundError:
    print(f"WARNING: {_GEOJSON_PATH} not found — falling back to centroid-only taluka detection for ALL talukas")
except ImportError:
    print("WARNING: shapely not installed — falling back to centroid-only taluka detection for ALL talukas")


def taluka_from_point(lat: float, lng: float, near_miss_buffer_km: float = 1.0) -> Optional[str]:
    """
    Reverse-geocode a lat/lng pair to a Goa taluka using real polygon
    boundaries when available (point-in-polygon — exact, handles irregular
    borders correctly), falling back to nearest-centroid matching when a
    polygon isn't available (Dharbandora, or if the boundary file/shapely
    failed to load).

    Also rescues "near misses": GPS readings have real-world error, and a
    point sitting right on a coastline or a taluka border can fall just
    outside every polygon even though the person is clearly in that taluka.
    If no polygon contains the point outright, this checks whether the
    point is within `near_miss_buffer_km` of any polygon's edge before
    giving up and falling back to centroid matching.

    Returns None if the point is nowhere near Goa at all.
    """
    if TALUKA_POLYGONS:
        point = _shapely_Point(lng, lat)  # GeoJSON/shapely order: (lng, lat)

        # Exact match: point genuinely falls inside a taluka's boundary.
        for taluka, polygon in TALUKA_POLYGONS.items():
            if polygon.contains(point):
                return taluka

        # Near miss: just outside every polygon (GPS error, coastline,
        # a border that's fuzzy at this resolution). Roughly convert the
        # buffer from km to degrees (1 deg latitude ~= 111km) — approximate,
        # but fine for a small rescue radius like this.
        buffer_deg = near_miss_buffer_km / 111.0
        best_taluka, best_dist = None, float("inf")
        for taluka, polygon in TALUKA_POLYGONS.items():
            dist = polygon.distance(point)
            if dist < best_dist:
                best_dist, best_taluka = dist, taluka
        if best_taluka is not None and best_dist <= buffer_deg:
            return best_taluka

    # No real polygons loaded, or the point missed all of them (including
    # the near-miss buffer) — e.g. Dharbandora, which has no polygon at
    # all, or a point genuinely outside Goa. Centroid matching still
    # enforces a 60km-from-Goa sanity cutoff, so this won't force a match
    # for e.g. a coordinate in another city.
    return _nearest_taluka_by_centroid(lat, lng)


# Backwards-compatible alias — existing callers (AdvancedRAGPipeline.query,
# the /location/taluka route) use this name; it now does real polygon
# matching instead of pure centroid-distance.
nearest_taluka = taluka_from_point

# ==========================================================
# SESSION MANAGEMENT (for follow-up / multi-turn chat)
# ==========================================================

# How many past turns (user+assistant pairs) get fed back into prompts —
# both the standalone-question condenser and the final answer prompt.
# Kept small since vLLM here is running a small (1.5B) model with a modest
# context window / max_tokens budget.
SESSION_HISTORY_TURNS = 6

# Idle sessions older than this get dropped on next access, so the
# in-memory store doesn't grow forever across a long-running process.
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours


class SessionStore:
    """
    Minimal in-memory chat history store, keyed by session_id.

    NOTE: this is process-local memory, not a database — history is lost on
    restart and isn't shared across multiple API instances/workers. That's
    fine for a single-process deployment; swap in Redis/Postgres if you
    scale out horizontally.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _expire_if_stale(self, session_id: str) -> None:
        entry = self._sessions.get(session_id)
        if entry and (time.time() - entry["last_active"] > SESSION_TTL_SECONDS):
            del self._sessions[session_id]

    def exists(self, session_id: str) -> bool:
        self._expire_if_stale(session_id)
        return session_id in self._sessions

    def touch(self, session_id: str) -> None:
        """Ensure a session entry exists and mark it as recently active."""
        self._expire_if_stale(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = {"messages": [], "last_active": time.time()}
        else:
            self._sessions[session_id]["last_active"] = time.time()

    def get_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        if not session_id:
            return []
        self._expire_if_stale(session_id)
        entry = self._sessions.get(session_id)
        return list(entry["messages"]) if entry else []

    def append(self, session_id: str, role: str, content: str) -> None:
        self.touch(session_id)
        self._sessions[session_id]["messages"].append({"role": role, "content": content})

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


session_store = SessionStore()


def format_history(history: List[Dict[str, str]], limit_turns: int = SESSION_HISTORY_TURNS) -> str:
    """Render the last `limit_turns` user/assistant pairs as plain text for prompting."""
    if not history:
        return ""
    # history is a flat list of alternating {"role": "user"/"assistant", ...}
    # entries; keep the most recent 2*limit_turns messages.
    recent = history[-(limit_turns * 2):]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def detect_taluka_filter(text: str) -> Optional[str]:
    """
    Look for a known Goa taluka name mentioned in free text (e.g. a user's
    question) and return it if found. Matches whole words only, case
    insensitive, so e.g. "ponda" won't false-positive inside an unrelated
    longer word. Returns the taluka lowercased (matching how it's stored in
    metadata) or None if no taluka is mentioned.
    """
    lowered = text.lower()
    for taluka in GOA_TALUKAS:
        if re.search(rf"\b{re.escape(taluka)}\b", lowered):
            return taluka
    return None


# Field name -> set of distinct (lowercased) values seen during ingestion.
# Populated automatically for "categorical" columns — see
# _register_categorical_columns — so a question's free text can be matched
# against real values from *your* dataset (category, sub_category, ...)
# instead of a hand-maintained list per field. Resets on process restart;
# it gets rebuilt as files are re-ingested via /upload.
FIELD_VALUE_REGISTRY: Dict[str, set] = defaultdict(set)

# Structural/bookkeeping fields that should never be auto-matched against —
# they aren't things a person would type into a question.
FIELD_REGISTRY_EXCLUDE = {
    "source_file", "file_type", "row", "sheet", "doc_index", "content_length",
}


def _register_categorical_columns(df: "pd.DataFrame", columns: List[str]) -> None:
    """
    Inspect a freshly-loaded CSV/Excel dataframe and record distinct values
    for any column that looks categorical (few unique values relative to
    row count) into FIELD_VALUE_REGISTRY.

    Heuristic: a column counts as categorical if it has at most 50 distinct
    non-empty values AND those values cover at most 20% of the row count —
    e.g. 12 talukas or ~15 categories across hundreds of rows qualifies;
    "place_name" or "description", which are close to unique per row,
    don't. This keeps the registry small and avoids matching on noisy,
    effectively-unique text.
    """
    n_rows = len(df)
    if n_rows == 0:
        return
    for col in columns:
        key = str(col).strip().lower().replace(" ", "_")
        if not key or key in FIELD_REGISTRY_EXCLUDE:
            continue
        series = df[col].astype(str).str.strip()
        series = series[series != ""]
        if series.empty:
            continue
        nunique = series.nunique()
        if nunique <= 50 and (nunique / n_rows) <= 0.2:
            for val in series.unique():
                FIELD_VALUE_REGISTRY[key].add(val.lower())


def detect_filters_from_text(text: str) -> Dict[str, str]:
    """
    Scan free text (e.g. a user's question) for values matching any
    registered categorical field (built from actually-ingested data — see
    _register_categorical_columns) and return a filter dict for whichever
    fields matched. Within a field, longer values are checked first so a
    more specific phrase (e.g. "beach / coastal") wins over a shorter one
    that happens to be a substring of it.
    """
    lowered = text.lower()
    detected: Dict[str, str] = {}
    for field, values in FIELD_VALUE_REGISTRY.items():
        for val in sorted(values, key=len, reverse=True):
            if re.search(rf"\b{re.escape(val)}\b", lowered):
                detected[field] = val
                break
    return detected


# ==========================================================
# CACHING
# ==========================================================
#
# Three independent in-memory caches, each sized/TTL'd for what it stores:
#
#   - embedding_cache: text -> embedding vector. Embeddings for a given
#     text+model never change, so this uses a long TTL. Saves a network
#     round-trip to the HF Inference API for repeated texts (very common
#     for query embeddings — popular questions repeat far more than
#     ingested document chunks do).
#   - retrieval_cache: (query, top_k, threshold, filter) -> retrieved docs.
#     Short TTL, and explicitly cleared whenever new documents are added,
#     since the correct answer to "what matches this filter" changes the
#     moment the index changes.
#   - llm_cache: exact prompt text -> generated text. Cleared alongside
#     retrieval_cache on upload, since a stale cached answer could be
#     grounded in context that's now outdated (a newer, better-matching
#     document was just ingested).
#
# All three are process-local (same caveat as SessionStore — swap in Redis
# etc. if you scale out horizontally).

EMBEDDING_CACHE_TTL_SECONDS = 60 * 60 * 6   # 6 hours
EMBEDDING_CACHE_MAX_SIZE = 5000

RETRIEVAL_CACHE_TTL_SECONDS = 60 * 5        # 5 minutes
RETRIEVAL_CACHE_MAX_SIZE = 500

LLM_CACHE_TTL_SECONDS = 60 * 30             # 30 minutes
LLM_CACHE_MAX_SIZE = 500


class TTLCache:
    """
    Thread-safe in-memory cache with a per-entry TTL and a max-size cap.

    Eviction once max_size is reached is FIFO (oldest-inserted key first)
    rather than strict LRU — simple, and good enough here since hit
    patterns are dominated by TTL expiry, not size pressure.
    """

    def __init__(self, ttl_seconds: float, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _is_expired(self, key: str) -> bool:
        ts = self._timestamps.get(key)
        return ts is None or (time.time() - ts) > self.ttl_seconds

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._store and not self._is_expired(key):
                self.hits += 1
                return self._store[key]
            if key in self._store:
                # Present but expired — drop it so it doesn't linger.
                del self._store[key]
                del self._timestamps[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_size and key not in self._store:
                oldest_key = next(iter(self._store), None)
                if oldest_key is not None:
                    self._store.pop(oldest_key, None)
                    self._timestamps.pop(oldest_key, None)
            self._store[key] = value
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._timestamps.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else None,
            }


def _cache_key(*parts: Any) -> str:
    """
    Build a stable cache key from arbitrary parts (strings, dicts, lists,
    numbers, ...) by JSON-serializing with sorted keys and hashing. Sorted
    keys mean e.g. {"a": 1, "b": 2} and {"b": 2, "a": 1} hash identically,
    and hashing keeps keys short/fixed-size regardless of how much text
    (prompts, context) goes into them.
    """
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


embedding_cache = TTLCache(ttl_seconds=EMBEDDING_CACHE_TTL_SECONDS, max_size=EMBEDDING_CACHE_MAX_SIZE)
retrieval_cache = TTLCache(ttl_seconds=RETRIEVAL_CACHE_TTL_SECONDS, max_size=RETRIEVAL_CACHE_MAX_SIZE)
llm_cache = TTLCache(ttl_seconds=LLM_CACHE_TTL_SECONDS, max_size=LLM_CACHE_MAX_SIZE)


def cached_llm_invoke(prompt: str) -> str:
    """
    Invoke the LLM with a cache keyed on the exact prompt text, so
    identical prompts (the same question hitting the same retrieved
    context, or a repeated follow-up condensation) skip a network
    round-trip to the vLLM pod entirely.

    Returns plain text (not the LangChain response object) since that's
    all every caller in this file actually uses.
    """
    key = _cache_key("llm", VLLM_MODEL_NAME, prompt)
    cached = llm_cache.get(key)
    if cached is not None:
        print("LLM cache hit")
        return cached

    response = llm.invoke([prompt])
    content = response.content
    llm_cache.set(key, content)
    return content


def invalidate_corpus_caches() -> None:
    """Drop retrieval + LLM caches — call whenever the index's contents
    change (new upload), since cached results may no longer be correct."""
    retrieval_cache.clear()
    llm_cache.clear()
    print("Retrieval + LLM caches invalidated after index change")


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

        Per-text results are served from `embedding_cache` where available —
        only texts that miss the cache are actually sent to the API — and
        results are stitched back together in the original input order.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        cached_vectors: Dict[int, np.ndarray] = {}
        texts_to_fetch: List[str] = []
        fetch_indices: List[int] = []

        for i, text in enumerate(texts):
            key = _cache_key("embed", self.model_name, text)
            cached = embedding_cache.get(key)
            if cached is not None:
                cached_vectors[i] = cached
            else:
                texts_to_fetch.append(text)
                fetch_indices.append(i)

        if texts_to_fetch:
            print(
                f"Generating embeddings for {len(texts_to_fetch)}/{len(texts)} texts via "
                f"HF Inference API ({len(texts) - len(texts_to_fetch)} served from cache)..."
            )

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts_to_fetch, "options": {"wait_for_model": True}},
                timeout=60,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HF Inference API error {response.status_code}: {response.text}"
                )

            fetched = np.array(response.json())
            for local_i, global_i in enumerate(fetch_indices):
                vec = fetched[local_i]
                cached_vectors[global_i] = vec
                key = _cache_key("embed", self.model_name, texts[global_i])
                embedding_cache.set(key, vec)
        else:
            print(f"All {len(texts)} text(s) served from embedding cache.")

        embeddings = np.array([cached_vectors[i] for i in range(len(texts))])
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
            # doc.metadata already carries filterable fields like
            # "location"/"category" (see load_csv/load_excel below), so this
            # just needs to pass scalars through as-is.
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

            # CACHE: the index's contents just changed, so any previously
            # cached retrieval results / LLM answers may now be stale or
            # incomplete (e.g. a newly-ingested row is a better match than
            # what got cached before it existed).
            invalidate_corpus_caches()

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
        self.history = []  # Store flat query history across all sessions (debug/back-compat)

    def _condense_followup(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Rewrite a follow-up question into a standalone one using prior
        conversation turns, so retrieval (embedding search + metadata
        filter detection) isn't blind to context like "there"/"it"/"what
        about hotels" that only makes sense given earlier turns.

        Falls back to the original question if there's no history, or if
        the LLM call fails for any reason (retrieval on the raw follow-up
        is still better than erroring out).
        """
        if not history:
            return question

        convo = format_history(history)
        condense_prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question as a standalone question that includes all necessary context from the conversation. If the follow-up question is already standalone, return it unchanged.

Respond with ONLY the rewritten standalone question — no explanation, no quotes.

Conversation history:
{convo}

Follow-up question: {question}

Standalone question:"""

        try:
            condensed = (cached_llm_invoke(condense_prompt) or "").strip().strip('"')
            return condensed if condensed else question
        except Exception as e:
            print(f"Follow-up condensing failed, falling back to raw question: {e}")
            return question

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.4,
        stream: bool = False,
        summarize: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_lat: Optional[float] = None,
        user_lng: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Pull prior turns for this session (empty list for a new/no session).
        history = session_store.get_history(session_id)

        # Rewrite follow-ups ("what about hotels there?") into a standalone
        # question using conversation context, before retrieval/filtering.
        search_question = self._condense_followup(question, history)
        if search_question != question:
            print(f"Condensed follow-up: '{question}' -> '{search_question}'")

        # Auto-scope using values seen during ingestion (category,
        # sub_category, etc.), without overriding anything the caller
        # already specified explicitly.
        metadata_filter = dict(metadata_filter) if metadata_filter else {}
        for field, value in detect_filters_from_text(search_question).items():
            metadata_filter.setdefault(field, value)

        # Taluka also gets a hardcoded fallback check — Goa's 12 talukas are
        # a small, fixed, well-known list, so this still works even if a
        # taluka column hasn't been ingested (or the registry was reset by
        # a restart).
        if "taluka" not in metadata_filter:
            detected_taluka = detect_taluka_filter(search_question)
            if detected_taluka:
                metadata_filter["taluka"] = detected_taluka

        # LOCATION: last-resort fallback — only used if the user didn't
        # name a place explicitly (by text match above) and didn't pass an
        # explicit filter. An explicitly-typed location always wins over
        # GPS position: "restaurants in Bardez" should scope to Bardez even
        # if the user is currently standing in Salcete.
        location_inferred_taluka = None
        if "taluka" not in metadata_filter and user_lat is not None and user_lng is not None:
            location_inferred_taluka = nearest_taluka(user_lat, user_lng)
            if location_inferred_taluka:
                metadata_filter["taluka"] = location_inferred_taluka
                print(
                    f"Inferred taluka '{location_inferred_taluka}' from user location "
                    f"({user_lat}, {user_lng})"
                )

        # Retrieve relevant documents using the (possibly condensed) question
        results = self.retriever.retrieve(
            search_question,
            top_k=top_k,
            score_threshold=min_score,
            metadata_filter=metadata_filter or None,
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

            # Include recent chat history so the model can resolve
            # pronouns/references ("it", "there", "that place") and keep a
            # conversational thread, while still being told to only pull
            # facts from the retrieved Context block below.
            history_block = ""
            if history:
                history_block = f"Conversation history (for context only — do not treat as source material):\n{format_history(history)}\n\n"

            # FIX: strict grounding prompt — the old version only *suggested*
            # using the context, which left the LLM free to blend in outside
            # knowledge or invent details not actually retrieved.
            prompt = f"""You are a document assistant that answers questions using ONLY the context provided below.

Rules:
- Use ONLY the information in the context to answer. Do NOT use outside knowledge, even if you know the answer.
- Do NOT guess, infer, or add details that are not explicitly present in the context.
- You may use the conversation history to understand what the user is referring to (e.g. pronouns like "it" or "there"), but never pull facts from the history itself — only from the context.
- If the context does not contain enough information to answer, respond exactly with:
  "I don't have information about this in the provided documents."
- Keep the answer concise and grounded strictly in the context.

{history_block}Context:
{context}

Question: {question}

Answer:"""

            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()

            # CACHE: identical prompts (same question, same retrieved
            # context, same history) skip the round-trip to the vLLM pod.
            answer = cached_llm_invoke(prompt)

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary = cached_llm_invoke(summary_prompt)

        # Store query history — both the flat/global debug log and, if a
        # session_id was supplied, the per-session store used for follow-ups.
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        if session_id:
            session_store.append(session_id, "user", question)
            session_store.append(session_id, "assistant", answer)

        return {
            'question': question,
            'search_question': search_question,
            'answer': answer,
            'sources': sources,
            'citations': citations,
            'summary': summary,
            'session_id': session_id,
            'history': self.history,
            'location_inferred_taluka': location_inferred_taluka,
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
        self._dummy_vector_cache: Optional[List[float]] = None

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
                {"location": "Sanguem"} or {"source_file": "goa_places.csv"} —
                lets a caller scope retrieval to a subset of ingested docs
                instead of always searching the whole index. Combine with
                Pinecone operators for more control, e.g.
                {"location": {"$eq": "Sanguem"}}.

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        # Filter values are matched literally by Pinecone, and ingestion
        # stores them lowercased (see _row_metadata), so normalize whatever
        # casing the caller/LLM produced before it ever reaches Pinecone.
        metadata_filter = normalize_metadata_filter(metadata_filter)

        # CACHE: same (query, top_k, threshold, filter) combo skips both the
        # embedding call and the Pinecone query entirely. Short TTL, and
        # explicitly dropped on any new upload (see invalidate_corpus_caches).
        cache_key = _cache_key("retrieve", query, top_k, score_threshold, metadata_filter)
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            print(f"Retrieval cache hit for query: '{query}' (filter={metadata_filter})")
            return cached

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

            retrieval_cache.set(cache_key, retrieved_docs)
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def list_by_filter(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return documents that match a metadata filter, ignoring semantic
        similarity entirely.

        Pinecone always ranks by vector distance, even when a filter is
        supplied — there's no native "just give me everything where
        location = Sanguem" call. The workaround is to query with a neutral
        (zero) vector and a large top_k: the filter still narrows the
        candidate set correctly, we just don't care about the resulting
        (meaningless) similarity ordering.

        Args:
            metadata_filter: Required Pinecone metadata filter, e.g.
                {"location": "Sanguem"}.
            limit: Max number of matches to return (Pinecone allows up to
                10,000 per query).
        """
        if not metadata_filter:
            raise ValueError("metadata_filter is required for list_by_filter")

        metadata_filter = normalize_metadata_filter(metadata_filter)

        # CACHE: same as retrieve() — same filter/limit combo is served from
        # the shared retrieval cache instead of re-hitting Pinecone.
        cache_key = _cache_key("list_by_filter", metadata_filter, limit)
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            print(f"Retrieval cache hit for list_by_filter: {metadata_filter}")
            return cached

        # FIX: an all-zero vector has undefined cosine similarity, and some
        # Pinecone index configurations return zero matches for it even
        # when the metadata filter alone would match plenty of rows. Use a
        # real (cached) embedding instead — the filter still does 100% of
        # the actual scoping; this vector only affects the meaningless
        # ordering of results within that filtered set.
        if self._dummy_vector_cache is None:
            self._dummy_vector_cache = self.embedding_manager.generate_embeddings([" "])[0].tolist()

        results = self.vector_store.index.query(
            vector=self._dummy_vector_cache,
            top_k=min(limit, 10000),
            filter=metadata_filter,
            include_metadata=True,
        )

        docs = []
        for match in results.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            document_text = metadata.pop("text", "")
            docs.append({
                'id': match.get("id"),
                'content': document_text,
                'metadata': metadata,
            })

        retrieval_cache.set(cache_key, docs)
        return docs

rag_retriever=RAGRetriever(vectorstore,embedding_manager)

# Wire up the pipeline now that the index is ready
rag_pipeline = AdvancedRAGPipeline(rag_retriever,llm)


# ==========================================================
# FILE LOADING (PDF / CSV / XLSX)
# ==========================================================

def _row_metadata(row: pd.Series, columns: List[str]) -> Dict[str, Any]:
    """
    Turn every column in a CSV/Excel row into filterable Pinecone metadata.

    Column names are normalized ("Location" -> "location") so filters like
    {"location": "Sanguem"} work regardless of the source file's exact header
    casing/spacing. Only scalar values are kept (Pinecone metadata can't hold
    nested structures), empty cells are skipped, and long strings are capped
    so one oversized column can't blow the per-vector metadata limit.
    """
    metadata: Dict[str, Any] = {}
    for col in columns:
        key = str(col).strip().lower().replace(" ", "_")
        if not key:
            continue
        val = row[col]
        if val is None or val == "":
            continue
        if isinstance(val, str):
            val = val.strip()
            if not val:
                continue
            if len(val) > MAX_METADATA_FIELD_CHARS:
                val = val[:MAX_METADATA_FIELD_CHARS]
            # Lowercase so filtering is case-insensitive: Pinecone's $eq is a
            # literal string match, so "Sanguem"/"sanguem"/"SANGUEM" would
            # otherwise be three different filter values. The original
            # casing is still readable in page_content/row_text — only the
            # metadata copy used for filtering is lowercased.
            val = val.lower()
        elif not isinstance(val, (int, float, bool)):
            # Skip anything that isn't a plain scalar (e.g. lists, dicts,
            # Timestamps) rather than risk an upsert error.
            continue
        metadata[key] = val
    return metadata


def normalize_metadata_filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Lowercase every string value inside a metadata filter (recursing into
    Pinecone operator dicts like {"$eq": "Sanguem"} and lists like
    {"$in": ["Sanguem", "Tiswadi"]}), so a filter built from a user's raw
    question ("Sanguem", "SANGUEM", "sanguem ") matches the lowercased
    values stored by _row_metadata. Field *names* (dict keys) are left
    alone — only the values are touched.
    """
    def _normalize(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, list):
            return [_normalize(v) for v in value]
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items()}
        return value

    if not metadata_filter:
        return metadata_filter
    return {k: _normalize(v) for k, v in metadata_filter.items()}


def load_pdf(path: str, filename: str) -> List[Document]:
    """Load a PDF file into LangChain Documents (one per page)."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = filename
        doc.metadata["file_type"] = "pdf"
    return docs


def load_csv(path: str, filename: str) -> List[Document]:
    """
    Load a CSV file into LangChain Documents (one per row).

    Every column (e.g. "location", "category", "name") is promoted to its
    own metadata field via _row_metadata, so ingested rows can later be
    scoped with a Pinecone filter such as {"location": "Sanguem"} instead of
    relying purely on vector similarity across the whole index.
    """
    # Detect the file's actual encoding instead of relying on the
    # OS default (cp1252 on Windows), which breaks on non-ASCII bytes.
    detected = from_path(path).best()
    encoding = detected.encoding if detected else "utf-8"

    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # Last-resort fallback: latin-1 maps every byte 0-255,
        # so it will never raise, even if it's not a perfect guess.
        df = pd.read_csv(path, encoding="latin-1")

    df = df.fillna("")
    columns = [str(c) for c in df.columns]
    _register_categorical_columns(df, columns)

    docs: List[Document] = []
    for row_idx, row in df.iterrows():
        row_text = "\n".join(f"{col}: {row[col]}" for col in columns)
        metadata = {
            "source_file": filename,
            "file_type": "csv",
            "row": int(row_idx),
        }
        metadata.update(_row_metadata(row, columns))
        docs.append(Document(page_content=row_text, metadata=metadata))

    return docs


def load_excel(path: str, filename: str) -> List[Document]:
    """
    Load an Excel workbook into LangChain Documents.
    Every sheet is read, and each row becomes its own Document so that
    retrieval can point back to a specific sheet/row. As with load_csv,
    every column is also promoted to a top-level metadata field so rows can
    be filtered by e.g. location/category at query time.
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
        _register_categorical_columns(df, columns)

        for row_idx, row in df.iterrows():
            row_text = "\n".join(f"{col}: {row[col]}" for col in columns)
            content = f"Sheet: {sheet_name}\n{row_text}"
            metadata = {
                "source_file": filename,
                "file_type": "xlsx",
                "sheet": sheet_name,
                "row": int(row_idx),
            }
            metadata.update(_row_metadata(row, columns))
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
    # FIX: generic metadata filter (e.g. {"location": "Sanguem"},
    # {"source_file": "goa_places.csv"}) so a query can be scoped to a
    # subset of the index instead of always searching everything.
    filter: Optional[Dict[str, Any]] = None
    # Chat session support: pass back the session_id you got from a
    # previous /query response to ask a follow-up question that has access
    # to that conversation's history. Omit (or pass a brand-new id) to
    # start a fresh conversation — a new session_id is generated and
    # returned either way.
    session_id: Optional[str] = None
    # Optional live GPS coordinates. If provided, and the question doesn't
    # already name/imply a specific taluka (by text match or an explicit
    # `filter`), the query is automatically scoped to whichever taluka the
    # user is nearest to — so "any good seafood restaurants nearby" can
    # resolve to an actual place instead of guessing from chat history.
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None


class DocumentsFilterRequest(BaseModel):
    # Required — this endpoint always scopes by metadata, e.g.
    # {"location": "Sanguem"} or {"category": "Hotel"}.
    filter: Dict[str, Any]
    limit: int = 100


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/")
def root():
    return {"message": "RAG API Running — go to /docs for Swagger UI"}


@app.get("/health")
def health():
    return {"status": "healthy", "documents": vectorstore.count()}


@app.get("/cache/stats")
def cache_stats():
    """Inspect hit/miss/size stats for each in-memory cache layer."""
    return {
        "embedding_cache": embedding_cache.stats(),
        "retrieval_cache": retrieval_cache.stats(),
        "llm_cache": llm_cache.stats(),
    }


@app.post("/cache/clear")
def cache_clear(target: Optional[str] = None):
    """
    Manually clear cache(s). `target` is optional and one of
    "embedding", "retrieval", "llm" — omit it to clear all three.
    """
    valid_targets = {"embedding": embedding_cache, "retrieval": retrieval_cache, "llm": llm_cache}

    if target is None:
        for cache in valid_targets.values():
            cache.clear()
        return {"message": "All caches cleared"}

    cache = valid_targets.get(target)
    if cache is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cache target '{target}'. Valid options: {sorted(valid_targets)}"
        )
    cache.clear()
    return {"message": f"'{target}' cache cleared"}


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
    # Always hand back a session_id — either the one the caller supplied
    # (so follow-ups keep accumulating in the same conversation) or a fresh
    # one if this is the start of a new conversation.
    session_id = request.session_id or str(uuid.uuid4())

    result = rag_pipeline.query(
        question=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        summarize=request.summarize,
        metadata_filter=dict(request.filter) if request.filter else None,
        session_id=session_id,
        user_lat=request.user_lat,
        user_lng=request.user_lng,
    )
    return result


@app.get("/location/taluka")
def location_taluka(lat: float, lng: float):
    """
    Standalone reverse-geocode: given GPS coordinates, return the nearest
    Goa taluka. Handy for testing the location feature independently of a
    full /query call, or for a frontend to show "Detected: Bardez" near a
    location prompt before the user even asks a question.
    """
    taluka = nearest_taluka(lat, lng)
    if taluka is None:
        return {"lat": lat, "lng": lng, "taluka": None, "message": "Location too far from Goa to infer a taluka"}
    return {"lat": lat, "lng": lng, "taluka": taluka}


@app.post("/sessions")
def create_session():
    """Explicitly start a new empty chat session (optional — /query will
    also create one automatically if you don't pass a session_id)."""
    session_id = str(uuid.uuid4())
    session_store.touch(session_id)
    return {"session_id": session_id}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Fetch the stored conversation turns for a session."""
    if not session_store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {"session_id": session_id, "messages": session_store.get_history(session_id)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Clear a session's history (e.g. user hits 'New chat')."""
    session_store.clear(session_id)
    return {"message": "Session cleared", "session_id": session_id}


@app.post("/documents")
def list_documents(request: DocumentsFilterRequest):
    """
    Return every ingested row/chunk matching a metadata filter, e.g.
    {"filter": {"location": "Sanguem"}} — no similarity ranking or
    score_threshold involved, so this is the right call for "give me
    everything about Sanguem" rather than /query, which only returns the
    top_k most semantically relevant chunks.
    """
    docs = rag_retriever.list_by_filter(request.filter, limit=request.limit)
    return {"count": len(docs), "documents": docs}