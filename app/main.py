# ===============================
#  FASTAPI PRODUCTION APP
#  Product Matching System
# ===============================

import os
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import spacy
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # parent of 'app'
sys.path.append(str(BASE_DIR / "scripts"))

from preprocessing_data import clean_text


# ===============================
# LOGGING SETUP
# ===============================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# PATHS
# ===============================

BASE_PATH = Path("C:/Users/raad2/Downloads/Product Matching")
MODEL_PATH = BASE_PATH / "models/embedding_model"
NER_MODEL_PATH = BASE_PATH / "models/ner_model"
FAISS_INDEX_PATH = BASE_PATH / "models/faiss_index.bin"
DATA_FILE = BASE_PATH / "data/Cleaned_data.xlsx"

# ===============================
# WEIGHTS FOR NER
# ===============================

WEIGHTS = {
    "BRAND": 3,
    "FORM": 2,
    "DOSAGE_VALUE": 1,
    "DOSAGE_UNIT": 1,
    "QUANTITY": 1
}

# ===============================
# PYDANTIC MODELS
# ===============================

class SearchRequest(BaseModel):
    alias: str = Field(..., min_length=1, description="Product alias to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")

class BatchSearchRequest(BaseModel):
    aliases: List[str] = Field(..., min_items=1, max_items=5000, description="List of product aliases")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results per alias")

class SearchResult(BaseModel):
    rank: int
    sku: str
    score: float
    l2_distance: float

class SearchResponse(BaseModel):
    query: str
    cleaned_query: str
    weighted_query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float

class BatchSearchResult(BaseModel):
    alias: str
    results: List[SearchResult]

class BatchSearchResponse(BaseModel):
    total_queries: int
    successful: int
    failed: int
    results: List[BatchSearchResult]
    total_processing_time_ms: float
    avg_time_per_query_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    faiss_index_size: int
    timestamp: str

# ===============================
# GLOBAL VARIABLES
# ===============================

embedding_model = None
ner_model = None
faiss_index = None
df_sku = None
weighted_skus = None

# ===============================
# UTILITY FUNCTIONS
# ===============================

def apply_weighting_text(text: str, nlp_model) -> str:
    """Apply NER-based weighting to text."""
    doc = nlp_model(str(text))
    components = []
    found = False
    
    for ent in doc.ents:
        lab = ent.label_
        if lab in WEIGHTS:
            found = True
            rep = WEIGHTS[lab]
            token = ent.text.strip()
            if token:
                components.extend([token] * rep)
    
    return " ".join(components) if found else str(text).strip()


def clean_single_text(text: str) -> str:
    """Clean text using preprocessing function."""
    temp_df = pd.DataFrame({"alias": [str(text)]})
    temp_df = clean_text(temp_df, "alias")
    return temp_df["alias"].iloc[0]


# ===============================
# FASTAPI APP
# ===============================

app = FastAPI(
    title="Product Matching API",
    description="Semantic search API for product SKU matching",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# STARTUP EVENT
# ===============================

@app.on_event("startup")
async def startup_event():
    """Load all models and data on startup."""
    global embedding_model, ner_model, faiss_index, df_sku, weighted_skus
    
    try:
        logger.info("=" * 60)
        logger.info("STARTING PRODUCT MATCHING API")
        logger.info("=" * 60)
        
        # Load Embedding Model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(str(MODEL_PATH))
        logger.info("Embedding model loaded")
        
        # Load NER Model
        logger.info("Loading NER model...")
        ner_model = spacy.load(str(NER_MODEL_PATH))
        logger.info("NER model loaded")
        
        # Load SKU Data
        logger.info("Loading SKU data...")
        df = pd.read_excel(DATA_FILE)
        df.rename(columns={"SKU_Name": "sku"}, inplace=True)
        df_sku = df[["sku"]].drop_duplicates().reset_index(drop=True)
        logger.info(f"Loaded {len(df_sku)} unique SKUs")
        
        # Load FAISS Index
        logger.info("Loading FAISS index...")
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index loaded ({faiss_index.ntotal} vectors)")
        
        # Precompute weighted SKUs
        logger.info("Preparing weighted SKUs...")
        weighted_skus = [apply_weighting_text(sku, ner_model) for sku in df_sku["sku"]]
        logger.info("Weighted SKUs prepared")
        
        logger.info("=" * 60)
        logger.info("API READY TO SERVE REQUESTS")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"STARTUP FAILED: {str(e)}")
        raise

# ===============================
# ENDPOINTS
# ===============================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Product Matching API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "batch-search": "/batch-search",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    
    return HealthResponse(
        status="healthy",
        models_loaded=all([embedding_model, ner_model, faiss_index, df_sku]),
        faiss_index_size=faiss_index.ntotal if faiss_index else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_products(request: SearchRequest):
    """
    Search for matching SKUs based on product alias.
    
    - **alias**: Product name or description to search
    - **top_k**: Number of top results to return (1-50)
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Search request: '{request.alias}' (top_k={request.top_k})")
        
        # Validate models are loaded
        if not all([embedding_model, ner_model, faiss_index]) or df_sku is None or df_sku.empty:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models or data not loaded. Please try again later."
            )
        
        # Clean and weight the alias
        cleaned_alias = clean_single_text(request.alias)
        weighted_alias = apply_weighting_text(cleaned_alias, ner_model)
        
        logger.info(f"Cleaned: '{cleaned_alias}' | Weighted: '{weighted_alias}'")
        
        # Encode the query
        query_embedding = embedding_model.encode(
            weighted_alias,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = faiss_index.search(query_embedding, request.top_k)
        
        # Calculate cosine similarity scores
        scores = 1.0 - (distances ** 2) / 2.0
        
        # Build results
        results = []
        for rank, (idx, score, dist) in enumerate(zip(indices[0], scores[0], distances[0]), 1):
            results.append(SearchResult(
                rank=rank,
                sku=df_sku.iloc[idx]["sku"],
                score=float(score),
                l2_distance=float(dist)
            ))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Search completed in {processing_time:.2f}ms | Top result: {results[0].sku} (score={results[0].score:.4f})")
        
        return SearchResponse(
            query=request.alias,
            cleaned_query=cleaned_alias,
            weighted_query=weighted_alias,
            results=results,
            total_results=len(results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/batch-search", response_model=BatchSearchResponse, tags=["Search"])
async def batch_search_products(request: BatchSearchRequest):
    """
    Batch search for multiple product aliases at once.
    
    - **aliases**: List of product names/descriptions (max 5000)
    - **top_k**: Number of results per alias (1-50)
    
    Optimized for processing large batches efficiently.
    """
    start_time = datetime.now()
    
    try:
        total_aliases = len(request.aliases)
        logger.info(f"Batch search request: {total_aliases} aliases (top_k={request.top_k})")
        
        # Validate models are loaded
        if not all([embedding_model, ner_model, faiss_index]) or df_sku is None or df_sku.empty:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models or data not loaded. Please try again later."
            )
        
        batch_results = []
        successful = 0
        failed = 0
        
        # Step 1: Clean and weight all aliases
        logger.info("Cleaning and weighting aliases...")
        cleaned_aliases = []
        weighted_aliases = []
        
        for alias in request.aliases:
            try:
                cleaned = clean_single_text(alias)
                weighted = apply_weighting_text(cleaned, ner_model)
                cleaned_aliases.append(cleaned)
                weighted_aliases.append(weighted)
            except Exception as e:
                logger.error(f"Failed to process alias '{alias}': {e}")
                cleaned_aliases.append(alias)
                weighted_aliases.append(alias)
        
        # Step 2: Batch encode all queries
        logger.info("Encoding all queries in batch...")
        batch_embeddings = embedding_model.encode(
            weighted_aliases,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)
        
        # Step 3: Batch search FAISS
        logger.info("Searching FAISS index...")
        distances, indices = faiss_index.search(batch_embeddings, request.top_k)
        
        # Calculate scores
        scores = 1.0 - (distances ** 2) / 2.0
        
        # Build results
        logger.info("Building response...")
        for i, (alias, dist_row, idx_row, score_row) in enumerate(zip(
            request.aliases, distances, indices, scores
        )):
            try:
                results = []
                for rank, (idx, score, dist) in enumerate(zip(idx_row, score_row, dist_row), 1):
                    results.append(SearchResult(
                        rank=rank,
                        sku=df_sku.iloc[idx]["sku"],
                        score=float(score),
                        l2_distance=float(dist)
                    ))
                
                batch_results.append(BatchSearchResult(
                    alias=alias,
                    results=results
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to build results for alias '{alias}': {e}")
                failed += 1
        
        # Calculate processing time
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        avg_time_per_query = total_processing_time / total_aliases
        
        logger.info(f"Batch search completed in {total_processing_time:.2f}ms")
        logger.info(f"Average time per query: {avg_time_per_query:.2f}ms")
        logger.info(f"Success: {successful}/{total_aliases} | Failed: {failed}/{total_aliases}")
        
        return BatchSearchResponse(
            total_queries=total_aliases,
            successful=successful,
            failed=failed,
            results=batch_results,
            total_processing_time_ms=round(total_processing_time, 2),
            avg_time_per_query_ms=round(avg_time_per_query, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )


# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )