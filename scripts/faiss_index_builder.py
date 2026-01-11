# ===============
#  FAISS INDEX
# ===============

import os
import spacy
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessing_data import clean_text
import faiss

# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = r"C:/Users/raad2/Downloads/Product Matching/models/embedding_model"
NER_MODEL_PATH = r"C:/Users/raad2/Downloads/Product Matching/models/ner_model"
DATA_FILE = r"C:/Users/raad2/Downloads/Product Matching/data/Cleaned_data.xlsx"
FAISS_INDEX_FILE = r"C:/Users/raad2/Downloads/Product Matching/models/faiss_index.bin"

# -----------------------------
# WEIGHTS
# -----------------------------
WEIGHTS = {
    "BRAND": 3,
    "FORM": 2,
    "DOSAGE_VALUE": 1,
    "DOSAGE_UNIT": 1,
    "QUANTITY": 1
}

# -----------------------------
# LOAD MODELS AND DATA
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_PATH)

print("Loading NER model...")
nlp = spacy.load(NER_MODEL_PATH)

print("Loading SKU data...")
df = pd.read_excel(DATA_FILE)
df.rename(columns={"SKU_Name": "sku"}, inplace=True)
df_sku = df[["sku"]].drop_duplicates().reset_index(drop=True)

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def apply_weighting_text(text, nlp_model):
    """Apply NER-based weighting to the text."""
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
    
    if not found:
        return str(text).strip()
    
    return " ".join(components)

def clean_single_text(text):
    """Clean text using the provided cleaning function."""
    temp_df = pd.DataFrame({"alias": [str(text)]})
    temp_df = clean_text(temp_df, "alias")
    return temp_df["alias"].iloc[0]

def encode_skus(skus_list):
    """Encode SKUs using SentenceTransformer and normalize embeddings."""
    embeddings = model.encode(
        skus_list,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings.astype(np.float32)

# -----------------------------
# FAISS INDEX HANDLER
# -----------------------------
def build_or_load_faiss_index(sku_embeddings, index_file=FAISS_INDEX_FILE):
    """Build FAISS index or load from disk if exists."""
    if os.path.exists(index_file):
        print(f"Loading existing FAISS index from {index_file}...")
        index = faiss.read_index(index_file)
        print("FAISS index loaded successfully.")
    else:
        print("Building new FAISS index...")
        dimension = sku_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(sku_embeddings)
        faiss.write_index(index, index_file)
        print(f"FAISS index built and saved to {index_file}.")
    return index

# -----------------------------
# PREPARE SKU EMBEDDINGS AND INDEX
# -----------------------------
print("Preparing weighted SKU embeddings...")
weighted_skus = [apply_weighting_text(sku, nlp) for sku in df_sku["sku"]]
sku_embeddings = encode_skus(weighted_skus)

index = build_or_load_faiss_index(sku_embeddings)

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_alias(alias, k=10):
    """Search for top-k SKUs for a given alias."""
    alias_clean = clean_single_text(alias)
    weighted_alias = apply_weighting_text(alias_clean, nlp)
    
    alias_emb = model.encode(weighted_alias, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    query_vector = alias_emb.reshape(1, -1)
    
    distances, indices = index.search(query_vector, k)
    
    scores = 1.0 - (distances ** 2) / 2.0
    results = []
    for rank, (idx, score, dist) in enumerate(zip(indices[0], scores[0], distances[0]), 1):
        results.append({
            "rank": rank,
            "sku": df_sku.iloc[idx]["sku"],
            "weighted_sku": weighted_skus[idx],
            "score": float(score),
            "l2_distance": float(dist)
        })
    
    return {
        "original_alias": alias,
        "cleaned_alias": alias_clean,
        "weighted_alias": weighted_alias,
        "results": results
    }

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    test_aliases = ["panadol 500mg tablets", "augmentin 625mg", "voltaren gel"]
    
    for alias in test_aliases:
        res = search_alias(alias, k=10)
        print(f"\nAlias: {alias}")
        for match in res["results"]:
            print(f"{match['rank']}. {match['sku']} | Score: {match['score']:.4f}")