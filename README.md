# Product Matching System

A semantic search API for matching product aliases to standardized SKU names using machine learning techniques including Named Entity Recognition (NER), sentence embeddings, and vector similarity search.

## Purpose

This project implements a product matching system that takes product aliases (user-input product descriptions) and finds the most similar standardized SKU names from a catalog. It uses a combination of:

- **Named Entity Recognition (NER)** to extract and weight important product attributes (brand, form, dosage, etc.)
- **Sentence Embeddings** to convert text into vector representations
- **FAISS Vector Search** for efficient similarity matching

The system achieves high accuracy with 98% top-10 accuracy and 92.96% top-1 accuracy on evaluation data.

## Technologies Used

- **Python 3.8+**
- **FastAPI** - Web framework for the API
- **Sentence Transformers** - For generating text embeddings (based on BAAI/bge-base-en-v1.5)
- **spaCy** - For Named Entity Recognition
- **FAISS** - Vector similarity search library
- **PyTorch** - Machine learning framework
- **pandas** - Data processing
- **NumPy** - Numerical computations
- **scikit-learn** - For evaluation metrics
- **tqdm** - Progress bars for training

## Project Structure

```
product-matching/
├── app/
│   └── main.py                 # FastAPI application with search endpoints
├── scripts/
│   ├── preprocessing_data.py   # Text cleaning and normalization
│   ├── regex_data.py           # Regex-based entity extraction
│   ├── training_ner.py         # NER model training
│   ├── ner_on_sku.py           # Apply NER to SKU data
│   ├── augmentation_data.py    # Data augmentation for training
│   ├── embedding.py            # Train sentence embedding model with contrastive learning
│   ├── faiss_index_builder.py  # Build FAISS vector index
│   ├── evaluation_result.py    # Model evaluation metrics
│   └── check_api_test.py       # API testing
├── models/
│   ├── embedding_model/        # Trained sentence transformer model
│   │   ├── final_model_20251124_233003/  # Latest trained model
│   │   └── eval/               # Evaluation results
│   └── ner_model/              # Trained spaCy NER model
├── data/
│   ├── api_test.json           # API test data
│   ├── Extract_entities_*.json # Extracted entity data
│   ├── labeled_datasets_specified/ # Labeled training data
│   └── Cleaned_data.xlsx       # SKU catalog (not shown in structure)
├── logs/                       # Application logs
|                       
├── requirements.txt            # Python dependencies
├── pipeline.txt                # Data processing pipeline description
├── result_test_api.txt         # API test results
└── README.md                   # This file
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 8GB RAM recommended for model loading

### Installation

1. Clone or download the project:
   ```bash
   cd "C:/Users/raad2/Downloads/Product Matching"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all model files are present in the `models/` directory:
   - `embedding_model/` - Sentence transformer model
   - `ner_model/` - spaCy NER model
   - `faiss_index.bin` - FAISS vector index (built by faiss_index_builder.py)

4. Ensure data file is present:
   - `data/Cleaned_data.xlsx` - Contains SKU catalog with SKU_Name column

## How to Run

### Start the API Server

Run the FastAPI application using uvicorn:

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
- **GET** `/health`
- Returns system status and model loading information

#### Single Search
- **POST** `/search`
- Body: `{"alias": "product description", "top_k": 10}`
- Returns top-k matching SKUs with similarity scores

#### Batch Search
- **POST** `/batch_search`
- Body: `{"aliases": ["product1", "product2"], "top_k": 10}`
- Returns matches for multiple aliases

### API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Data Processing Pipeline

The system follows this data processing pipeline:

1. **Raw Data** → `preprocessing_data.py` → Cleaned standardized data
2. **Cleaned Data** → `regex_data.py` → Extract entities using regex
3. **Entity Data** → `training_ner.py` → Train NER model
4. **Cleaned Data** → `ner_on_sku.py` → Apply NER to SKU data
5. **NER Results** → `augmentation_data.py` → Generate labeled training data
6. **Training Data** → `embedding.py` → Train embedding model using contrastive learning
7. **Cleaned Data** → `faiss_index_builder.py` → Build FAISS search index

## Model Training Details

### NER Model
- Extracts product attributes: BRAND, FORM, DOSAGE_VALUE, DOSAGE_UNIT, QUANTITY
- Weights: BRAND(3x), FORM(2x), DOSAGE_VALUE(1x), DOSAGE_UNIT(1x), QUANTITY(1x)
- Used for emphasizing important product features in text

### Embedding Model
- Based on BAAI/bge-base-en-v1.5 sentence transformer
- Trained using contrastive learning with MultipleNegativesRankingLoss
- Batch structure: 1 hard positive, 1 soft positive, 3 hard negatives, 4 soft negatives (9 total per batch)
- Training epochs: 5
- Uses weighted text representations from NER

### FAISS Index
- L2 distance-based similarity search
- Built from embedded SKU representations
- Enables fast retrieval of similar products

## Evaluation Results

Based on 952 test samples:

- **Top-1 Accuracy**: 92.96%
- **Top-10 Accuracy**: 98.00%
- **Top-10 Precision**: 83.08%
- **Top-10 Recall**: 92.96%
- **Top-10 F1 Score**: 87.73%

## Usage Examples

### Python Client

```python
import requests

# Single search
response = requests.post("http://localhost:8000/search",
    json={"alias": "aspirin 100mg tablet", "top_k": 5}
)
results = response.json()
print(results)

# Batch search
response = requests.post("http://localhost:8000/batch_search",
    json={"aliases": ["ibuprofen 200mg", "paracetamol 500mg"], "top_k": 3}
)
batch_results = response.json()
print(batch_results)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single search
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"alias": "vitamin c 1000mg tablet", "top_k": 10}'

# Batch search
curl -X POST "http://localhost:8000/batch_search" \
     -H "Content-Type: application/json" \
     -d '{"aliases": ["omeprazole 20mg", "lisinopril 10mg"], "top_k": 5}'
```

## Configuration

### Model Paths
Update paths in `app/main.py` if models are stored elsewhere:

```python
BASE_PATH = Path("C:/Users/raad2/Downloads/Product Matching")
MODEL_PATH = BASE_PATH / "models/embedding_model"
NER_MODEL_PATH = BASE_PATH / "models/ner_model"
FAISS_INDEX_PATH = BASE_PATH / "models/faiss_index.bin"
DATA_FILE = BASE_PATH / "data/Cleaned_data.xlsx"
```

### NER Weights
Modify entity weights in `app/main.py`:

```python
WEIGHTS = {
    "BRAND": 3,
    "FORM": 2,
    "DOSAGE_VALUE": 1,
    "DOSAGE_UNIT": 1,
    "QUANTITY": 1
}
```

## Training Scripts

### Running Individual Components

To retrain models or rebuild indices:

```bash
# Train NER model
python scripts/training_ner.py

# Apply NER to SKU data
python scripts/ner_on_sku.py

# Augment training data
python scripts/augmentation_data.py

# Train embedding model
python scripts/embedding.py

# Build FAISS index
python scripts/faiss_index_builder.py

# Evaluate model
python scripts/evaluation_result.py
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure all model files are present in `models/` directory
2. **Data File Missing**: Verify `data/Cleaned_data.xlsx` exists
3. **Port Already in Use**: Change port in uvicorn command
4. **Memory Issues**: FAISS index may require significant RAM for large datasets
5. **Training Failures**: Check that labeled_datasets_specified folder contains training data

### Logs
Check `logs/app_YYYYMMDD.log` for detailed error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

This project is proprietary. Please contact the development team for licensing information.

## Contact

For questions or support, please contact the development team.
