# TrustTrace 🔍

*"We don't ship rows. We ship trust."*

A prototype ingestion and traceability pipeline for patent data that combines traditional database storage with modern vector search capabilities.

## 🎯 Overview

TrustTrace is a FastAPI-based pipeline that processes patent data (25k+ rows), ensures full traceability through SHA256 fingerprinting, and provides semantic search capabilities using vector embeddings.

## ✨ Features

- **📊 Data Ingestion**: Processes 25,000+ patent records from CSV
- **🔒 Traceability**: SHA256 fingerprinting for every data row
- **🗄️ Dual Storage**: Raw + cleaned data in DuckDB
- **🧠 Vector Search**: Semantic search using SentenceTransformers + ChromaDB
- **🔍 REST API**: Simple POST endpoint for patent search
- **📈 Quality Scoring**: Similarity scores for search results

## 🏗️ Architecture

```
CSV Data → Data Cleaning → DuckDB Storage
                    ↓
            Text Embedding → ChromaDB Storage
                    ↓
              FastAPI Search Endpoint
```

## 🚀 Quick Start

### Prerequisites

bash--
Python 3.8+
pip install -r requirements.txt

### Installation
git clone <https://github.com/KarChikey420/TrustTrace_flow.git>
cd trusttrace
pip install -r requirements.txt

### Requirements
fastapi
uvicorn
sentence-transformers
chromadb
pandas
duckdb
pydantic

### Setup

1. **Prepare your data**: Place your patent CSV file as `titles.csv` in the project root
2. **Expected CSV format**:csv
   code,title,class,subclass,group,main_group
   A,Patent Title Here,01,B,23,456

3. **Run the pipeline**:
   python API.py

4. **Access the API**: http://localhost:8000

## 📡 API Endpoints

### POST /search

Search for patents using semantic similarity.

**Request:**
json
{
  "query": "artificial intelligence machine learning"
}

**Response:**
json
[
  {
    "title": "Machine Learning System for Data Processing",
    "patent_number": "A01B23/456",
    "quality_score": 0.85,
    "source_info": {
      "source_file": "titles.csv",
      "timestamp": "2024-01-15T10:30:00Z",
      "fingerprint": "a1b2c3d4e5f6..."
    }
  }
]

### Example Usage

bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "agriculture tools"}'

## 🔧 Technical Details

### Data Processing Pipeline

1. **Ingestion**: Reads 25,000 rows from CSV file
2. **Cleaning**: Removes invalid records, cleans classification fields
3. **Fingerprinting**: Generates SHA256 hash for each row
4. **Storage**: Saves to DuckDB with full traceability
5. **Embedding**: Creates vector embeddings for patent titles
6. **Indexing**: Stores embeddings in ChromaDB for fast similarity search

### Traceability Features

- **Source Fingerprint**: SHA256 hash of original row data
- **Ingestion Timestamp**: When the record was processed
- **Source File**: Original data file reference
- **Quality Score**: Similarity confidence (0-1 scale)

### Vector Search

- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Database**: ChromaDB for vector storage
- **Search**: Cosine similarity matching
- **Results**: Top 3 most relevant patents
## 🛠️ Development

### Project Structure

trusttrace/
├── API.py              # Main FastAPI application
├── titles.csv          # Input patent data
├── requirements.txt     # Python dependencies
├── chroma_db/          # ChromaDB storage (created automatically)
├── patents.db          # DuckDB database (created automatically)
└── README.md           # This file

### Logging

The application provides comprehensive logging:
- Data processing statistics
- Search query performance
- Error tracking and debugging

### Performance

- **Ingestion**: ~25,000 records processed in under 5 minutes
- **Search**: Sub-second response times
- **Storage**: Efficient compression with DuckDB
- **Embeddings**: Batch processing for optimal performance

## 🚨 Troubleshooting

### Common Issues

1. **ChromaDB Collection Error**:
   Collection patents already exists
   - Solution: The system automatically handles this by deleting and recreating collections

2. **Memory Issues**:
   - Reduce batch size in `index_in_chroma()` function
   - Process data in smaller chunks

3. **CSV Format Issues**:
   - Ensure CSV has required columns: `code`, `title`
   - Check for proper UTF-8 encoding

## 📝 License

MIT License - Feel free to use and modify as needed.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the logs for debugging information
- Verify your CSV data format

## 🚫 Failure Modes--

 Failure Mode 1: ChromaDB Write Failure

    Trigger: Disk full, corrupted ChromaDB folder, or client lock during .add() or .delete_collection().

    Auto-remediation:
    Fallback to in-memory Chroma collection and reattempt persistence on next run.

 Failure Mode 2: Corrupted or Missing CSV Input

    Trigger: titles.csv file not found, malformed rows, or incorrect encoding.

    Auto-remediation:
    Send Slack alert + skip ingestion; retry after backing up last known good input.

 Failure Mode 3: Schema Drift in CSV

    Trigger: Missing expected columns like title, code, class, or group.

    Auto-remediation:
    Log schema mismatch; quarantine file with timestamp and notify via webhook.

 Failure Mode 4: DuckDB Write Conflict

    Trigger: Concurrent writes or locked database file during INSERT INTO patents_data.

    Auto-remediation:
    Retry after short delay with exponential backoff; log write attempts in audit table.

 Failure Mode 5: Model Embedding Crash

    Trigger: SentenceTransformer model not loaded, incompatible input (e.g., null titles), or OOM error on batch.

    Auto-remediation:
    Catch and log model errors; skip problematic rows and continue indexing next batch.

