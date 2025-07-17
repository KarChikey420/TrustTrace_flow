from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import duckdb
import hashlib
import json
from datetime import datetime
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Patent Pipeline API", version="1.0.0")

CHROMA_DB_PATH = "chroma_db"
DUCKDB_PATH = "patents.db"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
PATENT_DATA_FILE = "titles.csv"

class SearchQuery(BaseModel):
    query: str

class SearchResult(BaseModel):
    title: str
    patent_number: str
    quality_score: float
    source_info: dict

def clean_patent_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate patent data"""
    
    df = df.dropna(subset=['code', 'title'])
    
    for col in ['class', 'subclass', 'group', 'main_group']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('*', '', regex=False)
            df[col] = df[col].replace('', None)
    
    return df

def process_data(input_file: str) -> pd.DataFrame:
    """Process raw data and add traceability"""
    try:
        raw_df = pd.read_csv(input_file, nrows=25000, encoding='utf-8')
        logger.info(f"Read {len(raw_df)} rows from {input_file}")
        
        cleaned_df = clean_patent_data(raw_df)
        
        cleaned_df['patent_number'] = cleaned_df.apply(
            lambda row: f"{row['code']}{row['class'] or ''}{row['subclass'] or ''}{row['group'] or ''}/{row['main_group'] or ''}".rstrip('/'),
            axis=1
        )
    
        cleaned_df['source_fingerprint'] = cleaned_df.apply(
            lambda row: hashlib.sha256(
                json.dumps(row.to_dict(), sort_keys=True, default=str).encode()
            ).hexdigest(), 
            axis=1
        )
        
        cleaned_df['ingestion_timestamp'] = datetime.utcnow()
        cleaned_df['source_file'] = input_file
        
        logger.info(f"Processed {len(cleaned_df)} rows with traceability")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")
        raise

def store_in_duckdb(df: pd.DataFrame):
    conn = duckdb.connect(database=DUCKDB_PATH)
    
    try:
        conn.execute("""
        CREATE OR REPLACE TABLE patents_data (
            patent_number VARCHAR,
            title TEXT,
            code VARCHAR,
            class VARCHAR,
            subclass VARCHAR,
            group_ VARCHAR,
            main_group VARCHAR,
            source_fingerprint VARCHAR,
            ingestion_timestamp TIMESTAMP,
            source_file VARCHAR
        );
        """)
        
        if 'group' in df.columns:
            df.rename(columns={'group': 'group_'}, inplace=True)
        
        conn.register('temp_df', df)
        conn.execute("""
        INSERT INTO patents_data
        SELECT 
            patent_number,
            title,
            code,
            class,
            subclass,
            group_,
            main_group,
            source_fingerprint,
            ingestion_timestamp,
            source_file
        FROM temp_df;
        """)
        conn.unregister('temp_df')
        
        count = conn.execute("SELECT COUNT(*) FROM patents_data").fetchone()[0]
        logger.info(f"Stored {count} records in DuckDB")
        
    except Exception as e:
        logger.error(f"Failed to store data in DuckDB: {str(e)}")
        raise
    finally:
        conn.close()

def index_in_chroma(df: pd.DataFrame):
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        try:
            chroma_client.delete_collection(name="patents")
        except:
            pass 
        
        collection = chroma_client.create_collection(name="patents")
        
        documents = df['title'].tolist()
        
        metadatas = []
        for _, row in df.iterrows():
            metadata = {
                'patent_number': str(row['patent_number']),
                'title': str(row['title']),
                'source_fingerprint': str(row['source_fingerprint']),
                'source_file': str(row['source_file'])
            }
            metadatas.append(metadata)
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            batch_ids = [f"patent_{j}" for j in range(i, min(i+batch_size, len(documents)))]
            
            embeddings = model.encode(batch_docs).tolist()
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=batch_ids,
                embeddings=embeddings
            )
        
        logger.info(f"Indexed {len(documents)} documents in ChromaDB")
        
    except Exception as e:
        logger.error(f"Failed to index in ChromaDB: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    try:
        cleaned_df = process_data(PATENT_DATA_FILE)
        store_in_duckdb(cleaned_df)
        index_in_chroma(cleaned_df)
        
        logger.info(f"Pipeline initialized with {len(cleaned_df)} records")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise

@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """Search route: accepts query, returns top 3 matching rows with source info and quality score"""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection(name="patents")
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        query_embedding = model.encode(query.query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if not results.get('documents') or not results['documents'][0]:
            return []
        
        response = []
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [None] * len(docs))[0]
        
        for i in range(len(docs)):
            metadata = metadatas[i]
            
            quality_score = max(0, 1 - distances[i]) if distances and distances[i] is not None else 0.0
            
            source_info = {
                "source_file": metadata.get('source_file'),
                "timestamp": datetime.utcnow().isoformat(),
                "fingerprint": metadata.get('source_fingerprint')
            }
            
            result = SearchResult(
                title=str(metadata.get('title', '')),
                patent_number=str(metadata.get('patent_number', '')),
                quality_score=quality_score,
                source_info=source_info
            )
            response.append(result)
        
        logger.info(f"Search for '{query.query}' returned {len(response)} results")
        return response
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)