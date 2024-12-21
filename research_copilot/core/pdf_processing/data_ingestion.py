# src/data_ingestion.py

import os
from typing import Dict, List, Any
from datetime import datetime
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.http import models

class DataIngestion:
    def __init__(
        self,
        postgres_url: str,
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "research_papers"
    ):
        # Initialize database connections
        self.db_engine = create_engine(postgres_url)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        # Initialize Qdrant client
        self.vector_db = QdrantClient(
            host=qdrant_url,
            port=qdrant_port
        )
        self.collection_name = collection_name
        
        # Initialize SPECTER model for scientific paper embeddings
        self.embedding_model = SentenceTransformer('allenai/specter')
        
        # Create Qdrant collection if it doesn't exist
        self._setup_vector_db()
    
    def _setup_vector_db(self):
        """Setup Qdrant collection"""
        try:
            self.vector_db.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # SPECTER embedding dimension
                    distance=models.Distance.COSINE
                )
            )
        except Exception as e:
            print(f"Collection might already exist: {e}")
    
    def _generate_paper_id(self, metadata: Dict) -> str:
        """Generate unique ID for paper based on metadata"""
        unique_string = f"{metadata.get('title', '')}-{metadata.get('authors', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        return self.embedding_model.encode(chunks, show_progress_bar=True)
    
    def store_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Store paper data in both PostgreSQL and Qdrant
        """
        try:
            # Generate unique paper ID
            paper_id = self._generate_paper_id(paper_data['metadata'])
            
            # Store metadata in PostgreSQL
            metadata = paper_data['metadata']
            metadata['paper_id'] = paper_id
            metadata['processed_date'] = datetime.now()
            
            # Insert into PostgreSQL
            self.db_session.execute(
                """
                INSERT INTO papers (
                    paper_id, title, sections_found, total_pages, 
                    processed_date, filename
                ) VALUES (
                    :paper_id, :title, :sections_found, :total_pages,
                    :processed_date, :filename
                )
                ON CONFLICT (paper_id) DO UPDATE SET
                    processed_date = EXCLUDED.processed_date
                """,
                metadata
            )
            self.db_session.commit()
            
            # Process text and generate embeddings
            full_text = paper_data.get('full_text', '')
            chunks = self._chunk_text(full_text)
            embeddings = self._generate_embeddings(chunks)
            
            # Store in Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append(models.PointStruct(
                    id=f"{paper_id}_{idx}",
                    vector=embedding.tolist(),
                    payload={
                        'paper_id': paper_id,
                        'chunk_index': idx,
                        'text': chunk,
                        'metadata': {
                            'title': metadata.get('title', ''),
                            'sections_found': metadata.get('sections_found', []),
                            'total_pages': metadata.get('total_pages', 0)
                        }
                    }
                ))
            
            # Upload to Qdrant
            self.vector_db.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return paper_id
            
        except Exception as e:
            self.db_session.rollback()
            raise Exception(f"Error storing paper: {str(e)}")
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar content using query
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in Qdrant
        results = self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for res in results:
            formatted_results.append({
                'text': res.payload['text'],
                'metadata': res.payload['metadata'],
                'similarity_score': res.score
            })
        
        return formatted_results
    
    def close(self):
        """Close database connections"""
        self.db_session.close()