import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import uuid

class PaperProcessor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="paper_chunks",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    def create_chunks(self, text, metadata):
        """Split text while preserving metadata"""
        chunks = []
        
        # Combine title and abstract for complete context
        full_text = f"{metadata['title']} {text}"
        
        # Create chunks with overlap
        for i in range(0, len(full_text), self.chunk_size // 2):
            chunk = full_text[i:i + self.chunk_size]
            if len(chunk) >= 100:  # Minimum chunk size
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'text': chunk,
                    'metadata': {
                        'paper_id': metadata['id'],
                        'title': metadata['title'],
                        'authors': metadata['authors'],
                        'categories': metadata['categories'],
                        'chunk_start': i,
                        'chunk_end': i + len(chunk)
                    }
                })
        return chunks

    def process_papers(self, json_file):
        """Process JSON papers file"""
        processed = 0
        
        with open(json_file, 'r') as f:
            for line in f:
                try:
                    paper = json.loads(line)
                    
                    # Create chunks from abstract
                    if paper.get('abstract'):
                        chunks = self.create_chunks(paper['abstract'], paper)
                        
                        # Prepare data for ChromaDB
                        ids = [chunk['id'] for chunk in chunks]
                        texts = [chunk['text'] for chunk in chunks]
                        metadatas = [chunk['metadata'] for chunk in chunks]
                        
                        # Add to ChromaDB
                        if chunks:
                            self.collection.upsert(
                                ids=ids,
                                documents=texts,
                                metadatas=metadatas
                            )
                            
                            processed += 1
                            if processed % 100 == 0:
                                print(f"Processed {processed} papers")
                                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing paper: {e}")
                    continue
        
        print(f"Processing complete. Total papers processed: {processed}")

    def query_papers(self, query_text, n_results=5):
        """Search for similar chunks"""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

# Usage
processor = PaperProcessor(chunk_size=1000)
processor.process_papers('arxiv-metadata-oai-snapshot.json')