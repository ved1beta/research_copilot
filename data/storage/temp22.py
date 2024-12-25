import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
from urllib.error import URLError
import re
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import uuid
from datetime import datetime, timedelta

class MLPapersPipeline:
    def __init__(self, pdf_dir="ml_papers", chunk_size=1000, chunk_overlap=200):
        # Initialize directories
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            
        # Initialize arXiv parameters with simpler categories for testing
        self.base_url = "http://export.arxiv.org/api/query?"
        self.cs_categories = [  'cs.LG', 'cs.CL', 'cs.CV']  # Simplified for initial testing
        self.ml_keywords = ['machine learning', 'deep learning', 'neural network'
        ]
        self.results_per_query = 100
        self.wait_time = 3
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="ml_papers",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        
        # Get already processed papers
        self.processed_ids = self._get_processed_ids()
        print(f"Found {len(self.processed_ids)} already processed papers")

    def _get_processed_ids(self):
        try:
            existing_metadata = self.collection.get()['metadatas']
            return set(meta['paper_id'] for meta in existing_metadata if meta)
        except:
            return set()

    def build_query_url(self, start=0, max_results=100):
        """Build query URL with proper formatting"""
        # Build category query
        cat_query = ' OR '.join(f'cat:{cat}' for cat in self.cs_categories)
        
        # Add date constraint for recent papers
        date_start = (datetime.now() - timedelta(days=2000)).strftime('%Y%m%d')
        date_end = datetime.now().strftime('%Y%m%d')
        date_query = f"submittedDate:[{date_start}0000 TO {date_end}2359]"
        
        # Combine queries
        full_query = f"({cat_query}) AND ({date_query})"
        
        params = {
            'search_query': full_query,
            'start': start,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        query_url = f"{self.base_url}{urllib.parse.urlencode(params)}"
        print(f"Query URL: {query_url}")
        return query_url

    def process_pdf(self, pdf_path, paper_metadata):
        """Process single PDF and create chunks"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            text = re.sub(r'\s+', ' ', text).strip()
            
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if len(chunk) >= 100:
                    chunk_id = str(uuid.uuid4())
                    metadata = {
                        'paper_id': paper_metadata.get('id'),
                        'title': paper_metadata.get('title'),
                        'authors': ', '.join(paper_metadata.get('authors', [])),
                        'categories': ', '.join(paper_metadata.get('categories', [])),
                        'chunk_index': len(chunks),
                        'chunk_start': i,
                        'chunk_end': i + len(chunk)
                    }
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk,
                        'metadata': metadata
                    })
            
            return chunks
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return None

    def download_and_process_papers(self, batch_size=10):
        """Download and process papers with better error handling and logging"""
        print(f"\nStarting batch download of {batch_size} papers...")
        
        try:
            url = self.build_query_url(max_results=batch_size)
            with urllib.request.urlopen(url) as response:
                tree = ET.fromstring(response.read().decode('utf-8'))
            
            processed_count = 0
            for entry in tree.findall('{http://www.w3.org/2005/Atom}entry'):
                try:
                    # Extract paper ID
                    paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                    if paper_id in self.processed_ids:
                        print(f"Skipping already processed paper: {paper_id}")
                        continue
                    
                    # Extract title and authors
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
                              for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
                    
                    # Find PDF link
                    pdf_link = None
                    for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                        if link.get('title') == 'pdf':
                            pdf_link = link.get('href')
                            break
                    
                    if pdf_link:
                        # Download and process PDF
                        pdf_path = os.path.join(self.pdf_dir, f"{paper_id}.pdf")
                        print(f"\nDownloading: {title[:100]}...")
                        
                        pdf_link = pdf_link.replace('abs', 'pdf')
                        urllib.request.urlretrieve(pdf_link, pdf_path)
                        
                        paper_metadata = {
                            'id': paper_id,
                            'title': title,
                            'authors': authors,
                            'categories': [cat.strip() for cat in 
                                         entry.find('{http://arxiv.org/schemas/atom}primary_category')
                                         .get('term').split(',')]
                        }
                        
                        chunks = self.process_pdf(pdf_path, paper_metadata)
                        if chunks:
                            self.collection.add(
                                ids=[chunk['id'] for chunk in chunks],
                                documents=[chunk['text'] for chunk in chunks],
                                metadatas=[chunk['metadata'] for chunk in chunks]
                            )
                            processed_count += 1
                            print(f"Successfully processed: {title[:100]}")
                            
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                            
                        self.processed_ids.add(paper_id)
                        time.sleep(self.wait_time)
                        
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
            
            print(f"\nBatch complete. Processed {processed_count} new papers")
            return processed_count
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return 0

    def search_papers(self, query, n_results=5):
        """Search for similar papers with improved output"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'][0]:
                print("No results found")
                return
            
            print(f"\nFound {len(results['documents'][0])} matches for: {query}")
            print("-" * 80)
            
            for i in range(len(results['documents'][0])):
                print(f"\nMatch {i+1}:")
                print(f"Title: {results['metadatas'][0][i]['title']}")
                print(f"Authors: {results['metadatas'][0][i]['authors']}")
                print(f"Paper ID: {results['metadatas'][0][i]['paper_id']}")
                print(f"arXiv URL: https://arxiv.org/abs/{results['metadatas'][0][i]['paper_id']}")
                print(f"Similarity Score: {1 - results['distances'][0][i]:.4f}")
                print(f"Relevant text: {results['documents'][0][i][:200]}...")
                print("-" * 40)
                
        except Exception as e:
            print(f"Error searching papers: {e}")

def main():
    pipeline = MLPapersPipeline(pdf_dir="ml_papers")
    
    # Process papers in smaller batches
    total_batches = 1234
    batch_size = 2
    
    total_processed = 0
    for i in range(total_batches):
        print(f"\nProcessing batch {i+1} of {total_batches}...")
        processed = pipeline.download_and_process_papers(batch_size=batch_size)
        total_processed += processed
        if processed == 0:
            print("No new papers processed in this batch, moving to search...")
            break
    
    print(f"\nTotal papers processed: {total_processed}")
    
    # Example searches
    print("\nSearching for transformer papers...")
    pipeline.search_papers(
        "transformer architecture for natural language processing",
        n_results=5
    )
    
    print("\nSearching for reinforcement learning papers...")
    pipeline.search_papers(
        "deep reinforcement learning with applications",
        n_results=5
    )

if __name__ == "__main__":
    main()