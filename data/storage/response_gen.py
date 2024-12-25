from langchain_community.llms import HuggingFacePipeline  # Updated import
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict
import os

class RAGResponseGenerator:
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Get existing collection - make sure this matches your collection name
        try:
            self.collection = self.client.get_collection("ml_papers")
        except ValueError:
            print("Collection not found. Please make sure you've added documents to ChromaDB first.")
            raise
        
        # Initialize the model and tokenizer
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        # Create the pipeline with updated parameters
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,  # Enable sampling
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain wrapper
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
    
    def get_relevant_chunks(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant document chunks from ChromaDB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'][0]:
                print("No matching documents found in ChromaDB")
                return ["No relevant context found in the database."]
                
            print(f"Found {len(results['documents'][0])} relevant chunks")
            return results['documents'][0]
            
        except Exception as e:
            print(f"Error querying ChromaDB: {str(e)}")
            return ["Error retrieving context from database."]
    
    def generate_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Generate a prompt combining the context and query."""
        context = "\n".join(context_chunks)
        prompt = f"""Context information is below:
        ---------------------
        {context}
        ---------------------
        Given the context information above, answer the following question. If the context doesn't contain relevant information, provide a general response:
        Question: {query}
        Answer: """
        return prompt
    
    def generate_response(self, query: str) -> str:
        """Generate a response using RAG pipeline."""
        try:
            # Get relevant chunks
            context_chunks = self.get_relevant_chunks(query)
            
            # Generate prompt
            prompt = self.generate_prompt(query, context_chunks)
            
            # Generate response using invoke instead of direct call
            response = self.llm.invoke(prompt)
            
            return response
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            return f"Error generating response: {str(e)}"

# Debug helper function
def check_chromadb_contents(db_path: str):
    """Helper function to check ChromaDB contents"""
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    print(f"Found collections: {[col.name for col in collections]}")
    
    for collection in collections:
        print(f"\nCollection '{collection.name}':")
        try:
            all_docs = collection.get()
            print(f"Number of documents: {len(all_docs['documents'])}")
            if len(all_docs['documents']) > 0:
                print(f"First document preview: {all_docs['documents'][0][:200]}...")
        except Exception as e:
            print(f"Error accessing collection: {str(e)}")

if __name__ == "__main__":
    # Debug: Check ChromaDB contents first
    print("Checking ChromaDB contents...")
    check_chromadb_contents("./chroma_db")
    
    # Initialize the generator
    try:
        generator = RAGResponseGenerator()
        
        # Example query
        query = "What is a neural network?"
        print(f"\nProcessing query: {query}")
        response = generator.generate_response(query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error initializing or running generator: {str(e)}")