import chromadb
from groq import Groq
from typing import List

class RAGResponseGenerator:
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.groq_client = Groq(
            api_key="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"
        )
        
        try:
            self.collection = self.client.get_collection("ml_papers")
        except ValueError:
            print("Collection not found. Please make sure you've added documents to ChromaDB first.")
            raise
    
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
            context_chunks = self.get_relevant_chunks(query)
            prompt = self.generate_prompt(query, context_chunks)
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192"
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            return f"Error generating response: {str(e)}"

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
    print("Checking ChromaDB contents...")
    check_chromadb_contents("./chroma_db")
    
    try:
        generator = RAGResponseGenerator()
        query = "What is a neural network?"
        print(f"\nProcessing query: {query}")
        response = generator.generate_response(query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error initializing or running generator: {str(e)}")