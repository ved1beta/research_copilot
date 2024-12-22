import torch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    PromptTemplate
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def create_rag_pipeline(
    data_path: str,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    embedding_model_name: str = "BAAI/bge-small-en",
    device: str = "cpu"
):
    """
    Create a RAG pipeline using local models.
    """
    print(f"Loading documents from {data_path}")
    documents = SimpleDirectoryReader(data_path).load_data()
    
    print("Initializing language model...")
    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as accurately as 
    possible based on the instructions and context provided.
    """
    
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    query_wrapper_prompt = PromptTemplate(template)

    # Initialize LLM with larger context window
    llm = HuggingFaceLLM(
        context_window=2048,  # Increased from 248
        max_new_tokens=512,   # Increased from 128
        generate_kwargs={
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9,
        },
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map=device,
        model_kwargs={"torch_dtype": torch.float16}
    )
    
    print("Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name=embedding_model_name,
        device=device
    )
    
    print("Configuring settings...")
    # Configure global settings with adjusted parameters
    Settings.chunk_size = 256          # Reduced from 512
    Settings.chunk_overlap = 20        # Reduced from 50
    Settings.num_output = 512
    Settings.context_window = 2048     # Match LLM context window
    Settings.num_edges = 10
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("Building vector store index...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    # Create query engine with response mode
    query_engine = index.as_query_engine(
        response_mode="compact",  # Use compact response mode
        streaming=False,
        similarity_top_k=3,      # Limit number of chunks retrieved
    )
    
    print("RAG pipeline ready!")
    return query_engine

def main():
    try:
        print("Starting RAG pipeline initialization...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Check data directory
        import os
        data_path = "../../data/uploads/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Created data directory: {data_path}")
            print("Please add some documents to the directory and run again.")
            return
        
        if not os.listdir(data_path):
            print(f"No files found in data directory: {data_path}")
            print("Please add some documents to the directory and run again.")
            return
            
        print(f"Found {len(os.listdir(data_path))} files in data directory")
        
        query_engine = create_rag_pipeline(
            data_path=data_path,
            device=device
        )
        
        print("\nSystem is ready for questions!")
        print("--------------------------------")
        
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            print("\nProcessing your question...")
            try:
                response = query_engine.query(query)
                print(f"\nResponse: {response}")
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                print("Please try a different question or rephrase your question.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()