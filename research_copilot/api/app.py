# app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    PromptTemplate
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

app = Flask(__name__)

def initialize_rag():
    """Initialize the RAG pipeline"""
    data_path = "../../data/uploads/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        return None

    if not os.listdir(data_path):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    system_prompt = """You are a Q&A assistant. Your goal is to answer questions accurately based on the context provided."""
    
    template = (
        "Context information:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    query_wrapper_prompt = PromptTemplate(template)

    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": True, "top_p": 0.9},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map=device,
        model_kwargs={"torch_dtype": torch.float16}
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en",
        device=device
    )
    
    Settings.chunk_size = 256
    Settings.chunk_overlap = 20
    Settings.num_output = 512
    Settings.context_window = 2048
    Settings.num_edges = 10
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    return index.as_query_engine(
        response_mode="compact",
        streaming=False,
        similarity_top_k=3,
    )

# Global query engine
query_engine = None

@app.route('/')
def home():
    global query_engine
    if query_engine is None:
        query_engine = initialize_rag()
    return render_template('index.html', has_documents=query_engine is not None)

@app.route('/ask', methods=['POST'])
def ask():
    global query_engine
    
    if query_engine is None:
        query_engine = initialize_rag()
        if query_engine is None:
            return jsonify({'error': 'No documents found in data/uploads/. Please add documents first.'})
    
    user_question = request.form.get('question', '')
    if not user_question:
        return jsonify({'error': 'Please enter a question'})
        
    try:
        response = query_engine.query(user_question)
        return jsonify({'response': str(response)})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)