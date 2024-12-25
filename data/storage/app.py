from flask import Flask, render_template, request, jsonify
from  response_gen import RAGResponseGenerator
 
import chromadb

app = Flask(__name__)

# Initialize RAG generator
generator = RAGResponseGenerator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query', '')
    
    try:
        # Get response from RAG
        response = generator.generate_response(user_query)
        
        # Get related papers
        results = generator.collection.query(
            query_texts=[user_query],
            n_results=3
        )
        
        papers = []
        if results['documents'][0]:
            for i in range(len(results['documents'][0])):
                paper = {
                    'title': results['metadatas'][0][i]['title'],
                    'authors': results['metadatas'][0][i]['authors'],
                    'paper_id': results['metadatas'][0][i]['paper_id'],
                    'url': f"https://arxiv.org/abs/{results['metadatas'][0][i]['paper_id']}",
                    'similarity': f"{(1 - results['distances'][0][i]):.2f}"
                }
                papers.append(paper)
        
        return jsonify({
            'response': response,
            'papers': papers
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)