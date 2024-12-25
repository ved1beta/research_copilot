from flask import Flask, render_template, request, jsonify
from response_gen import RAGResponseGenerator
import chromadb
from chromadb.config import Settings
import gc
import os
import logging
from functools import lru_cache
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.setup_error_handlers()
        self.initialize_rag()
        self.setup_signal_handlers()

    def initialize_rag(self):
        """Initialize RAG with proper error handling and ChromaDB configuration"""
        try:
            # Configure ChromaDB for minimal resource usage
            chroma_settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory="./chroma_db",
                allow_reset=False
            )

            # Initialize RAG generator with resource constraints
            self.generator = RAGResponseGenerator()
            
            # Verify ChromaDB connection
            collection_count = len(self.generator.collection.get()['ids'])
            logger.info(f"Successfully initialized ChromaDB with {collection_count} documents")

        except Exception as e:
            logger.error(f"Failed to initialize RAG: {str(e)}")
            sys.exit(1)

    def setup_routes(self):
        """Setup Flask routes"""
        self.app.route('/')(self.home)
        self.app.route('/query', methods=['POST'])(self.query)

    def setup_error_handlers(self):
        """Setup custom error handlers"""
        @self.app.errorhandler(500)
        def handle_500(e):
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

        @self.app.errorhandler(413)
        def handle_413(e):
            return jsonify({'error': 'Request too large'}), 413

    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self.cleanup_handler)
        signal.signal(signal.SIGINT, self.cleanup_handler)

    def cleanup_handler(self, signum, frame):
        """Handle cleanup when shutting down"""
        logger.info("Cleaning up resources...")
        gc.collect()
        sys.exit(0)

    def home(self):
        """Serve the home page"""
        return render_template('index.html')

    @lru_cache(maxsize=100)
    def cached_query(self, query_text):
        """Cache frequent queries to reduce database load"""
        return self.generator.collection.query(
            query_texts=[query_text],
            n_results=3
        )

    def query(self):
        """Handle query requests"""
        try:
            user_query = request.json.get('query', '')
            if not user_query:
                return jsonify({'error': 'Empty query'}), 400

            # Generate response
            response = self.generator.generate_response(user_query)

            # Get related papers using cached query
            results = self.cached_query(user_query)
            
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

            # Clean up memory after processing
            gc.collect()
            
            return jsonify({
                'response': response,
                'papers': papers
            })

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return jsonify({'error': 'Failed to process query', 'details': str(e)}), 500

# Create the application instance
flask_app = FlaskApp()
app = flask_app.app

if __name__ == '__main__':
    # For development only
    app.run(debug=False, host='0.0.0.0')