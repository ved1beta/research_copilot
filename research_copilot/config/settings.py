import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    
    # Database settings
    POSTGRES_URI = os.getenv('POSTGRES_URI', 'postgresql://localhost:5432/research_copilot')
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT = os.getenv('QDRANT_PORT', 6333)
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Redis settings
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Celery settings
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL