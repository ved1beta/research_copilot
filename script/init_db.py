from research_copilot.database.vector_store import init_qdrant
from research_copilot.database.graph_store import init_neo4j
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from research_copilot.config.settings import Config

def init_postgres():
    """Initialize PostgreSQL database with required tables"""
    conn = psycopg2.connect(Config.POSTGRES_URI)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create necessary tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            authors TEXT[],
            abstract TEXT,
            pdf_path VARCHAR(255),
            publication_date DATE,
            journal VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS sections (
            id SERIAL PRIMARY KEY,
            paper_id INTEGER REFERENCES papers(id),
            section_type VARCHAR(50),
            content TEXT,
            embedding_id VARCHAR(255)
        );
        
        CREATE TABLE IF NOT EXISTS citations (
            id SERIAL PRIMARY KEY,
            paper_id INTEGER REFERENCES papers(id),
            cited_paper_id INTEGER REFERENCES papers(id),
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cur.close()
    conn.close()

def init_databases():
    """Initialize all databases"""
    print("Initializing PostgreSQL...")
    init_postgres()
    
    print("Initializing Qdrant vector store...")
    init_qdrant()
    
    print("Initializing Neo4j graph database...")
    init_neo4j()
    
    print("Database initialization complete!")

if __name__ == "__main__":
    init_databases()