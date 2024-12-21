# src/database.py

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, DateTime, ARRAY, Text

def setup_database(postgres_url: str):
    """Setup PostgreSQL database schema"""
    engine = create_engine(postgres_url)
    metadata = MetaData()
    
    # Define papers table
    papers = Table('papers', metadata,
        Column('paper_id', String, primary_key=True),
        Column('title', String),
        Column('sections_found', ARRAY(String)),
        Column('total_pages', Integer),
        Column('processed_date', DateTime),
        Column('filename', String)
    )
    
    # Create tables
    metadata.create_all(engine)
    return engine