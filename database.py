"""Database setup and utilities for the Chinook music store database."""

import sqlite3
import requests
from pathlib import Path
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def download_chinook_database(db_path: str = "chinook.db") -> None:
    """
    Download the Chinook database SQL script and create a local SQLite database.
    
    Args:
        db_path: Path where the database file should be saved
    """
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    
    print(f"Downloading Chinook database from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    sql_script = response.text
    
    # Remove existing database if it exists
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()
    
    print(f"Creating database at {db_path}...")
    connection = sqlite3.connect(db_path)
    connection.executescript(sql_script)
    connection.commit()
    connection.close()
    
    print(f"Database successfully created at {db_path}")


def get_engine_for_chinook_db(db_path: str = "chinook.db"):
    """
    Create a SQLAlchemy engine for the Chinook database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        SQLAlchemy Engine instance
    """
    # Check if database exists, if not download it
    if not Path(db_path).exists():
        print(f"Database not found at {db_path}, downloading...")
        download_chinook_database(db_path)
    
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    return engine


def get_database(db_path: str = "chinook.db") -> SQLDatabase:
    """
    Get a LangChain SQLDatabase instance for the Chinook database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        LangChain SQLDatabase instance
    """
    engine = get_engine_for_chinook_db(db_path)
    return SQLDatabase(engine)


if __name__ == "__main__":
    # Download and create the database when run directly
    download_chinook_database()
    db = get_database()
    print("\nAvailable tables:")
    print(db.get_usable_table_names())
