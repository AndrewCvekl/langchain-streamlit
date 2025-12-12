"""Database setup and utilities for the Chinook music store database."""

import sqlite3
import requests
from pathlib import Path
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from guardrails import validate_sql_query, CURRENT_CUSTOMER_ID


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


class SecureSQLDatabase(SQLDatabase):
    """
    Wrapper around SQLDatabase that adds security guardrails.
    Validates all SQL queries before execution.
    """
    
    def __init__(self, engine, customer_id: int = CURRENT_CUSTOMER_ID):
        super().__init__(engine)
        self.customer_id = customer_id
    
    def run(self, command: str, fetch: str = "all", include_columns: bool = False, **kwargs):
        """
        Run a SQL command with security validation.
        
        Args:
            command: SQL command to execute
            fetch: How to fetch results
            include_columns: Whether to include column names
            **kwargs: Additional arguments
            
        Returns:
            Query results or error message if validation fails
        """
        # Validate the query
        is_valid, error_message = validate_sql_query(command, self.customer_id)
        
        if not is_valid:
            # Return error message instead of executing query
            return error_message
        
        # Query is valid, proceed with execution
        try:
            return super().run(command, fetch=fetch, include_columns=include_columns, **kwargs)
        except Exception as e:
            # Re-validate in case of unexpected issues
            logger = __import__('logging').getLogger(__name__)
            logger.error(f"Database error: {str(e)}")
            return f"❌ Database error: {str(e)}"
    
    def get_secure_connection(self):
        """
        Get a database connection with guardrails.
        Use this instead of direct _engine.connect() for secure queries.
        """
        return self._engine.connect()


def get_database(db_path: str = "chinook.db", customer_id: int = None) -> SecureSQLDatabase:
    """
    Get a secure SQLDatabase instance for the Chinook database with guardrails.
    
    Args:
        db_path: Path to the SQLite database file
        customer_id: Current customer ID for access control (defaults to CURRENT_CUSTOMER_ID from guardrails)
        
    Returns:
        SecureSQLDatabase instance with validation
    """
    if customer_id is None:
        customer_id = CURRENT_CUSTOMER_ID
    engine = get_engine_for_chinook_db(db_path)
    return SecureSQLDatabase(engine, customer_id=customer_id)


if __name__ == "__main__":
    # Download and create the database when run directly
    download_chinook_database()
    db = get_database()
    print("\nAvailable tables:")
    print(db.get_usable_table_names())
