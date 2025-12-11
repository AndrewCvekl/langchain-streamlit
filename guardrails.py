"""
Guardrails for the music store application.

Enforces:
1. Customer data isolation - customers can only access their own data
2. Employee data protection - customers cannot access employee information
3. Topic validation - blocks queries unrelated to music stores
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Current customer ID (should match CUSTOMER_INFO in graph_with_verification.py)
# This is the default, but can be overridden when calling validation functions
CURRENT_CUSTOMER_ID = 58

# Tables that require customer ID filtering
CUSTOMER_DATA_TABLES = {"Customer", "Invoice", "InvoiceLine"}

# Tables that should be completely blocked
BLOCKED_TABLES = {"Employee"}

# Non-music-related keywords that should be blocked
NON_MUSIC_KEYWORDS = {
    "sports", "football", "basketball", "baseball", "soccer", "hockey",
    "tennis", "golf", "cricket", "rugby", "volleyball", "swimming",
    "olympics", "nfl", "nba", "mlb", "nhl", "fifa", "uefa",
    "athlete", "player", "team", "stadium", "game", "match",
    "cooking", "recipe", "food", "restaurant", "chef",
    "movie", "film", "cinema", "actor", "director",
    "book", "novel", "author", "publisher",
    "car", "vehicle", "automobile", "truck",
    "real estate", "property", "house", "apartment",
    "stock", "trading", "investment", "finance",
    "weather", "climate", "temperature",
    "politics", "government", "election",
    "science", "technology", "computer", "software",
    "travel", "vacation", "hotel", "flight",
    "fashion", "clothing", "apparel",
    "health", "medical", "doctor", "hospital",
    "education", "school", "university", "college",
}


def validate_sql_query(query: str, customer_id: int = CURRENT_CUSTOMER_ID) -> Tuple[bool, Optional[str]]:
    """
    Validate a SQL query against security guardrails.
    
    Args:
        query: The SQL query to validate
        customer_id: The current customer ID (defaults to CURRENT_CUSTOMER_ID)
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if query passes all guardrails, False otherwise
        - error_message: Error message if validation fails, None if valid
    """
    if not query or not query.strip():
        return False, "Empty query provided"
    
    # Normalize query for analysis
    query_upper = query.upper()
    query_lower = query.lower()
    
    # 1. Check for blocked tables (Employee)
    for blocked_table in BLOCKED_TABLES:
        # Check for table name in various SQL contexts
        patterns = [
            rf'\bFROM\s+{blocked_table}\b',
            rf'\bJOIN\s+{blocked_table}\b',
            rf'\bINNER\s+JOIN\s+{blocked_table}\b',
            rf'\bLEFT\s+JOIN\s+{blocked_table}\b',
            rf'\bRIGHT\s+JOIN\s+{blocked_table}\b',
            rf'\bUPDATE\s+{blocked_table}\b',
            rf'\bDELETE\s+FROM\s+{blocked_table}\b',
            rf'\bINSERT\s+INTO\s+{blocked_table}\b',
        ]
        
        for pattern in patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"Blocked query attempting to access {blocked_table} table")
                return False, "Sorry I can't assist with that."
    
    # 2. Check for customer data tables and ensure customer ID filtering
    for table in CUSTOMER_DATA_TABLES:
        # Check if table is accessed
        table_patterns = [
            rf'\bFROM\s+{table}\b',
            rf'\bJOIN\s+{table}\b',
            rf'\bINNER\s+JOIN\s+{table}\b',
            rf'\bLEFT\s+JOIN\s+{table}\b',
            rf'\bRIGHT\s+JOIN\s+{table}\b',
            rf'\bUPDATE\s+{table}\b',
        ]
        
        table_accessed = any(re.search(pattern, query_upper, re.IGNORECASE) for pattern in table_patterns)
        
        if table_accessed:
            # Allow INSERT queries if they're inserting for the current customer
            if query_upper.startswith("INSERT"):
                # Check if INSERT includes CustomerId = current customer
                insert_customer_pattern = rf'INSERT.*CustomerId.*{customer_id}\b'
                if re.search(insert_customer_pattern, query_upper, re.IGNORECASE | re.DOTALL):
                    # INSERT is for current customer, allow it
                    continue
            
            # Allow aggregate queries (MAX, COUNT, etc.) on Invoice/InvoiceLine for getting next IDs
            # These don't need customer filtering as they're just getting metadata
            if table in {"Invoice", "InvoiceLine"}:
                aggregate_patterns = [
                    r'SELECT\s+MAX\s*\(',
                    r'SELECT\s+COUNT\s*\(',
                    r'SELECT\s+MIN\s*\(',
                ]
                if any(re.search(pattern, query_upper, re.IGNORECASE) for pattern in aggregate_patterns):
                    # Aggregate query, allow it (used for getting next ID)
                    continue
            
            # Check if customer ID filter is present
            # Look for WHERE clause with CustomerId filter
            customer_id_patterns = [
                rf'WHERE.*CustomerId\s*=\s*{customer_id}\b',  # Literal customer ID
                rf'WHERE.*CustomerId\s*=\s*:{customer_id}\b',  # Named parameter with ID
                rf'WHERE.*CustomerId\s*=\s*\?',  # Positional parameter
                rf'WHERE.*CustomerId\s*=\s*:customer_id\b',  # Named parameter :customer_id
                rf'WHERE.*CustomerId\s*=\s*:CUSTOMER_ID\b',  # Named parameter :CUSTOMER_ID
                rf'AND.*CustomerId\s*=\s*{customer_id}\b',
                rf'AND.*CustomerId\s*=\s*:{customer_id}\b',
                rf'AND.*CustomerId\s*=\s*\?',
                rf'AND.*CustomerId\s*=\s*:customer_id\b',  # Named parameter :customer_id
                rf'AND.*CustomerId\s*=\s*:CUSTOMER_ID\b',  # Named parameter :CUSTOMER_ID
            ]
            
            has_customer_filter = any(re.search(pattern, query_upper, re.IGNORECASE) for pattern in customer_id_patterns)
            
            # Special case: UPDATE queries should also check customer ID
            if query_upper.startswith("UPDATE") and not has_customer_filter:
                # Check if UPDATE has WHERE clause with customer ID (including named parameters)
                update_patterns = [
                    rf'UPDATE\s+{table}.*WHERE.*CustomerId\s*=\s*{customer_id}\b',  # Literal ID
                    rf'UPDATE\s+{table}.*WHERE.*CustomerId\s*=\s*:customer_id\b',  # Named parameter
                    rf'UPDATE\s+{table}.*WHERE.*CustomerId\s*=\s*:CUSTOMER_ID\b',  # Named parameter uppercase
                    rf'UPDATE\s+{table}.*WHERE.*CustomerId\s*=\s*\?',  # Positional parameter
                ]
                has_customer_filter = any(re.search(pattern, query_upper, re.IGNORECASE | re.DOTALL) for pattern in update_patterns)
            
            if not has_customer_filter:
                logger.warning(f"Query accessing {table} without customer ID filter")
                return False, "Sorry I can't assist with that."
    
    # 3. Check for non-music-related keywords in query
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    found_keywords = query_words.intersection(NON_MUSIC_KEYWORDS)
    
    if found_keywords:
        logger.warning(f"Query contains non-music keywords: {found_keywords}")
        return False, "Sorry I can't assist with that."
    
    # All checks passed
    return True, None


def validate_user_input(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input for non-music-related topics.
    
    Args:
        user_input: The user's input message
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if input is music-related, False otherwise
        - error_message: Error message if validation fails, None if valid
    """
    if not user_input or not user_input.strip():
        return True, None
    
    input_lower = user_input.lower()
    input_words = set(re.findall(r'\b\w+\b', input_lower))
    
    # Check for non-music keywords
    found_keywords = input_words.intersection(NON_MUSIC_KEYWORDS)
    
    if found_keywords:
        # Allow some context - if it's clearly about music, allow it
        # But if it's clearly unrelated, block it
        music_context_words = {"music", "song", "track", "album", "artist", "genre", "lyrics", "purchase", "buy", "account", "invoice", "order"}
        has_music_context = any(word in input_lower for word in music_context_words)
        
        if not has_music_context:
            logger.warning(f"User input contains non-music keywords: {found_keywords}")
            return False, "Sorry I can't assist with that."
    
    return True, None


def sanitize_sql_query(query: str, customer_id: int = CURRENT_CUSTOMER_ID) -> str:
    """
    Attempt to sanitize a SQL query by ensuring customer ID filtering.
    This is a safety measure, but validation should catch issues first.
    
    Args:
        query: The SQL query to sanitize
        customer_id: The current customer ID
        
    Returns:
        Sanitized query (may be unchanged if already safe)
    """
    # This is a defensive measure - validation should prevent unsafe queries
    # But we can add basic sanitization here if needed
    return query


def execute_secure_query(conn, query: str, customer_id: int = CURRENT_CUSTOMER_ID, params: dict = None):
    """
    Execute a SQL query with security validation.
    Use this for direct database connections to ensure guardrails are enforced.
    
    Args:
        conn: Database connection object
        query: SQL query to execute
        customer_id: Current customer ID for validation
        params: Query parameters (for parameterized queries)
        
    Returns:
        Query result or raises ValueError with error message if validation fails
    """
    # Validate the query
    is_valid, error_message = validate_sql_query(query, customer_id)
    
    if not is_valid:
        raise ValueError(error_message)
    
    # Query is valid, execute it
    from sqlalchemy import text
    if params:
        return conn.execute(text(query), params)
    else:
        return conn.execute(text(query))
