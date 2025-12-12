"""
Account management tools that require verification.
These tools handle sensitive account updates with proper security.
"""

from langchain_core.tools import tool
from database import get_database
from verification import get_verification_service
from guardrails import execute_secure_query

# Initialize database
db = get_database()

# Customer ID for demo
CUSTOMER_ID = 58


# =======================
# VERIFICATION TOOLS
# =======================

@tool
def request_phone_verification():
    """
    Start the phone verification process by sending an SMS code.
    This MUST be called before any account changes (email, address, etc.).
    
    Returns status of verification request.
    """
    # Get customer's phone number - db.run returns a string, not a list
    try:
        # Use secure query execution with guardrails
        with db.get_secure_connection() as conn:
            result = execute_secure_query(conn, f"SELECT Phone FROM Customer WHERE CustomerId = {CUSTOMER_ID}", CUSTOMER_ID)
            row = result.fetchone()
            
        if not row:
            return "Error: Customer not found"
        
        phone = row[0]
        
        if not phone:
            return "Error: No phone number on file. Please contact support."
        
        # Send verification code
        send_result = get_verification_service().send_verification_code(phone, CUSTOMER_ID)
        
        if send_result['success']:
            return f"✅ Verification code sent to {phone}. Please check your phone and enter the code when you receive it."
        else:
            return f"❌ Failed to send verification code: {send_result['message']}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


@tool
def verify_phone_code(verification_code: str):
    """
    Verify the SMS code that was sent to the customer's phone.
    
    Args:
        verification_code: The 6-digit code received via SMS
        
    Returns verification result.
    """
    # Get customer's phone number
    try:
        with db.get_secure_connection() as conn:
            result = execute_secure_query(conn, f"SELECT Phone FROM Customer WHERE CustomerId = {CUSTOMER_ID}", CUSTOMER_ID)
            row = result.fetchone()
            
        if not row:
            return "Error: Customer not found"
        
        phone = row[0]
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
    # Verify the code
    verify_result = get_verification_service().verify_code(phone, verification_code, CUSTOMER_ID)
    
    if verify_result['success']:
        return "✅ Phone verified successfully! You can now update your account information."
    else:
        message = verify_result['message']
        if 'attempts_remaining' in verify_result:
            return f"❌ {message}"
        elif verify_result.get('max_attempts_exceeded'):
            return "❌ Too many failed attempts. Please request a new verification code."
        else:
            return f"❌ {message}"


@tool
def check_verification_status():
    """
    Check if the customer is currently verified in this session.
    Returns whether phone verification is still active.
    """
    is_verified = get_verification_service().is_verified(CUSTOMER_ID)
    
    if is_verified:
        return "✅ You are verified and can make account changes."
    else:
        return "❌ Not verified. Please request a verification code first."


# =======================
# ACCOUNT UPDATE TOOLS
# (These require verification)
# =======================

@tool
def update_email_address(new_email: str):
    """
    Update the customer's email address. REQUIRES PHONE VERIFICATION.
    
    Args:
        new_email: The new email address to set
        
    Returns result of the update operation.
    """
    # Check if verified
    if not get_verification_service().is_verified(CUSTOMER_ID):
        return "❌ Security check required. You must verify your phone number before updating your email address. Please request a verification code first."
    
    # Validate email format (basic)
    if '@' not in new_email or '.' not in new_email:
        return "❌ Invalid email format. Please provide a valid email address."
    
    try:
        # Get current email
        with db.get_secure_connection() as conn:
            result = execute_secure_query(
                conn, 
                "SELECT Email FROM Customer WHERE CustomerId = :customer_id",
                CUSTOMER_ID,
                {"customer_id": CUSTOMER_ID}
            )
            row = result.fetchone()
            current_email = row[0] if row else "unknown"
        
            # Update email in database (validation ensures customer ID filter)
            execute_secure_query(
                conn,
                "UPDATE Customer SET Email = :new_email WHERE CustomerId = :customer_id",
                CUSTOMER_ID,
                {"new_email": new_email, "customer_id": CUSTOMER_ID}
            )
            conn.commit()
            
            # Verify the update
            result = execute_secure_query(
                conn,
                "SELECT Email FROM Customer WHERE CustomerId = :customer_id",
                CUSTOMER_ID,
                {"customer_id": CUSTOMER_ID}
            )
            row = result.fetchone()
            updated_email = row[0] if row else None
        
        if updated_email == new_email:
            return f"✅ Email address successfully updated!\n\n📧 Old email: {current_email}\n📧 New email: {updated_email}\n\nYour email has been changed in our system."
        else:
            return "❌ Email update failed. Please try again or contact support."
            
    except Exception as e:
        return f"❌ Error updating email: {str(e)}"


@tool
def update_mailing_address(street_address: str, city: str, state: str = None, postal_code: str = None, country: str = None):
    """
    Update the customer's mailing address. REQUIRES PHONE VERIFICATION.
    
    Args:
        street_address: Street address
        city: City
        state: State/Province (optional)
        postal_code: Postal/ZIP code (optional)
        country: Country (optional)
        
    Returns result of the update operation.
    """
    # Check if verified
    if not get_verification_service().is_verified(CUSTOMER_ID):
        return "❌ Security check required. You must verify your phone number before updating your address. Please request a verification code first."
    
    try:
        # Get current address
        with db.get_secure_connection() as conn:
            result = execute_secure_query(
                conn,
                "SELECT Address, City, State, PostalCode, Country FROM Customer WHERE CustomerId = :customer_id",
                CUSTOMER_ID,
                {"customer_id": CUSTOMER_ID}
            )
            row = result.fetchone()
            current_address = {
                'Address': row[0] if row else '',
                'City': row[1] if row else '',
                'State': row[2] if row else '',
                'PostalCode': row[3] if row else '',
                'Country': row[4] if row else ''
            }
        
            # Build update statement with parameters (avoid SQL injection)
            params = {
                "customer_id": CUSTOMER_ID,
                "address": street_address,
                "city": city,
                "state": state,
                "postal_code": postal_code,
                "country": country,
            }

            execute_secure_query(
                conn,
                """
                UPDATE Customer
                SET
                    Address = :address,
                    City = :city,
                    State = COALESCE(:state, State),
                    PostalCode = COALESCE(:postal_code, PostalCode),
                    Country = COALESCE(:country, Country)
                WHERE CustomerId = :customer_id
                """,
                CUSTOMER_ID,
                params
            )
            conn.commit()
            
            # Verify the update
            result = execute_secure_query(
                conn,
                "SELECT Address, City, State, PostalCode, Country FROM Customer WHERE CustomerId = :customer_id",
                CUSTOMER_ID,
                {"customer_id": CUSTOMER_ID}
            )
            row = result.fetchone()
            new_address = {
                'Address': row[0] if row else '',
                'City': row[1] if row else '',
                'State': row[2] if row else '',
                'PostalCode': row[3] if row else '',
                'Country': row[4] if row else ''
            }
        
        # Format response
        old_addr = f"{current_address.get('Address', '')}, {current_address.get('City', '')}, {current_address.get('State', '')} {current_address.get('PostalCode', '')}"
        new_addr = f"{new_address['Address']}, {new_address['City']}, {new_address.get('State', '')} {new_address.get('PostalCode', '')}"
        
        return f"✅ Mailing address successfully updated!\n\n📍 Old address: {old_addr}\n📍 New address: {new_addr}\n\nYour address has been changed in our system."
            
    except Exception as e:
        return f"❌ Error updating address: {str(e)}"


# Export all tools
VERIFICATION_TOOLS = [
    request_phone_verification,
    verify_phone_code,
    check_verification_status,
]

ACCOUNT_UPDATE_TOOLS = [
    update_email_address,
    update_mailing_address,
]

ALL_ACCOUNT_TOOLS = VERIFICATION_TOOLS + ACCOUNT_UPDATE_TOOLS
