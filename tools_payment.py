"""
Payment tools for purchasing tracks and managing invoices.
Follows LangGraph best practices for tool design.
"""

from datetime import datetime
from langchain_core.tools import tool
from database import get_database
from payment_service import get_payment_service, PaymentStatus
from typing import Optional

# Initialize database and payment service
db = get_database()
payment_service = get_payment_service()

# Default customer ID for demo
DEFAULT_CUSTOMER_ID = 58


# =======================
# TRACK PURCHASE TOOLS
# =======================

@tool
def get_track_details_for_purchase(track_id: int) -> str:
    """
    Get detailed information about a track for purchase confirmation.
    Shows track name, artist, album, price, and availability.
    
    Args:
        track_id: The ID of the track to get details for
        
    Returns:
        Track details formatted for purchase confirmation
    """
    try:
        result = db.run(
            f"""
            SELECT 
                t.TrackId,
                t.Name as TrackName,
                ar.Name as ArtistName,
                a.Title as AlbumName,
                g.Name as Genre,
                t.UnitPrice,
                ROUND(t.Milliseconds / 60000.0, 2) as DurationMinutes
            FROM Track t
            LEFT JOIN Album a ON t.AlbumId = a.AlbumId
            LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
            LEFT JOIN Genre g ON t.GenreId = g.GenreId
            WHERE t.TrackId = {track_id};
            """,
            include_columns=True
        )
        
        if not result or "TrackName" not in result:
            return f"❌ Track ID {track_id} not found in our catalog."
        
        return f"""✅ Track Details:

{result}

Ready to purchase! This track costs ${result.split('UnitPrice')[1].split()[0] if 'UnitPrice' in result else 'N/A'}.
"""
    except Exception as e:
        return f"❌ Error fetching track details: {str(e)}"


@tool
def initiate_track_purchase(track_id: int, track_name: str, track_price: float) -> str:
    """
    Step 1: Create a payment intent for purchasing a track.
    This starts the payment process and generates a payment ID.
    
    Args:
        track_id: The ID of the track to purchase
        track_name: The name of the track
        track_price: The price of the track
        
    Returns:
        Payment intent details with payment ID
    """
    try:
        # Create payment intent
        intent = payment_service.create_payment_intent(
            amount=track_price,
            customer_id=DEFAULT_CUSTOMER_ID,
            description=f"Track: {track_name}",
            metadata={
                "track_id": track_id,
                "track_name": track_name,
                "purchase_type": "single_track"
            }
        )
        
        return f"""✅ Payment Intent Created!

💳 Payment ID: {intent.payment_intent_id}
💰 Amount: ${intent.amount} {intent.currency}
🎵 Track: {track_name} (ID: {track_id})
📊 Status: {intent.status}

Ready to process payment. Use payment ID: {intent.payment_intent_id}
"""
    except Exception as e:
        return f"❌ Error creating payment intent: {str(e)}"


@tool
def confirm_and_process_payment(payment_intent_id: str) -> str:
    """
    Step 2: Confirm and process the payment.
    This charges the customer (mock) and updates payment status.
    
    Args:
        payment_intent_id: The payment intent ID to process
        
    Returns:
        Payment confirmation or error message
    """
    try:
        # Process the payment
        intent = payment_service.confirm_payment(payment_intent_id)
        
        if intent.status == PaymentStatus.SUCCEEDED:
            return f"""✅ PAYMENT SUCCESSFUL!

💳 Payment ID: {intent.payment_intent_id}
💰 Amount Charged: ${intent.amount} {intent.currency}
✅ Status: {intent.status}
📅 Date: {intent.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

Payment processed successfully! Now creating your invoice...
Use this payment ID to create the invoice: {intent.payment_intent_id}
"""
        else:
            return f"""❌ PAYMENT FAILED

💳 Payment ID: {intent.payment_intent_id}
❌ Status: {intent.status}
💰 Amount: ${intent.amount} {intent.currency}

Please try again or use a different payment method.
"""
    except Exception as e:
        return f"❌ Error processing payment: {str(e)}"


@tool
def create_invoice_from_payment(payment_intent_id: str) -> str:
    """
    Step 3: Create an invoice in the database after successful payment.
    This records the purchase in the customer's order history.
    
    Args:
        payment_intent_id: The successful payment intent ID
        
    Returns:
        Invoice creation confirmation with invoice details
    """
    try:
        # Get payment intent
        intent = payment_service.get_payment_intent(payment_intent_id)
        
        if not intent:
            return f"❌ Payment intent {payment_intent_id} not found."
        
        if intent.status != PaymentStatus.SUCCEEDED:
            return f"❌ Cannot create invoice. Payment status is {intent.status}, not succeeded."
        
        # Get customer billing info
        from sqlalchemy import text
        with db._engine.connect() as conn:
            customer_result = conn.execute(
                text(f"SELECT Address, City, State, Country, PostalCode FROM Customer WHERE CustomerId = {DEFAULT_CUSTOMER_ID}")
            )
            customer_row = customer_result.fetchone()
            
            if not customer_row:
                return "❌ Customer not found."
            
            # Get track info from payment metadata
            track_id = intent.metadata.get("track_id")
            if not track_id:
                return "❌ Track ID not found in payment metadata."
            
            # Get next invoice ID
            max_invoice_result = conn.execute(text("SELECT MAX(InvoiceId) FROM Invoice"))
            max_invoice_id = max_invoice_result.fetchone()[0] or 0
            new_invoice_id = max_invoice_id + 1
            
            # Create invoice
            invoice_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(text(f"""
                INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, 
                                    BillingState, BillingCountry, BillingPostalCode, Total)
                VALUES ({new_invoice_id}, {DEFAULT_CUSTOMER_ID}, '{invoice_date}', 
                        '{customer_row[0] or ''}', '{customer_row[1] or ''}', 
                        '{customer_row[2] or ''}', '{customer_row[3] or ''}', 
                        '{customer_row[4] or ''}', {intent.amount})
            """))
            
            # Get next invoice line ID
            max_line_result = conn.execute(text("SELECT MAX(InvoiceLineId) FROM InvoiceLine"))
            max_line_id = max_line_result.fetchone()[0] or 0
            new_line_id = max_line_id + 1
            
            # Create invoice line
            conn.execute(text(f"""
                INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
                VALUES ({new_line_id}, {new_invoice_id}, {track_id}, {intent.amount}, 1)
            """))
            
            conn.commit()
            
            # Get track details for confirmation
            track_result = conn.execute(text(f"""
                SELECT t.Name, ar.Name as Artist, a.Title as Album
                FROM Track t
                LEFT JOIN Album a ON t.AlbumId = a.AlbumId
                LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
                WHERE t.TrackId = {track_id}
            """))
            track_row = track_result.fetchone()
            
            track_name = track_row[0] if track_row else "Unknown"
            artist_name = track_row[1] if track_row else "Unknown"
            album_name = track_row[2] if track_row else "Unknown"
        
        return f"""✅ PURCHASE COMPLETE! 🎉

📄 Invoice #{new_invoice_id} Created
📅 Date: {invoice_date}
💰 Total: ${intent.amount} {intent.currency}

🎵 Track Purchased:
   • {track_name}
   • Artist: {artist_name}
   • Album: {album_name}

📧 A receipt has been saved to your account.
🎧 You can now listen to this track anytime!

Use 'show my purchase history' to view all your purchases.
"""
    except Exception as e:
        return f"❌ Error creating invoice: {str(e)}"


@tool
def cancel_payment(payment_intent_id: str) -> str:
    """
    Cancel a pending payment intent.
    Use this if customer changes their mind before payment is processed.
    
    Args:
        payment_intent_id: The payment intent ID to cancel
        
    Returns:
        Cancellation confirmation
    """
    try:
        intent = payment_service.cancel_payment(payment_intent_id)
        
        return f"""✅ Payment Cancelled

💳 Payment ID: {intent.payment_intent_id}
❌ Status: {intent.status}
💰 Amount: ${intent.amount} {intent.currency}

No charges were made. Feel free to browse more tracks!
"""
    except Exception as e:
        return f"❌ Error cancelling payment: {str(e)}"


@tool
def get_recent_purchases(limit: int = 5) -> str:
    """
    Get the customer's most recent purchases/invoices.
    
    Args:
        limit: Maximum number of recent purchases to show (default 5)
        
    Returns:
        List of recent purchases with details
    """
    try:
        result = db.run(
            f"""
            SELECT 
                i.InvoiceId,
                i.InvoiceDate,
                i.Total,
                GROUP_CONCAT(t.Name || ' by ' || ar.Name, ', ') as Tracks
            FROM Invoice i
            LEFT JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
            LEFT JOIN Track t ON il.TrackId = t.TrackId
            LEFT JOIN Album a ON t.AlbumId = a.AlbumId
            LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId
            WHERE i.CustomerId = {DEFAULT_CUSTOMER_ID}
            GROUP BY i.InvoiceId
            ORDER BY i.InvoiceDate DESC
            LIMIT {limit};
            """,
            include_columns=True
        )
        
        if not result or "InvoiceId" not in result:
            return "📦 No purchases found yet. Start shopping to see your purchase history!"
        
        return f"""📦 Your Recent Purchases:

{result}

Total purchases shown: {limit}
"""
    except Exception as e:
        return f"❌ Error fetching purchases: {str(e)}"


@tool
def check_if_already_purchased(track_id: int) -> str:
    """
    Check if a customer has already purchased a specific track.
    Useful to prevent duplicate purchases.
    
    Args:
        track_id: The track ID to check
        
    Returns:
        Whether the track has been purchased or not
    """
    try:
        from sqlalchemy import text
        with db._engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT COUNT(*) as PurchaseCount
                FROM Invoice i
                JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                WHERE i.CustomerId = {DEFAULT_CUSTOMER_ID}
                AND il.TrackId = {track_id}
            """))
            row = result.fetchone()
            count = row[0] if row else 0
        
        if count > 0:
            return f"✅ You already own this track! You purchased it {count} time(s)."
        else:
            return f"❌ You haven't purchased this track yet. Would you like to buy it?"
    except Exception as e:
        return f"❌ Error checking purchase status: {str(e)}"


# Export all tools
PAYMENT_TOOLS = [
    get_track_details_for_purchase,
    initiate_track_purchase,
    confirm_and_process_payment,
    create_invoice_from_payment,
    cancel_payment,
    get_recent_purchases,
    check_if_already_purchased,
]


if __name__ == "__main__":
    # Test the payment tools
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("Testing Payment Tools")
    print("=" * 60)
    
    # Test 1: Get track details
    print("\n1. Getting track details (Track ID 1)...")
    result = get_track_details_for_purchase.invoke({"track_id": 1})
    print(result)
    
    # Test 2: Check if already purchased
    print("\n2. Checking if track already purchased...")
    result = check_if_already_purchased.invoke({"track_id": 1})
    print(result)
    
    # Test 3: Create payment intent
    print("\n3. Creating payment intent...")
    result = initiate_track_purchase.invoke({
        "track_id": 1,
        "track_name": "Test Track",
        "track_price": 0.99
    })
    print(result)
    
    print("\n" + "=" * 60)
    print("Tool tests completed!")
    print("=" * 60)
