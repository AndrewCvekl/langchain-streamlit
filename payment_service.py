"""
Mock Payment Service API for testing payment flows.
Simulates a real payment processor like Stripe/Square with fake API calls.
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum


class PaymentStatus(str, Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PaymentMethod(str, Enum):
    """Payment method types"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    MOCK = "mock"


@dataclass
class PaymentIntent:
    """Represents a payment intent (similar to Stripe's PaymentIntent)"""
    payment_intent_id: str
    amount: float
    currency: str
    status: PaymentStatus
    customer_id: int
    description: str
    payment_method: PaymentMethod
    created_at: datetime
    updated_at: datetime
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['status'] = self.status.value
        data['payment_method'] = self.payment_method.value
        return data


class MockPaymentService:
    """
    Mock payment service that simulates a real payment processor.
    Provides realistic payment flows without actual transactions.
    """
    
    def __init__(self):
        """Initialize payment service with in-memory storage"""
        self.payment_intents: Dict[str, PaymentIntent] = {}
        self.default_currency = "USD"
    
    def create_payment_intent(
        self,
        amount: float,
        customer_id: int,
        description: str,
        metadata: Optional[Dict] = None
    ) -> PaymentIntent:
        """
        Create a new payment intent (step 1 of payment flow).
        
        Args:
            amount: Amount to charge (in dollars)
            customer_id: Customer ID making the purchase
            description: Description of what's being purchased
            metadata: Optional metadata (track_id, track_name, etc.)
            
        Returns:
            PaymentIntent object
        """
        # Generate unique payment intent ID
        payment_intent_id = f"pi_mock_{uuid.uuid4().hex[:16]}"
        
        # Create payment intent
        intent = PaymentIntent(
            payment_intent_id=payment_intent_id,
            amount=round(amount, 2),
            currency=self.default_currency,
            status=PaymentStatus.PENDING,
            customer_id=customer_id,
            description=description,
            payment_method=PaymentMethod.MOCK,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store it
        self.payment_intents[payment_intent_id] = intent
        
        return intent
    
    def confirm_payment(
        self,
        payment_intent_id: str,
        simulate_failure: bool = False
    ) -> PaymentIntent:
        """
        Confirm and process a payment intent (step 2 of payment flow).
        
        Args:
            payment_intent_id: ID of the payment intent to process
            simulate_failure: If True, simulates a failed payment (for testing)
            
        Returns:
            Updated PaymentIntent object
            
        Raises:
            ValueError: If payment intent not found or already processed
        """
        # Check if payment intent exists
        if payment_intent_id not in self.payment_intents:
            raise ValueError(f"Payment intent {payment_intent_id} not found")
        
        intent = self.payment_intents[payment_intent_id]
        
        # Check if already processed
        if intent.status in [PaymentStatus.SUCCEEDED, PaymentStatus.FAILED]:
            raise ValueError(f"Payment intent {payment_intent_id} already processed")
        
        # Update to processing
        intent.status = PaymentStatus.PROCESSING
        intent.updated_at = datetime.now()
        
        # Simulate processing delay
        time.sleep(0.5)
        
        # Process payment (simulate success/failure)
        if simulate_failure:
            intent.status = PaymentStatus.FAILED
        else:
            intent.status = PaymentStatus.SUCCEEDED
        
        intent.updated_at = datetime.now()
        
        return intent
    
    def cancel_payment(self, payment_intent_id: str) -> PaymentIntent:
        """
        Cancel a pending payment intent.
        
        Args:
            payment_intent_id: ID of the payment intent to cancel
            
        Returns:
            Updated PaymentIntent object
        """
        if payment_intent_id not in self.payment_intents:
            raise ValueError(f"Payment intent {payment_intent_id} not found")
        
        intent = self.payment_intents[payment_intent_id]
        
        if intent.status == PaymentStatus.SUCCEEDED:
            raise ValueError("Cannot cancel a succeeded payment")
        
        intent.status = PaymentStatus.CANCELLED
        intent.updated_at = datetime.now()
        
        return intent
    
    def get_payment_intent(self, payment_intent_id: str) -> Optional[PaymentIntent]:
        """
        Retrieve a payment intent by ID.
        
        Args:
            payment_intent_id: ID of the payment intent
            
        Returns:
            PaymentIntent if found, None otherwise
        """
        return self.payment_intents.get(payment_intent_id)
    
    def get_customer_payments(self, customer_id: int) -> list[PaymentIntent]:
        """
        Get all payment intents for a customer.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of PaymentIntent objects
        """
        return [
            intent for intent in self.payment_intents.values()
            if intent.customer_id == customer_id
        ]


# Singleton instance
_payment_service: Optional[MockPaymentService] = None


def get_payment_service() -> MockPaymentService:
    """
    Get or create payment service singleton.
    
    Returns:
        MockPaymentService instance
    """
    global _payment_service
    
    if _payment_service is None:
        _payment_service = MockPaymentService()
    
    return _payment_service


if __name__ == "__main__":
    # Test the payment service
    service = get_payment_service()
    
    print("=" * 60)
    print("Testing Mock Payment Service")
    print("=" * 60)
    
    # Test 1: Create payment intent
    print("\n1. Creating payment intent...")
    intent = service.create_payment_intent(
        amount=1.99,
        customer_id=58,
        description="Track: Blackened",
        metadata={"track_id": 123, "track_name": "Blackened"}
    )
    print(f"✅ Created: {intent.payment_intent_id}")
    print(f"   Amount: ${intent.amount} {intent.currency}")
    print(f"   Status: {intent.status}")
    
    # Test 2: Confirm payment (success)
    print("\n2. Confirming payment...")
    confirmed = service.confirm_payment(intent.payment_intent_id)
    print(f"✅ Confirmed: {confirmed.payment_intent_id}")
    print(f"   Status: {confirmed.status}")
    
    # Test 3: Create and fail a payment
    print("\n3. Testing failed payment...")
    failed_intent = service.create_payment_intent(
        amount=0.99,
        customer_id=58,
        description="Test Track"
    )
    failed = service.confirm_payment(failed_intent.payment_intent_id, simulate_failure=True)
    print(f"❌ Failed: {failed.payment_intent_id}")
    print(f"   Status: {failed.status}")
    
    # Test 4: Get customer payments
    print("\n4. Getting customer payment history...")
    payments = service.get_customer_payments(58)
    print(f"✅ Found {len(payments)} payments for customer 58")
    for p in payments:
        print(f"   - {p.payment_intent_id}: ${p.amount} ({p.status})")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

