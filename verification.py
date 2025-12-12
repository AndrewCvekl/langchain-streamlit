"""
Twilio SMS verification service for account security.
Implements secure verification codes with retry logic.
"""

import os
import random
import string
import re
from datetime import datetime, timedelta
from typing import Optional, Dict
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException


class VerificationService:
    """Handles SMS verification using Twilio API."""
    
    def __init__(self, verification_store: Optional[Dict] = None):
        """
        Initialize Twilio client from environment variables.
        
        Args:
            verification_store: Optional external store (e.g., st.session_state) for persistence
        """
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.verify_service_sid = os.getenv('TWILIO_VERIFY_SERVICE_SID')
        self.demo_phone = os.getenv('DEMO_USER_PHONE', '+19144342859')

        # In demos / takehomes, reviewers often won't have Twilio credentials.
        # Instead of crashing, fall back to a local "mock verification" mode.
        self.twilio_enabled = all([self.account_sid, self.auth_token, self.verify_service_sid])
        self.client = Client(self.account_sid, self.auth_token) if self.twilio_enabled else None
        
        # Use provided store or create new in-memory store
        # If using Streamlit, pass st.session_state dict to persist across reruns
        if verification_store is not None:
            self.verification_store = verification_store
        else:
            self.verification_store: Dict[str, Dict] = {}
    
    def generate_code(self, length: int = 6) -> str:
        """Generate a random numeric verification code."""
        return ''.join(random.choices(string.digits, k=length))
    
    def _format_phone_number(self, phone: str) -> str:
        """
        Format phone number to E.164 format required by Twilio.
        
        E.164 format: +[country code][area code][number]
        Example: +14155552671
        
        Args:
            phone: Phone number in various formats
            
        Returns:
            Properly formatted E.164 phone number
        """
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # If already starts with +, check if valid
        if phone.startswith('+'):
            # Already formatted, just ensure no spaces/dashes
            return '+' + digits
        
        # If 10 digits, assume US/Canada (+1)
        if len(digits) == 10:
            return f'+1{digits}'
        
        # If 11 digits and starts with 1, assume US/Canada
        if len(digits) == 11 and digits.startswith('1'):
            return f'+{digits}'
        
        # If already has country code but no +
        if len(digits) > 10:
            return f'+{digits}'
        
        # Default: add +1 for US
        return f'+1{digits}'
    
    def send_verification_code(self, phone_number: str, customer_id: int) -> Dict:
        """
        Send verification code via Twilio Verify API.
        
        Args:
            phone_number: Phone number to send code to (will be formatted to E.164)
            customer_id: Customer ID for tracking
            
        Returns:
            Dict with status and message
        """
        try:
            # Format phone number to E.164 format
            formatted_phone = self._format_phone_number(phone_number)

            # Demo fallback: if Twilio isn't configured, generate a local code.
            if not self.twilio_enabled:
                code = self.generate_code()
                store_key = f"customer_{customer_id}"
                self.verification_store[store_key] = {
                    'phone': formatted_phone,
                    'status': 'pending',
                    'attempts': 0,
                    'max_attempts': 3,
                    'created_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(minutes=10),
                    'demo_code': code,
                }
                masked_phone = formatted_phone[:-4] + '****' if len(formatted_phone) > 4 else '****'
                return {
                    'success': True,
                    'status': 'pending',
                    'message': f'🧪 Demo mode (no SMS sent). Code for {masked_phone}: {code}',
                    'sid': None,
                }

            # Twilio Verify API - handles code generation and validation
            verification = self.client.verify.v2.services(self.verify_service_sid).verifications.create(
                to=formatted_phone, channel='sms'
            )
            
            # Store verification attempt
            store_key = f"customer_{customer_id}"
            self.verification_store[store_key] = {
                'phone': formatted_phone,
                'status': 'pending',
                'attempts': 0,
                'max_attempts': 3,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=10)
            }
            
            # Mask phone number for display (show last 4 digits)
            masked_phone = formatted_phone[:-4] + '****' if len(formatted_phone) > 4 else '****'
            
            return {
                'success': True,
                'status': verification.status,
                'message': f'✅ Verification code sent to {masked_phone}',
                'sid': verification.sid
            }
            
        except TwilioRestException as e:
            # Handle specific Twilio errors
            error_msg = str(e)
            if e.code == 60200:
                error_msg = "Invalid phone number format. Please check the number and try again."
            elif e.code == 60203:
                error_msg = "Maximum verification attempts reached. Please try again later."
            elif e.code == 20003:
                error_msg = "Authentication failed. Please check Twilio credentials."
            elif e.code == 60202:
                error_msg = "Maximum verification sends reached. Please try again later."
            else:
                error_msg = f"Twilio error ({e.code}): {e.msg}"
            
            return {
                'success': False,
                'message': f'❌ Failed to send verification code: {error_msg}',
                'error_code': e.code if hasattr(e, 'code') else None
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'❌ Failed to send verification code: {str(e)}'
            }
    
    def verify_code(self, phone_number: str, code: str, customer_id: int) -> Dict:
        """
        Verify the code using Twilio Verify API.
        
        Args:
            phone_number: Phone number that received the code (will be formatted to E.164)
            code: The verification code entered by user
            customer_id: Customer ID for tracking
            
        Returns:
            Dict with verification result
        """
        store_key = f"customer_{customer_id}"
        
        # Check if verification exists
        if store_key not in self.verification_store:
            return {
                'success': False,
                'message': '❌ No verification request found. Please request a new code first.'
            }
        
        verification_data = self.verification_store[store_key]
        
        # Check if expired
        if datetime.now() > verification_data['expires_at']:
            del self.verification_store[store_key]
            return {
                'success': False,
                'message': '⏱️ Verification code expired. Please request a new code.'
            }
        
        # Check attempts
        verification_data['attempts'] += 1
        
        if verification_data['attempts'] > verification_data['max_attempts']:
            del self.verification_store[store_key]
            return {
                'success': False,
                'message': '🚫 Maximum verification attempts exceeded. Please request a new code.',
                'max_attempts_exceeded': True
            }
        
        try:
            # Clean and validate code
            clean_code = code.strip()
            
            # Format phone number to E.164 format
            formatted_phone = self._format_phone_number(phone_number)

            # Demo fallback: local verification
            if not self.twilio_enabled:
                expected = verification_data.get('demo_code')
                if expected and clean_code == expected:
                    verification_data['status'] = 'verified'
                    verification_data['verified_at'] = datetime.now()
                    return {
                        'success': True,
                        'message': '✅ Phone number verified successfully! You can now make account changes.',
                        'status': 'verified',
                    }
                attempts_remaining = verification_data['max_attempts'] - verification_data['attempts']
                return {
                    'success': False,
                    'message': f'❌ Invalid verification code. {attempts_remaining} attempt(s) remaining.',
                    'attempts_remaining': attempts_remaining,
                }

            # Verify code with Twilio
            verification_check = self.client.verify.v2.services(self.verify_service_sid).verification_checks.create(
                to=formatted_phone, code=clean_code
            )

            if verification_check.status == 'approved':
                # Success! Mark as verified
                verification_data['status'] = 'verified'
                verification_data['verified_at'] = datetime.now()
                
                return {
                    'success': True,
                    'message': '✅ Phone number verified successfully! You can now make account changes.',
                    'status': 'verified'
                }
            else:
                attempts_remaining = verification_data['max_attempts'] - verification_data['attempts']
                return {
                    'success': False,
                    'message': f'❌ Invalid verification code. {attempts_remaining} attempt(s) remaining.',
                    'attempts_remaining': attempts_remaining
                }
                
        except TwilioRestException as e:
            attempts_remaining = verification_data['max_attempts'] - verification_data['attempts']
            
            # Handle specific Twilio errors
            error_msg = str(e)
            if e.code == 60202:
                error_msg = "Maximum verification checks reached. Please request a new code."
            elif e.code == 60203:
                error_msg = "Verification code has expired. Please request a new code."
            elif e.code == 60200:
                error_msg = "Invalid phone number format."
            else:
                error_msg = f"Verification error: {e.msg}"
            
            return {
                'success': False,
                'message': f'❌ {error_msg} ({attempts_remaining} attempt(s) remaining)',
                'attempts_remaining': attempts_remaining,
                'error_code': e.code if hasattr(e, 'code') else None
            }
        except Exception as e:
            attempts_remaining = verification_data['max_attempts'] - verification_data['attempts']
            return {
                'success': False,
                'message': f'❌ Verification failed: {str(e)}. {attempts_remaining} attempt(s) remaining.',
                'attempts_remaining': attempts_remaining
            }
    
    def is_verified(self, customer_id: int) -> bool:
        """
        Check if customer is currently verified in this session.
        
        Args:
            customer_id: Customer ID to check
            
        Returns:
            True if verified and not expired
        """
        store_key = f"customer_{customer_id}"
        
        if store_key not in self.verification_store:
            return False
        
        verification_data = self.verification_store[store_key]
        
        # Check if verified and not expired
        if verification_data['status'] == 'verified':
            # Verification valid for 30 minutes
            if 'verified_at' in verification_data:
                if datetime.now() < verification_data['verified_at'] + timedelta(minutes=30):
                    return True
        
        return False
    
    def clear_verification(self, customer_id: int):
        """Clear verification status for customer."""
        store_key = f"customer_{customer_id}"
        if store_key in self.verification_store:
            del self.verification_store[store_key]


# Singleton instance
_verification_service: Optional[VerificationService] = None
_verification_store: Optional[Dict] = None


def get_verification_service(verification_store: Optional[Dict] = None) -> VerificationService:
    """
    Get or create verification service singleton.
    
    Args:
        verification_store: Optional persistent store (e.g., st.session_state['verification_store'])
                           Pass this on first call to make verification persist across app reruns
    
    Returns:
        VerificationService instance
    """
    global _verification_service, _verification_store
    
    # If store is provided and different from current, recreate service with new store
    if verification_store is not None and verification_store is not _verification_store:
        _verification_store = verification_store
        _verification_service = VerificationService(verification_store)
    elif _verification_service is None:
        # First time initialization
        _verification_service = VerificationService(verification_store)
        _verification_store = verification_store
    
    return _verification_service


if __name__ == "__main__":
    # Test the verification service
    from dotenv import load_dotenv
    load_dotenv()
    
    service = get_verification_service()
    print("✅ Verification service initialized")
    print(f"Demo phone: {service.demo_phone}")
    
    # Test sending code
    result = service.send_verification_code(service.demo_phone, 58)
    print(f"\nSend result: {result}")
