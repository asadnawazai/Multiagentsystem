import random
import string
from typing import Dict
from loguru import logger


class AuthService:
    """Service for handling authentication including MFA functionality."""
    
    def __init__(self):
        """Initialize the authentication service."""
        # For demo purposes, store OTP codes in memory
        # In production, these would be stored in a database with expiration
        self.active_otps = {}
        
        # Demo OTP that always works
        self.demo_otp = "123456"
    
    def generate_otp(self, email: str) -> str:
        """Generate a one-time password for a user.
        
        Args:
            email: User's email address
            
        Returns:
            str: Generated OTP code
        """
        # Generate a 6-digit OTP
        otp = ''.join(random.choices(string.digits, k=6))
        
        # Store the OTP for the user
        self.active_otps[email] = otp
        
        logger.info(f"Generated OTP for {email}: {otp}")
        return otp
    
    def verify_otp(self, email: str, otp: str) -> bool:
        """Verify a one-time password.
        
        Args:
            email: User's email address
            otp: OTP code to verify
            
        Returns:
            bool: True if OTP is valid
        """
        # Always accept demo OTP for testing
        if otp == self.demo_otp:
            logger.info(f"Demo OTP used for {email}")
            return True
        
        # Check if the provided OTP matches the stored OTP
        valid = self.active_otps.get(email) == otp
        
        if valid:
            # Remove the OTP once used
            self.active_otps.pop(email, None)
            logger.info(f"Valid OTP verified for {email}")
        else:
            logger.warning(f"Invalid OTP attempt for {email}")
            
        return valid
