from typing import Dict, Tuple
from loguru import logger


class CreditService:
    """Service for managing Lapis Credits in the trial version."""
    
    def __init__(self, initial_credits: int = 30, credits_per_document: int = 5, low_credit_threshold: int = 10):
        """Initialize the credit service.
        
        Args:
            initial_credits: Starting number of credits for new users
            credits_per_document: Number of credits to deduct per document processed
            low_credit_threshold: Threshold to show low credit warning
        """
        self.initial_credits = initial_credits
        self.credits_per_document = credits_per_document
        self.low_credit_threshold = low_credit_threshold
        
        # In a real system, this would be a database
        # For demo purposes, store in memory
        self.user_credits = {}
    
    def get_user_credits(self, user_id: str) -> int:
        """Get the number of credits available for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            int: Number of credits available
        """
        # Initialize credits for new users
        if user_id not in self.user_credits:
            self.user_credits[user_id] = self.initial_credits
            logger.info(f"Initialized {self.initial_credits} credits for user {user_id}")
        
        return self.user_credits.get(user_id, 0)
    
    def deduct_credits(self, user_id: str) -> Tuple[bool, int]:
        """Deduct credits for processing a document.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple[bool, int]: (success, remaining_credits)
        """
        current_credits = self.get_user_credits(user_id)
        
        # Check if user has enough credits
        if current_credits < self.credits_per_document:
            logger.warning(f"User {user_id} has insufficient credits: {current_credits}")
            return False, current_credits
        
        # Deduct credits
        remaining_credits = current_credits - self.credits_per_document
        self.user_credits[user_id] = remaining_credits
        
        logger.info(f"Deducted {self.credits_per_document} credits for user {user_id}. Remaining: {remaining_credits}")
        return True, remaining_credits
    
    def is_low_credit(self, user_id: str) -> bool:
        """Check if user has low credits remaining.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if credits are below the low threshold
        """
        current_credits = self.get_user_credits(user_id)
        return current_credits <= self.low_credit_threshold
    
    def get_credit_status(self, user_id: str) -> Dict:
        """Get complete credit status for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with credit information
        """
        current_credits = self.get_user_credits(user_id)
        is_low = current_credits <= self.low_credit_threshold
        
        status = {
            "credits": current_credits,
            "credits_per_document": self.credits_per_document,
            "is_low": is_low,
            "message": "You're running low on credits — top up or subscribe." if is_low else ""
        }
        
        return status
