import os
import json
from typing import Dict, List, Any, Optional
from loguru import logger
import openai
from openai import OpenAI

class EmbeddingAgent:
    """Agent responsible for generating embeddings from document fields.
    
    This agent uses OpenAI's text-embedding-3-small model to create vector
    representations of structured document data.
    """
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the Embedding Agent.
        
        Args:
            openai_api_key: API key for OpenAI (if None, will try to get from env vars)
        """
        # Initialize OpenAI client
        try:
            self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
                
            # Just validate the API key format, don't make a real API call yet
            if not (self.api_key.startswith("sk-") or self.api_key.startswith("sk-proj-")):
                raise ValueError("Invalid OpenAI API key format")
                
            self.client = OpenAI(api_key=self.api_key, timeout=3.0)  # Short timeout
            logger.info("Embedding Agent initialized successfully")
            self.model_name = "text-embedding-3-small"
            self.embedding_dim = 1536  # Dimensionality of the embedding model
        except Exception as e:
            logger.error(f"Error initializing Embedding Agent: {str(e)}")
            logger.warning("The Embedding Agent will run in limited mode without vector embedding functionality")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if the embedding functionality is available.
        
        Returns:
            bool: True if embedding can be performed, False otherwise
        """
        return self.client is not None
    
    async def create_text_from_fields(self, fields: Dict[str, Any]) -> str:
        """Create a formatted text string from extracted fields.
        
        Args:
            fields: Dictionary of extracted fields
            
        Returns:
            String representation of the fields
        """
        # Create a formatted string from the fields
        text_parts = []
        
        # Add all fields in a consistent format
        for key, value in fields.items():
            # Skip empty values
            if not value:
                continue
                
            # Format the field key (snake_case to Title Case)
            formatted_key = key.replace('_', ' ').title()
            text_parts.append(f"{formatted_key}: {value}")
        
        # Join all parts with commas
        return ", ".join(text_parts)
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector, or None if error
        """
        if not self.is_available():
            logger.error("Embedding Agent is not available - missing OpenAI API key")
            return None
            
        try:
            # Call OpenAI's embedding API
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            
            # Extract the embedding from the response
            embedding = response.data[0].embedding
            
            logger.info(f"Successfully generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    async def process_document(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document to generate its embedding.
        
        Args:
            fields: Dictionary of extracted fields
            
        Returns:
            Dictionary with the original fields and generated embedding
        """
        try:
            # Create text representation
            text = await self.create_text_from_fields(fields)
            
            # Generate embedding
            embedding = await self.generate_embedding(text)
            
            result = {
                "text_representation": text,
                "embedding": embedding,
                "embedding_model": self.model_name,
                "embedding_dim": len(embedding) if embedding else None,
                "fields": fields
            }
            
            logger.info(f"Successfully processed document and generated embedding")
            return result
            
        except Exception as e:
            logger.error(f"Error in embedding process: {str(e)}")
            return {
                "text_representation": None,
                "embedding": None,
                "embedding_model": self.model_name,
                "embedding_dim": None,
                "fields": fields,
                "error": str(e)
            }