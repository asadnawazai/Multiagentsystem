from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class SimilarDocument(BaseModel):
    """Schema for a similar document retrieved through vector search"""
    id: str
    document_type: str
    fields: Dict[str, Any]
    risk_score: int
    match_score: float

class RAGProcessingResult(BaseModel):
    """Schema for the result of RAG processing"""
    input_fields: Dict[str, Any]
    document_type: str
    embedding_id: Optional[str] = None
    similar_documents: List[SimilarDocument] = []
    rag_context: str = ""
    error: Optional[str] = None
