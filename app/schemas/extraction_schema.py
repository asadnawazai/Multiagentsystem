from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field


class ExtractedFields(BaseModel):
    """Schema for fields extracted from a document."""
    name: Optional[str] = Field(None, description="Person or organization name")
    policy_number: Optional[str] = Field(None, description="Insurance policy number")
    claim_number: Optional[str] = Field(None, description="Claim reference number")
    date: Optional[str] = Field(None, description="Relevant date in ISO format (YYYY-MM-DD)")
    amount: Optional[str] = Field(None, description="Monetary amount")
    document_type: Optional[str] = Field(None, description="Type of document identified")


class ConfidenceScores(BaseModel):
    """Confidence scores for the extracted fields."""
    name: Optional[float] = Field(None, description="Confidence score for name extraction")
    policy_number: Optional[float] = Field(None, description="Confidence score for policy number extraction")
    claim_number: Optional[float] = Field(None, description="Confidence score for claim number extraction")
    date: Optional[float] = Field(None, description="Confidence score for date extraction")
    amount: Optional[float] = Field(None, description="Confidence score for amount extraction")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    extraction_method: str = Field(..., description="Method used for text extraction (pdf_text, ocr, etc.)")
    page_count: int = Field(..., description="Number of pages in the document")
    confidence: Optional[float] = Field(None, description="Overall OCR confidence score if applicable")
    original_file: str = Field(..., description="Original filename")


class TextExtractionResult(BaseModel):
    """Schema for text extraction results."""
    extracted_text: str = Field(..., description="Extracted and normalized text from the document")
    metadata: ExtractionMetadata = Field(..., description="Metadata about the extraction process")


class FieldExtractionResult(BaseModel):
    """Schema for field extraction results."""
    fields: ExtractedFields = Field(..., description="Extracted structured fields")
    confidence_scores: ConfidenceScores = Field(..., description="Confidence scores for field extraction")


class DocumentProcessingResult(BaseModel):
    """Schema for complete document processing results."""
    extracted_text: str = Field(..., description="Extracted and normalized text from the document")
    fields: ExtractedFields = Field(..., description="Extracted structured fields")
    confidence_scores: ConfidenceScores = Field(..., description="Confidence scores for field extraction")
    metadata: ExtractionMetadata = Field(..., description="Metadata about the extraction process")
