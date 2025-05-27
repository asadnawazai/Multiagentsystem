from pydantic import BaseModel, Field
from typing import Optional


class DocumentMetadata(BaseModel):
    """Schema for document metadata response."""
    filename: str = Field(..., description="Original filename of the uploaded document")
    upload_time: str = Field(..., description="Timestamp when the document was uploaded (ISO format)")
    file_size: str = Field(..., description="Human-readable file size (e.g., '201KB')")
    client_id: Optional[str] = Field(None, description="Optional client identifier")
    checksum: str = Field(..., description="SHA-256 checksum of the file content")
