import os
import json
from datetime import datetime
from typing import Optional, Union, Dict, Any, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from dotenv import load_dotenv

# Import our application components
import yaml
import hashlib

def generate_milestone1_html(metadata):
    """Generate HTML for Milestone 1: Document Metadata"""
    html = "<dl class='result-data'>"
    html += f"<dt>File Name:</dt><dd>{metadata.get('original_filename', metadata.get('filename', 'Unknown'))}</dd>"
    html += f"<dt>File Size:</dt><dd>{metadata.get('filesize', '0')} bytes</dd>"
    html += f"<dt>MIME Type:</dt><dd>{metadata.get('content_type', 'Unknown')}</dd>"
    html += f"<dt>Page Count:</dt><dd>{metadata.get('page_count', 'Unknown')}</dd>"
    html += f"<dt>Extraction Method:</dt><dd>{metadata.get('extraction_method', 'Unknown')}</dd>"
    if metadata.get('confidence'):
        html += f"<dt>OCR Confidence:</dt><dd>{metadata['confidence']:.2f}</dd>"
    html += "</dl>"
    
    # Add a preview of the extracted text
    html += "<h4>Text Preview</h4>"
    html += f"<div class='extracted-text'><pre>{metadata.get('extracted_text', '')[:500]}...</pre></div>"
    
    return html

def generate_milestone2_html(fields_data):
    """Generate HTML for Milestone 2: Field Extraction"""
    html = "<dl class='result-data'>"
    for key, value in fields_data["fields"].items():
        key_formatted = key.replace('_', ' ').title()
        confidence = fields_data["confidence_scores"].get(key, 0)
        confidence_html = f"<span class='confidence'>({confidence:.2f})</span>" if confidence else ""
        html += f"<dt>{key_formatted}: {confidence_html}</dt><dd>{value}</dd>"
    html += "</dl>"
    
    return html

def generate_vector_search_html(vector_data):
    """Generate HTML for Milestone 3: Vector Similarity Search"""
    # Check if there was an error
    if "error" in vector_data:
        error_html = f'<div class="alert alert-warning">\n<p><strong>Status:</strong> {vector_data.get("error", "Vector search unavailable")}</p>\n'
        
        # Add RAG context if available (this provides a human-readable explanation)
        if "rag_context" in vector_data and vector_data["rag_context"]:
            error_html += f'<p>{vector_data["rag_context"]}</p>\n'
        
        # Check if there's additional connection error information
        if "connection_error" in vector_data:
            error_html += f'<details class="error-details">\n<summary>Technical Details</summary>\n<pre>{vector_data["connection_error"]}</pre>\n</details>\n'
        
        # Add help message if available
        if "db_connection_message" in vector_data:
            # Fix the syntax error by using a variable to store the replaced message
            message = vector_data["db_connection_message"].replace("\n", "<br>")
            error_html += f'<div class="help-message"><strong>Solution:</strong><br>{message}</div>\n'
            
        error_html += '</div>'
        return error_html
    
    html = f'''
    <div class="vector-data">
        <dl class="result-data">
            <dt>Document Type:</dt><dd>{vector_data.get("document_type", "Unknown")}</dd>
            <dt>Embedding ID:</dt><dd>{vector_data.get("embedding_id", "Not stored")}</dd>
        </dl>
    '''
    
    # Format similar documents if they exist
    if vector_data.get("similar_documents"):
        html += '<h4>Similar Documents</h4>'
        html += '''
        <table class="data-table">
            <thead>
                <tr>
                    <th>Document</th>
                    <th>Similarity</th>
                    <th>Risk Score</th>
                    <th>Fields</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for doc in vector_data["similar_documents"]:
            # Format fields as a list
            fields_html = "<ul class='field-list'>"
            for field, value in doc.get("fields", {}).items():
                fields_html += f"<li><strong>{field}:</strong> {value}</li>"
            fields_html += "</ul>"
            
            # Add row for this document
            html += f'''
            <tr>
                <td>{doc.get("document_type", "Unknown")}</td>
                <td>{doc.get("similarity", 0) * 100:.2f}%</td>
                <td>{doc.get("risk_score", 0)}</td>
                <td>{fields_html}</td>
            </tr>
            '''
            
        html += '''
            </tbody>
        </table>
        '''
    else:
        html += "<p>No similar documents found. This appears to be the first document of this type.</p>"
    
    html += "</div>"
    return html
from app.agents.document_ingest_agent import DocumentIngestAgent
from app.agents.metadata_hashing_agent import MetadataHashingAgent
from app.agents.ocr_normalization_agent import OCRNormalizationAgent
from app.services.risk_scoring_service import RiskScoringService
from app.services.field_validation_service import FieldValidationService
from app.services.auth_service import AuthService
from app.services.credit_service import CreditService
from app.agents.nlu_extraction_agent import NLUExtractionAgent
from app.agents.embedding_agent import EmbeddingAgent
from app.agents.rag_agent import RAGAgent
from app.schemas.document_schema import DocumentMetadata
from app.schemas.extraction_schema import TextExtractionResult, FieldExtractionResult, DocumentProcessingResult
from app.schemas.rag_schema import RAGProcessingResult, SimilarDocument
from app.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Configure logger
setup_logger(os.getenv("LOG_LEVEL", "INFO"))

# Load configuration from environment variables
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "pdf,jpg,jpeg,png,csv").split(",")
API_KEY_NAME = os.getenv("API_KEY_NAME", "X-API-Key")
API_KEY = os.getenv("API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="PanoramaScore API",
    description="AI-powered agent-based pipeline for document risk analysis",
    version="0.1.0",
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define configuration paths
REAL_ESTATE_CONFIG = os.path.join('app', 'config', 'real_estate.yaml')

# Initialize agents and services
document_ingest_agent = DocumentIngestAgent(
    upload_folder=UPLOAD_FOLDER,
    allowed_extensions=ALLOWED_EXTENSIONS,
    max_file_size_mb=MAX_FILE_SIZE_MB
)
metadata_hashing_agent = MetadataHashingAgent()
ocr_normalization_agent = OCRNormalizationAgent()
nlu_extraction_agent = NLUExtractionAgent()
embedding_agent = EmbeddingAgent()
rag_agent = RAGAgent(
    db_params={
        'host': os.getenv('PG_HOST', 'localhost'),
        'port': int(os.getenv('PG_PORT', '5432')),
        'database': os.getenv('PG_DATABASE', 'postgres'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.getenv('PG_PASSWORD', 'password')
    }
)

# Initialize additional services
risk_scoring_service = RiskScoringService(config_path=REAL_ESTATE_CONFIG)
field_validation_service = FieldValidationService(config_path=REAL_ESTATE_CONFIG)
auth_service = AuthService()
credit_service = CreditService(initial_credits=30, credits_per_document=5, low_credit_threshold=10)

# Initialize endpoints
from app.endpoints.rag_endpoint import RAGEndpoint
rag_endpoint = RAGEndpoint(
    document_ingest_agent=document_ingest_agent,
    metadata_hashing_agent=metadata_hashing_agent,
    ocr_normalization_agent=ocr_normalization_agent,
    nlu_extraction_agent=nlu_extraction_agent,
    embedding_agent=embedding_agent,
    rag_agent=rag_agent,
    risk_scoring_service=risk_scoring_service,
    field_validation_service=field_validation_service,
    auth_service=auth_service,
    credit_service=credit_service
)


# Simple API key security - use more robust auth in production
async def verify_api_key(x_api_key: str = Header(None), api_key: str = Form(None)):
    # Check header first, then form data
    effective_key = x_api_key if x_api_key else api_key
    
    if API_KEY and effective_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return effective_key


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the home page."""
    with open("app/static/index.html", "r") as f:
        return f.read()

@app.get("/process-form")
async def process_form():
    return FileResponse("app/static/process-form.html")

@app.get("/rag-process")
async def rag_process_form():
    return FileResponse("app/static/rag-process.html")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": app.version}


@app.post("/upload", response_model=None, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    """Upload a document and get metadata.
    
    This endpoint implements the first two agents in the PanoramaScore pipeline:
    1. Document Ingest Agent - Accepts and saves the file
    2. Metadata Hashing Agent - Generates checksum and extracts metadata
    """
    try:
        # Step 1: Document Ingest Agent processes the file
        file_path, original_filename, validation_info = await document_ingest_agent.ingest_document(
            file=file,
            document_type=document_type,
            client_id=client_id
        )
        
        # Check validation results
        if not validation_info["is_valid"]:
            accept_header = request.headers.get("accept", "")
            error_message = validation_info["message"]
            
            # For browser requests, return an HTML error page
            if "text/html" in accept_header:
                return HTMLResponse(
                    content=f"<html><body><h1>File Validation Error</h1><p>{error_message}</p><a href=\'/\'>Try Again</a></body></html>",
                    status_code=400
                )
            
            # For API requests, return a JSON error
            return JSONResponse(
                content={"error": error_message},
                status_code=400
            )
        
        # Check if this is a valid real estate document (when document_type is Real Estate)
        if document_type == "Real Estate" and not validation_info.get("is_real_estate_doc", True):
            accept_header = request.headers.get("accept", "")
            error_message = validation_info["message"]
            
            # For browser requests, return an HTML error page
            if "text/html" in accept_header:
                return HTMLResponse(
                    content=f"<html><body><h1>Invalid Document Type</h1><p>{error_message}</p><a href=\'/\'>Try Again</a></body></html>",
                    status_code=400
                )
            
            # For API requests, return a JSON error
            return JSONResponse(
                content={"error": error_message},
                status_code=400
            )
        
        # Step 2: Metadata Hashing Agent extracts metadata and generates hash
        metadata = metadata_hashing_agent.process_document(
            file_path=file_path,
            original_filename=original_filename,
            client_id=client_id
        )
        
        logger.info(f"Successfully processed document: {original_filename}")
        
        # Determine if this is a browser request or API call
        accept_header = request.headers.get('accept', '')
        
        # If this is a browser request (form submission), return HTML
        if 'text/html' in accept_header:
            # Read the result template
            with open("app/static/result_template.html", "r") as f:
                template = f.read()
            
            # Format the metadata as HTML
            result_html = "<dl class='result-data'>"
            for key, value in metadata.items():
                key_formatted = key.replace('_', ' ').title()
                result_html += f"<dt>{key_formatted}:</dt><dd>{value}</dd>"
            result_html += "</dl>"
            
            # Add raw JSON display
            result_html += f"<h3>Raw JSON:</h3><pre>{json.dumps(metadata, indent=2)}</pre>"
            
            # Replace the placeholder in the template
            html_content = template.replace("{result_content}", result_html)
            
            return HTMLResponse(content=html_content, status_code=201)
        
        # For API calls, return JSON
        return JSONResponse(content=metadata, status_code=201)
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing document")


@app.post("/process", response_model=None)
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
    document_type: str = Form("Real Estate", description="Type of document"),
    risk_score: int = Form(50, description="Initial risk score"),
    enable_vector_search: bool = Form(True, description="Enable vector similarity search"),
    redact_pii: Optional[bool] = Form(False),  # Make optional to handle form submission without checkbox
    api_key: str = Depends(verify_api_key)
):
    """Process a document to extract text and structured fields.
    
    This endpoint implements the OCR & NLU steps of the PanoramaScore pipeline:
    1. Document Ingest Agent - Accepts and saves the file
    2. Metadata Hashing Agent - Generates checksum and extracts metadata
    3. OCR & Normalization Agent - Extracts and cleans text
    4. NLU Extraction Agent - Identifies structured fields
    """
    try:
        # Step 1: Document Ingest Agent processes the file
        file_path, original_filename, validation_info = await document_ingest_agent.ingest_document(
            file=file,
            document_type=document_type,
            client_id=client_id
        )
        
        # Check validation results
        if not validation_info["is_valid"]:
            accept_header = request.headers.get("accept", "")
            error_message = validation_info["message"]
            
            # For browser requests, return an HTML error page
            if "text/html" in accept_header:
                return HTMLResponse(
                    content=f"<html><body><h1>File Validation Error</h1><p>{error_message}</p><a href=\'/\'>Try Again</a></body></html>",
                    status_code=400
                )
            
            # For API requests, return a JSON error
            return JSONResponse(
                content={"error": error_message},
                status_code=400
            )
        
        # Check if this is a valid real estate document (when document_type is Real Estate)
        if document_type == "Real Estate" and not validation_info.get("is_real_estate_doc", True):
            accept_header = request.headers.get("accept", "")
            error_message = validation_info["message"]
            
            # For browser requests, return an HTML error page
            if "text/html" in accept_header:
                return HTMLResponse(
                    content=f"<html><body><h1>Invalid Document Type</h1><p>{error_message}</p><a href=\'/\'>Try Again</a></body></html>",
                    status_code=400
                )
            
            # For API requests, return a JSON error
            return JSONResponse(
                content={"error": error_message},
                status_code=400
            )
        
        # Step 2: Metadata Hashing Agent extracts metadata and generates hash
        metadata = metadata_hashing_agent.process_document(
            file_path=file_path,
            original_filename=original_filename,
            client_id=client_id
        )
        
        # Step 3: OCR & Normalization Agent extracts text
        try:
            text_extraction = await ocr_normalization_agent.process_document(file_path)
            
            # Optionally redact PII
            if redact_pii:
                text_extraction["extracted_text"] = ocr_normalization_agent.redact_pii(
                    text_extraction["extracted_text"]
                )
            
            # Add file metadata to text extraction result
            text_extraction.update({
                "file_name": original_filename,
                "file_size_bytes": validation_info.get("file_size_bytes", 0),
                "file_size_formatted": validation_info.get("file_size_formatted", "Unknown"),
                "mime_type": validation_info.get("mime_type", "Unknown")
            })
        except Exception as e:
            logger.error(f"Error during OCR & text extraction: {str(e)}")
            text_extraction = {
                "extracted_text": f"Error extracting text: {str(e)}. This may be due to missing dependencies like Tesseract OCR or pdf2image.",
                "extraction_method": "error",
                "page_count": 0,
                "confidence": None,
                "original_file": original_filename,
                "file_name": original_filename,
                "file_size_bytes": validation_info.get("file_size_bytes", 0),
                "file_size_formatted": validation_info.get("file_size_formatted", "Unknown"),
                "mime_type": validation_info.get("mime_type", "Unknown")
            }
        
        # Step 4: NLU Extraction Agent identifies structured fields
        try:
            field_extraction = await nlu_extraction_agent.extract_fields(
                text_extraction["extracted_text"]
            )
        except Exception as e:
            logger.error(f"Error during NLU field extraction: {str(e)}")
            field_extraction = {
                "fields": {
                    "document_type": "Unknown Document",
                    "note": f"Error extracting fields: {str(e)}. This may be due to missing NLP dependencies."
                },
                "confidence_scores": {}
            }
            
        # Step 4.5: Calculate a proper risk score based on extracted fields
        try:
            # Use the risk scoring service to calculate a score based on fields
            extracted_fields = field_extraction["fields"]
            calculated_risk = risk_scoring_service.calculate_risk_score(extracted_fields)
            
            # Override the default risk score with the calculated one
            risk_score = calculated_risk["risk_score"]
            logger.info(f"Calculated risk score: {risk_score} based on extracted fields")
        except Exception as e:
            logger.warning(f"Error calculating risk score: {str(e)}. Using default score: {risk_score}")
            # Keep using the default risk score in case of error
        
        # Base result with extracted text and fields
        result = {
            "metadata": text_extraction,
            "fields": field_extraction,
            "validation_info": validation_info  # Include the validation info with file metadata
        }
        
        # Step 5 (Optional): If vector search is enabled, generate embeddings and find similar documents
        if enable_vector_search:
            try:
                # Only proceed if the embedding agent is available
                if not embedding_agent.is_available():
                    logger.warning("Embedding Agent is not available - OpenAI API key missing or invalid")
                    result["vector_search"] = {
                        "error": "Vector search disabled - OpenAI API key missing or invalid"
                    }
                else:
                    # Generate embedding from the extracted fields
                    embedding_result = await embedding_agent.process_document(field_extraction["fields"])
                    
                    if not embedding_result.get("embedding"):
                        logger.warning("Failed to generate embedding")
                        result["vector_search"] = {
                            "error": "Failed to generate embedding from extracted fields"
                        }
                    else:
                        # Only proceed if the RAG agent (database) is available
                        if not rag_agent.is_available():
                            error_msg = rag_agent.get_connection_error() or "Database connection failed"
                            logger.warning(f"RAG Agent is not available - {error_msg}")
                            result["vector_search"] = {
                                "error": "Vector search disabled - Database not connected",
                                "connection_error": error_msg,
                                "help_message": "To enable vector similarity search (Milestone 3), please ensure PostgreSQL is running and properly configured in .env"
                            }
                        else:
                            # Store the embedding and find similar documents
                            rag_result = await rag_agent.process_document(
                                document_type=document_type,
                                fields=field_extraction["fields"],
                                embedding=embedding_result["embedding"],
                                risk_score=risk_score
                            )
                            
                            # Add the RAG results to the complete result
                            result["vector_search"] = {
                                "document_type": document_type,
                                "embedding_id": rag_result.get("embedding_id"),
                                "similar_documents": rag_result.get("similar_documents", []),
                                "rag_context": rag_result.get("rag_context", "")
                            }
            except Exception as e:
                logger.error(f"Error during vector similarity search: {str(e)}")
                result["vector_search"] = {
                    "error": f"Error during vector similarity search: {str(e)}"
                }
        
        logger.info(f"Successfully processed document: {original_filename}")
        
        # Determine if this is a browser request or API call
        accept_header = request.headers.get('accept', '')
        
        # If this is a browser request (form submission), return HTML
        if 'text/html' in accept_header:
            # Generate HTML content for each milestone
            milestone1_content = generate_milestone1_html(result['metadata'])
            milestone2_content = generate_milestone2_html(result['fields'])
            
            # Generate vector search results content if available
            vector_search_section = "<p>Vector similarity search was not enabled for this document.</p>"
            if "vector_search" in result:
                vector_search_section = generate_vector_search_html(result["vector_search"])
            
            # Get the HTML template
            with open("./app/static/result_template.html", "r") as file:
                html_template = file.read()
            
            # Get risk score and band information
            risk_score = result.get('risk_score', 50)  # Default to 50 if not present
            
            # Determine risk band based on score
            risk_band = "Low Risk"
            risk_band_class = "low"
            if risk_score >= 70:
                risk_band = "High Risk"
                risk_band_class = "high"
            elif risk_score >= 40:
                risk_band = "Moderate Risk"
                risk_band_class = "moderate"
            
            # Get field values for risk analysis
            extracted_fields = result.get('fields', {})
            
            # Format field names to be more human-readable
            readable_fields = []
            important_fields = ['property_address', 'price', 'amount', 'tax_value', 'parcel_id', 'name', 'date']
            
            # First add important fields that exist
            for field in important_fields:
                if field in extracted_fields and extracted_fields[field]:
                    value = extracted_fields[field]
                    if field == 'amount' or field == 'price' or field == 'tax_value':
                        # Add dollar sign to monetary values if not present
                        if isinstance(value, str) and not value.startswith('$'):
                            value = f'${value}'
                    readable_fields.append(f"'{field}' ({value})")
            
            # Then add any other fields that weren't in the important list
            for k, v in extracted_fields.items():
                if k not in important_fields and k != 'text' and v:
                    readable_fields.append(f"'{k}'")
            
            # Join the field names with commas
            if readable_fields:
                field_names_text = ', '.join(readable_fields)
            else:
                field_names_text = "the document content and metadata"
            
            # Generate risk analysis text
            risk_analysis = f"Key factors influencing this assessment include {field_names_text}."
            
            # Get file metadata with better fallbacks
            # Try to get filename from multiple possible sources
            file_name = result.get('file_name', 
                       result.get('original_filename', 
                       result.get('filename', 
                       validation_info.get('original_filename', 'Unknown'))))
            
            # Try to get document type from extracted fields or fallback to result
            document_type = 'Unknown Document'
            if 'fields' in result and isinstance(result['fields'], dict):
                # Check if document_type is in the extracted fields
                if 'document_type' in result['fields']:
                    document_type = result['fields']['document_type']
                # If MLS is in fields, it's likely an MLS report
                elif any(field for field in ['mls', 'MLS'] if field in result['fields']):
                    document_type = 'MLS Report'
            # Fallback to the document_type in the result if not found in fields
            if document_type == 'Unknown Document':
                document_type = result.get('document_type', 'Real Estate Document')
            
            # Get upload date, file size, mime type, and page count
            upload_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get file size from validation info if available
            file_size = 'Unknown'
            if 'validation_info' in result and 'file_size_formatted' in result['validation_info']:
                file_size = result['validation_info']['file_size_formatted']
            else:
                file_size = result.get('file_size_formatted', 
                           result.get('filesize_formatted', 'Unknown'))
                
            # Get mime type from validation info if available
            mime_type = 'Unknown'
            if 'validation_info' in result and 'mime_type' in result['validation_info']:
                mime_type = result['validation_info']['mime_type']
            else:
                mime_type = result.get('mime_type', 
                            result.get('content_type', 'Unknown'))
                
            # Get page count
            page_count = result.get('page_count', '1')
            
            # Get credits information from our existing service
            credits_remaining = 25  # Default value if we can't get actual credits
            try:
                # Check if client_id exists and use the existing credit_service
                if client_id:
                    credits_remaining = credit_service.get_remaining_credits(client_id)
            except Exception as e:
                logger.error(f"Error getting credits: {str(e)}")
                # Fall back to default value
            
            # Replace the placeholders with content
            html_content = html_template.replace("{milestone1_content}", milestone1_content)
            html_content = html_content.replace("{milestone2_content}", milestone2_content)
            html_content = html_content.replace("{vector_search_section}", vector_search_section)
            
            # Replace risk score and band
            html_content = html_content.replace("{risk_score}", str(risk_score))
            html_content = html_content.replace("{risk_band}", risk_band)
            html_content = html_content.replace("{risk_band_class}", risk_band_class)
            html_content = html_content.replace("{risk_analysis}", risk_analysis)
            
            # Replace metadata
            html_content = html_content.replace("{file_name}", file_name)
            html_content = html_content.replace("{document_type}", document_type)
            html_content = html_content.replace("{upload_date}", upload_date)
            html_content = html_content.replace("{file_size}", file_size)
            html_content = html_content.replace("{mime_type}", mime_type)
            html_content = html_content.replace("{page_count}", str(page_count))
            
            # Replace credits
            html_content = html_content.replace("{credits_remaining}", str(credits_remaining))
            
            # Return the HTML response
            return HTMLResponse(content=html_content, status_code=200)
        
        # For API calls, return JSON
        return JSONResponse(content=result, status_code=200)
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing document")

@app.post("/rag-process", response_model=None)
async def rag_process_document(
    request: Request,
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
    document_type: str = Form("Real Estate"),
    risk_score: Optional[int] = Form(None),  # Make risk_score optional as we'll calculate it
    redact_pii: Optional[bool] = Form(False),
    api_key: str = Depends(verify_api_key),
    user_email: Optional[str] = Form(None),
    mfa_code: Optional[str] = Form(None)
):
    """Process a document with vector similarity search (RAG) and real estate risk scoring.
    
    This endpoint implements the enhanced PanoramaScore pipeline:
    1. MFA Verification - Validates user identity
    2. Credits Management - Tracks usage of Lapis credits
    3. Document Ingest & Validation - Accepts only valid real estate documents
    4. Field Extraction & Validation - Identifies missing critical fields
    5. Risk Scoring - Calculates risk based on the domain adapter
    6. RAG - Performs vector similarity search with clear source attribution
    """
    # Use the RAG endpoint to process the document
    return await rag_endpoint.process_document(
        request=request,
        file=file,
        client_id=client_id,
        document_type=document_type,
        risk_score=risk_score,
        redact_pii=redact_pii,
        user_email=user_email,
        mfa_code=mfa_code,
        api_key=api_key
    )
# Create uploads directory on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting PanoramaScore API")
    logger.info(f"Configuration: Upload folder={UPLOAD_FOLDER}, Max file size={MAX_FILE_SIZE_MB}MB")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# This function has been moved to the RAGEndpoint class for better modularity
# The _format_rag_response function is now implemented in app/endpoints/rag_endpoint.py


@app.get("/onboarding")
async def onboarding_flow():
    """Serve the onboarding flow page."""
    with open("app/static/onboarding_flow.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=200)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting PanoramaScore API")
    logger.info(f"Configuration: Upload folder={UPLOAD_FOLDER}, Max file size={MAX_FILE_SIZE_MB}MB")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
