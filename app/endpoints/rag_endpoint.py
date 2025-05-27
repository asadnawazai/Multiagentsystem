import os
import json
from typing import Optional, Dict, Any, List
from fastapi import Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger

# Import application components
from ..agents.document_ingest_agent import DocumentIngestAgent
from ..agents.metadata_hashing_agent import MetadataHashingAgent
from ..agents.ocr_normalization_agent import OCRNormalizationAgent
from ..agents.nlu_extraction_agent import NLUExtractionAgent
from ..agents.embedding_agent import EmbeddingAgent
from ..agents.rag_agent import RAGAgent
from ..services.risk_scoring_service import RiskScoringService
from ..services.field_validation_service import FieldValidationService
from ..services.auth_service import AuthService
from ..services.credit_service import CreditService

class RAGEndpoint:
    """RAG endpoint for document processing with vector similarity search."""
    
    def __init__(
        self,
        document_ingest_agent: DocumentIngestAgent,
        metadata_hashing_agent: MetadataHashingAgent,
        ocr_normalization_agent: OCRNormalizationAgent,
        nlu_extraction_agent: NLUExtractionAgent,
        embedding_agent: EmbeddingAgent,
        rag_agent: RAGAgent,
        risk_scoring_service: RiskScoringService,
        field_validation_service: FieldValidationService,
        auth_service: AuthService,
        credit_service: CreditService
    ):
        """Initialize the RAG endpoint."""
        self.document_ingest_agent = document_ingest_agent
        self.metadata_hashing_agent = metadata_hashing_agent
        self.ocr_normalization_agent = ocr_normalization_agent
        self.nlu_extraction_agent = nlu_extraction_agent
        self.embedding_agent = embedding_agent
        self.rag_agent = rag_agent
        self.risk_scoring_service = risk_scoring_service
        self.field_validation_service = field_validation_service
        self.auth_service = auth_service
        self.credit_service = credit_service
    
    async def process_document(
        self,
        request: Request,
        file: UploadFile,
        client_id: Optional[str] = None,
        document_type: str = "Real Estate",
        risk_score: Optional[int] = None,
        redact_pii: Optional[bool] = False,
        user_email: Optional[str] = None,
        mfa_code: Optional[str] = None,
        api_key: Optional[str] = None
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
        logger.info(f"Processing document {file.filename} with Real Estate RAG")
        
        # Use client_id for credit tracking if provided, otherwise use a default
        user_id = client_id or 'default_user'
        
        try:
            # Check if MFA verification is required and not already completed
            if not mfa_code and request.headers.get('accept', '').startswith('text/html'):
                # Return the HTML form with MFA modal included
                with open("app/static/rag_result_template.html", "r") as f:
                    template = f.read()
                    
                with open("app/static/mfa_modal.html", "r") as f:
                    mfa_modal = f.read()
                    
                # Replace content with MFA form
                mfa_form = f"""
                <h3>Document Upload Requires Verification</h3>
                <p>For security purposes, please verify your identity before uploading sensitive real estate documents.</p>
                <form id='upload-form' action='/rag-process' method='post' enctype='multipart/form-data'>
                    <input type='hidden' name='document_type' value='{document_type}'>
                    <input type='hidden' name='api_key' value='{api_key}'>
                    <div class='form-group'>
                        <label for='file'>Select a document to analyze:</label>
                        <input type='file' id='file' name='file' required>
                    </div>
                    <button type='submit' class='button primary'>Continue to Verification</button>
                </form>
                """
                    
                html_content = template.replace("{result_content}", mfa_form) + mfa_modal
                return HTMLResponse(content=html_content, status_code=200)
                
            # Verify MFA code if provided (for form submissions)
            if mfa_code and user_email:
                if not self.auth_service.verify_otp(user_email, mfa_code):
                    return HTMLResponse(
                        content=f"<html><body><h1>Error</h1><p>Invalid verification code. Please try again.</p><a href='/rag-process'>Back</a></body></html>",
                        status_code=400
                    )
            
            # Check and deduct Lapis credits
            sufficient_credits, remaining_credits = self.credit_service.deduct_credits(user_id)
            if not sufficient_credits:
                return HTMLResponse(
                    content=f"<html><body><h1>Insufficient Credits</h1><p>You need at least {self.credit_service.credits_per_document} Lapis credits to process a document. You currently have {remaining_credits} credits.</p><a href='/onboarding'>Upgrade Account</a></body></html>",
                    status_code=402  # Payment Required
                )
            
            # Step 1: Document Ingest - Save the uploaded file with validation
            file_path, original_filename, validation_info = await self.document_ingest_agent.ingest_document(
                file=file,
                document_type=document_type,
                client_id=client_id
            )
            
            # Store validation_info for later use
            rag_result = {'validation_info': validation_info}
            
            # Check if there's a filename warning but continue processing
            filename_warning = validation_info.get("filename_warning", None)
            warning_html = ""
            if filename_warning:
                # We'll show a warning but continue processing
                warning_html = f"<div style='background-color: #fff3cd; padding: 10px; margin-bottom: 15px; border-radius: 5px;'><strong>Warning:</strong> {filename_warning}</div>"
                logger.info(f"Processing document with filename warning: {filename_warning}")
                
            # Check other validation errors (empty files, duplicates, etc.)
            if not validation_info["is_valid"]:
                return HTMLResponse(
                    content=f"<html><body><h1>File Validation Error</h1><p>{validation_info['message']}</p><a href='/rag-process'>Try Again</a></body></html>",
                    status_code=400
                )
            
            # Step 2: Metadata Hashing - Generate checksum and extract metadata
            metadata = self.metadata_hashing_agent.process_document(file_path, original_filename, client_id)
            
            # Step 3: OCR Normalization - Extract and normalize text
            extracted_text, page_count, confidence = await self.ocr_normalization_agent.process_document(file_path)
            metadata.update({
                'extracted_text': extracted_text,
                'page_count': page_count,
                'ocr_confidence': confidence,
                'extraction_method': 'OCR'
            })
            
            # Step 4: NLU Extraction - Extract structured fields
            fields = await self.nlu_extraction_agent.process_document(extracted_text, document_type)
            
            # Check for missing critical fields
            fields_validation = self.field_validation_service.check_missing_fields(fields)
            
            # Step 5: Calculate risk score based on the document fields
            risk_result = self.risk_scoring_service.calculate_risk_score(fields)
            calculated_risk_score = risk_result["risk_score"]
            risk_band = risk_result["risk_band"]
            contributing_factors = risk_result["contributing_factors"]
            
            # Use the calculated risk score instead of the default value
            fields['risk_score'] = calculated_risk_score
            fields['risk_band'] = risk_band
            fields['contributing_factors'] = contributing_factors
            
            # Step 6: Embedding Generation - Generate embeddings
            embedding = await self.embedding_agent.generate_embedding(extracted_text)
            
            # Step 7: RAG - Vector similarity search
            rag_result = await self.rag_agent.process_document(
                document_type=document_type,
                fields=fields,
                embedding=embedding,
                risk_score=calculated_risk_score
            )
            
            # Add metadata and input fields to the result
            rag_result['metadata'] = metadata
            rag_result['input_fields'] = fields
            rag_result['risk_score'] = calculated_risk_score
            rag_result['risk_band'] = risk_band
            rag_result['contributing_factors'] = contributing_factors
            
            # Add credit information
            credit_status = self.credit_service.get_credit_status(user_id)
            rag_result['credits'] = credit_status
            
            # Add missing fields information
            rag_result['missing_fields'] = fields_validation
            
            # Format and return the response
            return self._format_response(request, rag_result, original_filename, warning_html)
        
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            return HTMLResponse(
                content=f"<html><body><h1>Error</h1><p>{str(e)}</p><a href='/rag-process'>Try Again</a></body></html>",
                status_code=400
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing document")
    
    def _format_response(self, request: Request, rag_result: Dict, original_filename: str, warning_html: str = ""):
        """Format the RAG processing result based on the request type.
        
        Args:
            request: The HTTP request
            rag_result: The RAG processing result
            original_filename: The original filename
            warning_html: Optional HTML warning message to display
            
        Returns:
            HTML or JSON response
        """
        # Determine if this is a browser request or API call
        accept_header = request.headers.get('accept', '')
        
        # If this is a browser request (form submission), return HTML
        if 'text/html' in accept_header:
            # Read the result template
            with open("app/static/rag_result_template.html", "r") as f:
                template = f.read()
            
            # Current date as upload date if not provided
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')

            # Set risk class based on risk band
            risk_class = "risk-high" if rag_result.get('risk_band') == "High" else \
                        "risk-moderate" if rag_result.get('risk_band') == "Moderate" else "risk-low"
            
            # Get amount from extracted fields if available
            amount = "Not specified"
            if rag_result.get('input_fields') and rag_result['input_fields'].get('amount'):
                amount = f"${rag_result['input_fields']['amount']}"
            
            # Format risk factors as a comma-separated list
            risk_factors = "None identified"
            if rag_result.get('contributing_factors') and len(rag_result['contributing_factors']) > 0:
                risk_factors = ", ".join(rag_result['contributing_factors'])
            
            # Generate recommendation text based on risk score and similar documents
            recommendation_text = ""
            if rag_result.get('similar_documents') and len(rag_result['similar_documents']) > 0:
                top_doc = rag_result['similar_documents'][0]
                similarity = top_doc.get('similarity', 0) * 100  # Convert to percentage
                recommendation_text = f"It is similar ({similarity:.0f}%) to a previously uploaded document, which had a risk score of {top_doc.get('risk_score', 'unknown')}."
            recommendation_text += " Please review the contributing factors for final decision."
            
            # Format similar documents as table rows
            similar_documents_html = ""
            if rag_result.get('similar_documents') and len(rag_result['similar_documents']) > 0:
                for i, doc in enumerate(rag_result['similar_documents'], 1):
                    similarity = doc.get('similarity', 0) * 100  # Convert to percentage
                    risk_score = doc.get('risk_score', 'N/A')
                    
                    # Format matching fields
                    match_fields_html = ""
                    if doc.get('fields'):
                        for field_name, field_value in doc['fields'].items():
                            if isinstance(field_value, (dict, list)) or field_name.startswith('_') or field_name in ['file_checksum']:
                                continue
                            match_fields_html += f"<span class='match-field'>`{field_name}`: {field_value}</span> "
                    
                    # Emoji for the match number
                    emoji = "1️⃣" if i == 1 else ("2️⃣" if i == 2 else ("3️⃣" if i == 3 else f"{i}"))
                    # Format similar document row
                    similar_documents_html += f"<tr>"
                    similar_documents_html += f"<td>{emoji}</td>"
                    similar_documents_html += f"<td><span class='similarity-score'>{similarity:.2f}%</span></td>"
                    similar_documents_html += f"<td>{risk_score}</td>"
                    similar_documents_html += f"<td>{match_fields_html}</td>"
                    similar_documents_html += f"</tr>"
            else:
                similar_documents_html = "<tr><td colspan='4'>No similar documents found. This appears to be the first document of this type.</td></tr>"
            
            # Get file metadata from the result
            file_size = "Unknown"
            mime_type = "Unknown"
            page_count = "Unknown"
            extraction_method = "Unknown"
            ocr_confidence = ""
            
            # Try to get metadata from different possible locations
            if 'metadata' in rag_result:
                # Get from metadata object
                metadata = rag_result.get('metadata', {})
                file_size = metadata.get('file_size_formatted', 'Unknown')
                mime_type = metadata.get('mime_type', 'Unknown')
                page_count = str(metadata.get('page_count', 'Unknown'))
                extraction_method = metadata.get('extraction_method', 'Unknown')
                if metadata.get('confidence') is not None:
                    ocr_confidence = f"{metadata.get('confidence'):.2f}"
            elif 'validation_info' in rag_result:
                # Try to get from validation_info
                validation_info = rag_result.get('validation_info', {})
                file_size = validation_info.get('file_size_formatted', 'Unknown')
                mime_type = validation_info.get('mime_type', 'Unknown')
                if 'file_size_bytes' in validation_info:
                    bytes_size = validation_info.get('file_size_bytes', 0)
                    if bytes_size > 0:
                        mb_size = bytes_size / (1024 * 1024)
                        if mb_size >= 1:
                            file_size = f"{mb_size:.2f} MB"
                        else:
                            kb_size = bytes_size / 1024
                            file_size = f"{kb_size:.2f} KB"
            
            # Replace template placeholders with actual values
            template = template.replace("{{document_type}}", rag_result.get('document_type', 'Unknown'))
            template = template.replace("{{upload_date}}", current_date)
            template = template.replace("{{file_name}}", original_filename)
            template = template.replace("{{file_size}}", file_size)
            template = template.replace("{{mime_type}}", mime_type)
            template = template.replace("{{page_count}}", page_count)
            template = template.replace("{{extraction_method}}", extraction_method)
            template = template.replace("{{ocr_confidence}}", ocr_confidence)
            template = template.replace("{{amount}}", amount)
            template = template.replace("{{risk_score}}", str(rag_result.get('risk_score', 'N/A')))
            template = template.replace("{{risk_class}}", risk_class)
            template = template.replace("{{risk_band}}", rag_result.get('risk_band', 'Unknown'))
            template = template.replace("{{risk_factors}}", risk_factors)
            template = template.replace("{{recommendation_text}}", recommendation_text)
            template = template.replace("{{similar_documents}}", similar_documents_html)
            template = template.replace("{{credits_remaining}}", str(rag_result.get('credits', {}).get('remaining', 'Unknown')))
            
            # Insert warning HTML if present
            if warning_html:
                template = template.replace("<main>", f"<main>{warning_html}")
            
            return HTMLResponse(content=template, status_code=200)
        
        # For API calls, return JSON
        return JSONResponse(content=rag_result, status_code=200)
