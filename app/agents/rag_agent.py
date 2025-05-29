import os
import json
import uuid
import sqlite3
import psycopg2
from psycopg2.extras import Json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import hashlib

class RAGAgent:
    """Agent responsible for vector storage and similarity search.
    
    This agent connects to a PostgreSQL database with pgvector extension
    to store document embeddings and retrieve similar documents.
    """
    
    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize RAG Agent with database connection
        
        Args:
            db_params: Optional database connection parameters
        """
        # Get database connection parameters from environment or passed params
        self.db_params = db_params or {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5432'),
            'database': os.getenv('PG_DATABASE', 'test_db'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'password')
        }
        
        # Set embedding table name
        schema = os.getenv('PG_SCHEMA', 'public')
        table = os.getenv('PG_TABLE', 'real_estate_documents')
        self.embedding_table = f"{schema}.{table}"
        
        # Track database connection status
        self.db_connection_error = None
        self.db_setup_completed = False
        self.pgvector_available = False
        
        # Create extension and table if they don't exist
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                # Check if pgvector extension is available
                try:
                    # First check if the vector extension is already created
                    cursor.execute("""
                        SELECT 1 FROM pg_extension WHERE extname = 'vector';
                    """)
                    extension_exists = cursor.fetchone() is not None
                    
                    if not extension_exists:
                        # Try to create the extension
                        logger.info("Vector extension not found, attempting to create it...")
                        cursor.execute("""
                            CREATE EXTENSION IF NOT EXISTS vector;
                        """)
                        conn.commit()
                        self.pgvector_available = True
                        logger.info("Successfully created pgvector extension")
                    else:
                        self.pgvector_available = True
                        logger.info("Vector extension already exists")
                except Exception as ext_error:
                    logger.error(f"Error creating vector extension: {str(ext_error)}")
                    logger.warning("The pgvector extension is not available - vector similarity search will be disabled")
                    self.pgvector_available = False
                
                # Only create the table if pgvector is available
                if self.pgvector_available:
                    try:
                        cursor.execute(f"""
                            CREATE TABLE IF NOT EXISTS {self.embedding_table} (
                                id TEXT PRIMARY KEY,
                                file_name TEXT,
                                file_checksum TEXT,
                                extracted_text TEXT,
                                document_type TEXT,
                                fields JSONB,
                                risk_score INTEGER,
                                risk_band TEXT,
                                mls_listing TEXT,
                                build_year TEXT,
                                land_use_code TEXT,
                                flood_risk_score TEXT,
                                zoning_record TEXT,
                                outdated_tax_delta TEXT,
                                infrastructure_opacity TEXT,
                                regional_data_variation TEXT,
                                climate_score TEXT,
                                embedding vector(1536),
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                        """)
                        conn.commit()
                        logger.info(f"Successfully created or confirmed table {self.embedding_table}")
                    except Exception as table_error:
                        logger.error(f"Error creating embedding table: {str(table_error)}")
                        logger.warning("Vector similarity search will be disabled due to table creation failure")
                        self.pgvector_available = False
            
            conn.close()
            
            self.db_setup_completed = True
            if self.pgvector_available:
                logger.info("RAG Agent initialized with full vector search capabilities")
            else:
                logger.info("RAG Agent initialized with limited capabilities (no vector search)")

        except Exception as e:
            self.db_connection_error = str(e)
            logger.error(f"Error connecting to database: {str(e)}")
            logger.warning("The RAG Agent will run in limited mode without vector database functionality")
    
    def _get_connection(self):
        """Get a database connection.
        
        Returns:
            psycopg2 connection object
        """
        try:
            # First try with PG_* parameters
            try:
                logger.info(f"Attempting to connect to database {self.db_params['database']} as user {self.db_params['user']}...")
                # Clean any quotes from the password
                password = self.db_params['password']
                if password and (password.startswith('"') or password.startswith("'")):
                    password = password.strip('"').strip("'")
                    
                conn = psycopg2.connect(
                    host=self.db_params['host'],
                    port=self.db_params['port'],
                    dbname=self.db_params['database'],
                    user=self.db_params['user'],
                    password=password,
                    connect_timeout=3  # Set a short timeout to fail fast
                )
                logger.info("Database connection successful with PG_* parameters")
                return conn
            except Exception as primary_error:
                # If that fails, try with DB_* parameters as fallback
                logger.warning(f"Primary connection failed: {str(primary_error)}")
                logger.info("Trying fallback connection with DB_* parameters...")
                
                # Use the credentials from .env file with fallback values
                fallback_params = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': os.getenv('DB_PORT', '5432'),
                    'database': os.getenv('DB_NAME', 'postgres'),  # Changed from test_db to postgres
                    'user': os.getenv('DB_USER', 'postgres'),  # Changed from admin_user to postgres
                    'password': os.getenv('DB_PASSWORD', 'admin')  # Changed default to match .env
                }
                
                # Clean any quotes from the password
                password = fallback_params['password']
                if password and (password.startswith('"') or password.startswith("'")):
                    password = password.strip('"').strip("'")
                
                conn = psycopg2.connect(
                    host=fallback_params['host'],
                    port=fallback_params['port'],
                    dbname=fallback_params['database'],
                    user=fallback_params['user'],
                    password=password,
                    connect_timeout=3
                )
                logger.info("Database connection successful with DB_* parameters")
                return conn
        except Exception as e:
            self.db_connection_error = str(e)
            logger.error(f"All database connection attempts failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if the database connection is available.
        
        Returns:
            bool: True if connection can be established, False otherwise
        """
        # If we've already detected a connection error during initialization, return False
        if not self.db_setup_completed:
            return False
            
        # If pgvector extension is not available, we can't do vector operations
        if not self.pgvector_available:
            self.db_connection_error = "PostgreSQL connection is available, but pgvector extension is not installed or not working"
            return False
            
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                # Check for basic database connectivity
                cursor.execute("SELECT 1")
                cursor.fetchone()
                
                # Verify pgvector is working by trying a simple vector operation
                try:
                    cursor.execute("SELECT '[1,2,3]'::vector")
                    cursor.fetchone()
                except Exception as vector_error:
                    self.db_connection_error = f"PostgreSQL is connected, but pgvector extension is not working: {str(vector_error)}"
                    self.pgvector_available = False
                    logger.error(f"Vector operations not available: {str(vector_error)}")
                    conn.close()
                    return False
                    
            conn.close()  # Close the connection when done
            return True
        except Exception as e:
            self.db_connection_error = str(e)
            logger.error(f"Database connection check failed: {str(e)}")
            return False
            
    def get_connection_error(self) -> Optional[str]:
        """Get the database connection error message if any.
        
        Returns:
            Optional[str]: Error message if connection failed, None otherwise
        """
        return self.db_connection_error
    
    async def store_embedding(self, 
                        document_type: str, 
                        fields: Dict[str, Any], 
                        embedding: List[float],
                        risk_score: int = 0,
                        file_name: Optional[str] = None) -> Optional[str]:
        """Store an embedding in the database.
        
        Args:
            document_type: Type of document
            fields: Dictionary of extracted fields
            embedding: Vector embedding
            risk_score: Initial risk score
            file_name: Name of the uploaded file
            
        Returns:
            UUID of inserted record, or None if error
        """
        # Check if database is available first
        if not self.is_available():
            logger.warning("Embedding storage skipped - database not available")
            return None
            
        conn = None
        try:
            # Generate UUID for the record
            record_id = str(uuid.uuid4())
            
            # Get connection
            conn = self._get_connection()
            
            with conn.cursor() as cursor:
                # Insert the embedding
                # Need to properly format the vector for PostgreSQL
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                # Extract original_filename from fields if available and file_name not provided
                if file_name is None and 'original_filename' in fields:
                    file_name = fields['original_filename']
                elif file_name is None:
                    # Use a placeholder filename if none is provided
                    file_name = f"document_{record_id}.pdf"
                
                # Calculate risk band based on score
                risk_band = 'low'
                if risk_score >= 70:
                    risk_band = 'high'
                elif risk_score >= 40:
                    risk_band = 'moderate'
                
                # Extract all fields from the extracted data
                # Prioritize 'extracted_text' which is shown in the UI, then fall back to 'text'
                extracted_text = fields.get('extracted_text', fields.get('text', None))  # Full document text
                
                # Real estate specific fields - convert 'Not Found' back to None for database storage
                mls_listing = fields.get('mls_listing', None)
                if mls_listing == 'Not Found':
                    mls_listing = None
                
                # Convert build_year to int if possible
                build_year = fields.get('build_year', None)
                if build_year == 'Not Found':
                    build_year = None
                elif build_year and isinstance(build_year, str) and build_year.isdigit():
                    build_year = int(build_year)
                
                land_use_code = fields.get('land_use_code', None)
                if land_use_code == 'Not Found':
                    land_use_code = None
                    
                flood_risk_score = fields.get('flood_risk_score', None)
                if flood_risk_score == 'Not Found':
                    flood_risk_score = None
                    
                zoning_record = fields.get('zoning_record', None)
                if zoning_record == 'Not Found':
                    zoning_record = None
                    
                outdated_tax_delta = fields.get('outdated_tax_delta', None)
                if outdated_tax_delta == 'Not Found':
                    outdated_tax_delta = None
                    
                infrastructure_opacity = fields.get('infrastructure_opacity', None)
                if infrastructure_opacity == 'Not Found':
                    infrastructure_opacity = None
                    
                regional_data_variation = fields.get('regional_data_variation', None)
                if regional_data_variation == 'Not Found':
                    regional_data_variation = None
                    
                climate_score = fields.get('climate_score', None)
                if climate_score == 'Not Found':
                    climate_score = None
                
                # Generate a file checksum if one doesn't exist
                file_checksum = fields.get('file_checksum', hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest())
            
                # Insert using the new simplified database schema
                cursor.execute(f"""
                    INSERT INTO {self.embedding_table} (
                        id, file_name, file_checksum, extracted_text,
                        risk_score, risk_band, embedding, created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, 
                        %s, %s, %s::vector, NOW()
                    )
                """, (
                    record_id,
                    file_name,
                    file_checksum,
                    extracted_text,
                    risk_score,
                    risk_band,
                    embedding_str  # Format as a proper vector for PostgreSQL
                ))
                
                # Commit the transaction
                conn.commit()
            
            logger.info(f"Successfully stored embedding with ID: {record_id}")
            conn.close()  # Close the connection when done
            return record_id
                
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            self.db_connection_error = str(e)  # Update the error message
            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except Exception:
                    pass  # Ignore errors during cleanup
            return None
    
    async def find_similar_documents(self, 
                              embedding: List[float], 
                              limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar documents by vector similarity.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with similarity scores
        """
        # Check if database is available first
        if not self.is_available():
            logger.warning("Vector similarity search skipped - database not available")
            return []
            
        try:
            # Get database connection
            conn = self._get_connection()
            
            with conn.cursor() as cursor:
                # Convert embedding to PostgreSQL vector format and find similar items
                # Use cosine similarity (1 - (A <=> B))
                # Need to properly format the vector for PostgreSQL
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                query = f"""
                    SELECT id, file_name, extracted_text, risk_score, risk_band, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM {self.embedding_table}
                    ORDER BY similarity DESC
                    LIMIT %s;
                """
                
                cursor.execute(query, (embedding_str, limit))
                results = cursor.fetchall()
                
                # Process results
                similar_docs = []
                for row in results:
                    # Extract structured fields from the extracted_text (if possible)
                    # In our simplified schema, all fields are contained within the extracted_text
                    extracted_text = row[2]
                    
                    # Create a basic fields dictionary with the extracted text
                    fields = {
                        "text": extracted_text
                    }
                    
                    # Try to parse the extracted_text to see if it contains our structured format
                    # with the 9 required fields at the top
                    if "===== EXTRACTED REAL ESTATE FIELDS =====" in extracted_text:
                        # Parse the structured fields section
                        try:
                            fields_section = extracted_text.split("===== EXTRACTED REAL ESTATE FIELDS =====")[1]
                            fields_section = fields_section.split("===== FULL DOCUMENT TEXT =====")[0]
                            
                            # Extract each field
                            for line in fields_section.strip().split("\n"):
                                if ": " in line:
                                    field_name, field_value = line.split(": ", 1)
                                    # Convert display name back to field name (lowercase with underscores)
                                    field_key = field_name.lower().replace(" ", "_")
                                    fields[field_key] = field_value
                        except Exception as e:
                            logger.warning(f"Error parsing structured fields: {e}")
                    
                    # Determine document type based on the content
                    document_type = "Real Estate Document"  # Default
                    if fields.get("mls_listing") and fields.get("mls_listing") != "Not Found":
                        document_type = "MLS Listing"
                    
                    similar_docs.append({
                        "id": row[0],
                        "file_name": row[1],
                        "document_type": document_type,
                        "fields": fields,
                        "risk_score": row[3],
                        "risk_band": row[4],
                        "similarity": row[5]  # Similarity is now at index 5 (after removing extra columns)
                    })
                
                logger.info(f"Found {len(similar_docs)} similar documents")
                conn.close()  # Close the connection when done
                return similar_docs
                
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            self.db_connection_error = str(e)  # Update the error message
            return []
    
    async def generate_rag_context(self, similar_docs: List[Dict[str, Any]]) -> str:
        """Generate a natural language context from similar documents.
        
        Args:
            similar_docs: List of similar documents from vector search
            
        Returns:
            String with human-readable context about similar documents
        """
        if not similar_docs:
            return "No similar documents found in our database."
        
        try:
            # Extract key information from similar documents
            doc_types = set(doc.get("document_type", "Unknown") for doc in similar_docs)
            avg_risk = sum(doc.get("risk_score", 0) for doc in similar_docs) / len(similar_docs) if similar_docs else 0
            risk_level = "high" if avg_risk > 70 else "moderate" if avg_risk > 40 else "low"
            
            # Generate context text
            context = f"This case is similar to {len(similar_docs)} previous {', '.join(doc_types)} documents. "
            context += f"The average risk for similar cases was {avg_risk:.1f} (considered {risk_level} risk). "
            
            # Add specific details from the most similar document
            if similar_docs:
                top_match = similar_docs[0]
                context += f"The closest match had the following attributes: "
                for key, value in top_match.get("fields", {}).items():
                    if key not in ["id", "document_type", "risk_score", "match_score"]:
                        context += f"{key.replace('_', ' ').title()}: {value}, "
            
            return context.rstrip(", ")
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {str(e)}")
            return "Error generating context from similar documents."
    
    async def process_document(self, 
                        document_type: str,
                        fields: Dict[str, Any], 
                        embedding: List[float],
                        risk_score: int = 0) -> Dict[str, Any]:
        """Process a document through the RAG pipeline.
        
        Args:
            document_type: Type of document
            fields: Dictionary of extracted fields
            embedding: Vector embedding
            risk_score: Initial risk score
            
        Returns:
            Dictionary with input fields, similar documents, and RAG context
        """
        # First check if database is available
        if not self.is_available():
            # Generate a record ID anyway so we can track the document
            mock_record_id = str(uuid.uuid4())
            
            error_msg = self.get_connection_error() or "Database connection not available"
            logger.warning(f"Skipping RAG processing due to database unavailability: {error_msg}")
            
            # Create a simulated response with helpful information
            return {
                "input_fields": fields,
                "document_type": document_type,
                "embedding_id": mock_record_id,  # Provide a mock ID for tracking
                "similar_documents": [],
                "rag_context": "The document has been processed successfully, but vector similarity search is not available because the PostgreSQL database could not be accessed. Your document and its embedding were generated but not stored in the database.",
                "error": f"Database connection issue: {error_msg}",
                "db_connection_message": """To enable vector similarity search, please check:
1. PostgreSQL is installed and running
2. The database service is started
3. Connection parameters in .env match your PostgreSQL setup (especially host, port, user, password)
4. The pgvector extension is installed in your database""",
                "db_connection_status": "unavailable"
            }
            
        try:
            # Store the embedding
            # Get original filename from fields if available
            file_name = fields.get('original_filename', None)
            
            # Use the calculated risk_score from fields if available (dynamically calculated)
            calculated_risk_score = fields.get('risk_score', risk_score)
            
            # Pass the calculated risk_score to store_embedding
            record_id = await self.store_embedding(document_type, fields, embedding, calculated_risk_score, file_name)
            
            # Find similar documents
            similar_docs = await self.find_similar_documents(embedding)
            
            # Generate RAG context
            rag_context = await self.generate_rag_context(similar_docs)
            
            # Prepare result
            result = {
                "input_fields": fields,
                "document_type": document_type,
                "embedding_id": record_id,
                "similar_documents": similar_docs,
                "rag_context": rag_context
            }
            
            logger.info(f"Successfully processed document through RAG pipeline")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG process: {str(e)}")
            return {
                "input_fields": fields,
                "document_type": document_type,
                "embedding_id": None,
                "similar_documents": [],
                "rag_context": f"Error processing document: {str(e)}",
                "error": str(e),
                "db_connection_message": "Database error occurred during vector processing. Check database configuration and logs for details."
            }
