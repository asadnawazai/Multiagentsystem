import os
import json
import uuid
import sqlite3
import psycopg2
from psycopg2.extras import Json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

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
                                document_type TEXT,
                                fields JSONB,
                                risk_score INTEGER,
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
                conn = psycopg2.connect(
                    host=self.db_params['host'],
                    port=self.db_params['port'],
                    dbname=self.db_params['database'],
                    user=self.db_params['user'],
                    password=self.db_params['password'],
                    connect_timeout=3  # Set a short timeout to fail fast
                )
                logger.info("Database connection successful with PG_* parameters")
                return conn
            except Exception as primary_error:
                # If that fails, try with DB_* parameters as fallback
                logger.warning(f"Primary connection failed: {str(primary_error)}")
                logger.info("Trying fallback connection with DB_* parameters...")
                
                fallback_params = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': os.getenv('DB_PORT', '5432'),
                    'database': os.getenv('DB_NAME', 'test_db'),
                    'user': os.getenv('DB_USER', 'admin_user'),
                    'password': os.getenv('DB_PASSWORD', 'dbuser123')
                }
                
                conn = psycopg2.connect(
                    host=fallback_params['host'],
                    port=fallback_params['port'],
                    dbname=fallback_params['database'],
                    user=fallback_params['user'],
                    password=fallback_params['password'],
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
                
                # Handle file_checksum - ensure it's never NULL
                file_checksum = fields.get('file_checksum')
                if file_checksum is None:
                    # Generate a default checksum based on the record ID if none is available
                    import hashlib
                    default_checksum = hashlib.sha256(record_id.encode()).hexdigest()
                    file_checksum = default_checksum
                    logger.warning(f"No checksum available for {file_name}, using generated default")
                
                # Determine risk band based on risk score
                risk_band = 'low'
                if risk_score >= 75:
                    risk_band = 'high'
                elif risk_score >= 50:
                    risk_band = 'moderate'
                
                # Identify contributing factors (basic implementation)
                contributing_factors = []
                for field_name, field_value in fields.items():
                    # Add critical fields that might contribute to risk
                    if field_name in ['flood_zone', 'property_condition', 'year_built']:
                        if field_value:
                            contributing_factors.append(f"{field_name}: {field_value}")
                
                if not contributing_factors:
                    contributing_factors = ["Standard document with no elevated risk factors"]
                
                cursor.execute(f"""
                    INSERT INTO {self.embedding_table} (id, file_name, file_checksum, document_type, fields, embedding, risk_score, risk_band, contributing_factors, status)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
                """, (
                    record_id,
                    file_name,
                    file_checksum,  # Now always has a value
                    document_type,
                    json.dumps(fields),
                    embedding_str,  # Format as a proper vector for PostgreSQL
                    risk_score,
                    risk_band,
                    json.dumps(contributing_factors),
                    'complete'  # Mark as complete immediately
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
                    SELECT id, document_type, fields, risk_score, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.embedding_table}
                    ORDER BY similarity DESC
                    LIMIT %s;
                """
                
                cursor.execute(query, (embedding_str, limit))
                results = cursor.fetchall()
                
                # Process results
                similar_docs = []
                for row in results:
                    # Parse fields if it's a string, otherwise use as is
                    fields = row[2]
                    if isinstance(fields, str):
                        try:
                            fields = json.loads(fields)
                        except json.JSONDecodeError:
                            fields = {"error": "Failed to parse fields"}
                
                    similar_docs.append({
                        "id": row[0],
                        "document_type": row[1],
                        "fields": fields,
                        "risk_score": row[3],
                        "similarity": row[4]  # Renamed from match_score to similarity for consistency
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
            record_id = await self.store_embedding(document_type, fields, embedding, risk_score, file_name)
            
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
