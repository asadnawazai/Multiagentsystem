import re
import json
import spacy
import dateparser
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from loguru import logger
import price_parser


class NLUExtractionAgent:
    """Agent responsible for extracting structured fields from text.
    
    This agent uses a combination of regex patterns, spaCy NLP, and other
    techniques to identify and extract key information fields from text.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the NLU Extraction Agent.
        
        Args:
            spacy_model: Name of the spaCy model to use
        """
        self.nlp = None
        try:
            # First check if spacy is available
            if spacy.__version__:
                try:
                    self.nlp = spacy.load(spacy_model)
                    logger.info(f"Loaded spaCy model: {spacy_model}")
                except Exception as e:
                    logger.warning(f"Could not load spaCy model {spacy_model}: {e}")
                    logger.warning("To download the model, run: python -m spacy download en_core_web_sm")
                    logger.info("Continuing without NLP capabilities (regex-only mode)")
        except Exception as e:
            logger.warning(f"spaCy not available: {e}")
            logger.warning("NLU extraction will use regex patterns only")
        
        # Patterns for field extraction
        self.patterns = {
            "name": [
                r'(?i)name[\s:]*([\w\s.-]+)',
                r'(?i)policyholder[\s:]*([\w\s.-]+)',
                r'(?i)insured[\s:]*([\w\s.-]+)',
                r'(?i)client[\s:]*([\w\s.-]+)',
                r'(?i)customer[\s:]*([\w\s.-]+)'
            ],
            "policy_number": [
                r'(?i)policy[\s#:]*([A-Z0-9-]+)',
                r'(?i)policy\s+number[\s:]*([A-Z0-9-]+)',
                r'(?i)contract[\s#:]*([A-Z0-9-]+)'
            ],
            "claim_number": [
                r'(?i)claim[\s#:]*([A-Z0-9-]+)',
                r'(?i)claim\s+number[\s:]*([A-Z0-9-]+)',
                r'(?i)case[\s#:]*([A-Z0-9-]+)'
            ],
            "date": [
                r'(?i)date[\s:]*([\d\w\s,/-]+)',
                r'(?i)submitted\s+on[\s:]*([\d\w\s,/-]+)',
                r'(?i)filed\s+on[\s:]*([\d\w\s,/-]+)',
                r'(?i)reported\s+on[\s:]*([\d\w\s,/-]+)'
            ]
        }
    
    async def extract_fields(self, text: str) -> Dict[str, Any]:
        """Extract structured fields from text.
        
        Args:
            text: Normalized text to extract fields from
            
        Returns:
            Dictionary of extracted fields
        """
        try:
            # Initialize the results dictionary
            fields = {}
            confidence_scores = {}
            
            # Run spaCy NLP if available
            doc = None
            if self.nlp:
                try:
                    doc = self.nlp(text)
                except Exception as e:
                    logger.warning(f"Error processing text with spaCy: {e}")
            
            # Extract various fields
            fields["name"], confidence_scores["name"] = self._extract_name(text, doc)
            fields["policy_number"], confidence_scores["policy_number"] = self._extract_pattern(text, self.patterns["policy_number"])
            fields["claim_number"], confidence_scores["claim_number"] = self._extract_pattern(text, self.patterns["claim_number"])
            fields["date"], confidence_scores["date"] = self._extract_date(text, doc)
            fields["amount"], confidence_scores["amount"] = self._extract_amount(text)
            fields["document_type"] = self._identify_document_type(text)
            
            # Remove None values
            fields = {k: v for k, v in fields.items() if v is not None}
            
            # If no fields were extracted, add a placeholder
            if not fields:
                fields["document_type"] = "Unknown Document"
                fields["note"] = "No specific fields could be extracted from this document"
            
            logger.info(f"Extracted {len(fields)} fields from document")
            
            return {
                "fields": fields,
                "confidence_scores": confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error extracting fields: {str(e)}")
            raise
    
    def _extract_name(self, text: str, doc) -> Tuple[Optional[str], float]:
        """Extract person name from text.
        
        Uses a combination of regex patterns and NER.
        
        Args:
            text: Text to extract from
            doc: spaCy Doc object (or None)
            
        Returns:
            Tuple of (extracted name, confidence score)
        """
        # Try regex patterns first
        for pattern in self.patterns["name"]:
            match = re.search(pattern, text)
            if match and match.group(1).strip():
                name = match.group(1).strip()
                # Simple validation: names shouldn't be too short or too long
                if 4 <= len(name) <= 40:
                    return name, 0.7  # Moderate confidence for regex matches
        
        # If spaCy is available, try NER
        if doc:
            person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if person_entities:
                # Take the first PERSON entity as the name
                return person_entities[0], 0.9  # Higher confidence for NER
        
        return None, 0.0
    
    def _extract_pattern(self, text: str, patterns: List[str]) -> Tuple[Optional[str], float]:
        """Extract text using multiple regex patterns.
        
        Args:
            text: Text to extract from
            patterns: List of regex patterns to try
            
        Returns:
            Tuple of (extracted value, confidence score)
        """
        for pattern in patterns:
            match = re.search(pattern, text)
            if match and match.group(1).strip():
                return match.group(1).strip(), 0.8
        
        return None, 0.0
    
    def _extract_date(self, text: str, doc) -> Tuple[Optional[str], float]:
        """Extract date from text.
        
        Uses a combination of regex patterns and dateparser.
        
        Args:
            text: Text to extract from
            doc: spaCy Doc object (or None)
            
        Returns:
            Tuple of (extracted date in ISO format, confidence score)
        """
        # Try regex patterns first
        for pattern in self.patterns["date"]:
            match = re.search(pattern, text)
            if match and match.group(1).strip():
                date_str = match.group(1).strip()
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date.strftime("%Y-%m-%d"), 0.8
        
        # If spaCy is available, try NER for dates
        if doc:
            date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            for date_str in date_entities:
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date.strftime("%Y-%m-%d"), 0.9
        
        # Look for common date formats in the text
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY or DD-MM-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date.strftime("%Y-%m-%d"), 0.7
        
        return None, 0.0
    
    def _extract_amount(self, text: str) -> Tuple[Optional[str], float]:
        """Extract monetary amount from text.
        
        Uses price-parser library and regex patterns.
        
        Args:
            text: Text to extract from
            
        Returns:
            Tuple of (extracted amount, confidence score)
        """
        # Use price-parser to find amounts
        amounts = []
        
        # First look for specific amount-related keywords
        amount_patterns = [
            r'(?i)amount[\s:]*\$?(\d[\d,.]*)',
            r'(?i)claim[\s:]*\$?(\d[\d,.]*)',
            r'(?i)total[\s:]*\$?(\d[\d,.]*)',
            r'(?i)sum[\s:]*\$?(\d[\d,.]*)',
            r'(?i)policy\s+value[\s:]*\$?(\d[\d,.]*)',
            r'(?i)coverage[\s:]*\$?(\d[\d,.]*)',
            r'(?i)premium[\s:]*\$?(\d[\d,.]*)',
            r'(?i)\$\s*(\d[\d,.]*)'  # Any dollar amount
        ]
        
        for pattern in amount_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.group(1).strip():
                    amount_str = match.group(1).strip().replace(',', '')
                    try:
                        amount = float(amount_str)
                        amounts.append((f"${amount:.2f}", 0.9))  # High confidence for pattern matches
                    except ValueError:
                        pass
        
        # If we didn't find any amounts with patterns, try using price-parser
        if not amounts:
            # Find all potential price mentions
            price_matches = re.finditer(r'\$?\d+(?:[.,]\d+)*(?:\s*(?:dollars|USD))?', text)
            for match in price_matches:
                price_str = match.group(0)
                price = price_parser.parse_price(price_str)
                if price.amount is not None:
                    amounts.append((f"${price.amount:.2f}", 0.7))  # Moderate confidence
        
        # Return the highest amount found, which is often the most relevant one
        if amounts:
            amounts.sort(key=lambda x: float(x[0].replace('$', '').replace(',', '')), reverse=True)
            return amounts[0]
        
        return None, 0.0
    
    def _identify_document_type(self, text: str) -> Optional[str]:
        """Identify the type of document based on text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Document type classification
        """
        text_lower = text.lower()
        
        # Check for common document type indicators
        if "claim form" in text_lower or "insurance claim" in text_lower:
            return "Insurance Claim"
        elif "policy" in text_lower and ("renewal" in text_lower or "certificate" in text_lower):
            return "Insurance Policy"
        elif "quote" in text_lower or "proposal" in text_lower:
            return "Insurance Quote"
        elif "invoice" in text_lower or "bill" in text_lower:
            return "Invoice"
        elif "assessment" in text_lower or "evaluation" in text_lower:
            return "Risk Assessment"
        elif "report" in text_lower:
            return "Report"
        elif "application" in text_lower or "enrollment" in text_lower:
            return "Application Form"
        elif "contract" in text_lower or "agreement" in text_lower:
            return "Contract"
        
        # Default if no specific type is identified
        return "Unknown Document"
