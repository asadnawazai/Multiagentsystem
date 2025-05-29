import yaml
from typing import Dict, List, Any
from loguru import logger


class RiskScoringService:
    """Service for calculating risk scores based on document fields."""
    
    def __init__(self, config_path: str):
        """Initialize the risk scoring service.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.fields_config = {}
        self.risk_bands = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load the risk scoring configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            self.fields_config = config.get('fields', {})
            self.risk_bands = config.get('risk_bands', {'high': 75, 'moderate': 50, 'low': 25})
            logger.info(f"Loaded risk scoring configuration for {len(self.fields_config)} fields")
        except Exception as e:
            logger.error(f"Error loading risk scoring config: {e}")
            # Set defaults if config loading fails
            self.fields_config = {}
            self.risk_bands = {'high': 75, 'moderate': 50, 'low': 25}
    
    def calculate_risk_score(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score based on field values and configured thresholds.
        
        Args:
            fields: Dictionary of extracted field values
            
        Returns:
            Dict with risk_score, risk_band, and contributing_factors
        """
        total_score = 0
        total_weight = 0
        contributing_factors = []
        
        for field_name, field_config in self.fields_config.items():
            # Only include fields that were actually found in the document
            # Skip any fields marked as 'Not Found' or empty values
            if field_name in fields and fields[field_name] and fields[field_name] != 'Not Found':
                field_value = fields[field_name]
                weight = field_config.get('weight', 0.1)
                thresholds = field_config.get('thresholds', {})
                
                # Score this field
                field_score, risk_level = self._score_field(field_name, field_value, thresholds)
                
                # Add to total score, weighted appropriately
                total_score += field_score * weight
                total_weight += weight
                
                # Add to contributing factors if significant
                if risk_level in ['high', 'medium']:
                    field_formatted = field_name.replace('_', ' ').title()
                    contributing_factors.append(f"{risk_level.title()} {field_formatted} risk")
        
        # Normalize score to 0-100 range
        if total_weight > 0:
            normalized_score = round((total_score / total_weight) * 100)
        else:
            normalized_score = 50  # Default score if no scorable fields
        
        # Determine risk band
        risk_band = self._determine_risk_band(normalized_score)
        
        return {
            "risk_score": normalized_score,
            "risk_band": risk_band,
            "contributing_factors": contributing_factors
        }
    
    def _score_field(self, field_name: str, field_value: Any, thresholds: Dict) -> tuple:
        """Score an individual field based on thresholds.
        
        Args:
            field_name: Name of the field
            field_value: Value of the field
            thresholds: Configured thresholds for this field
            
        Returns:
            tuple: (score, risk_level) where score is 0-1 and risk_level is 'high', 'medium', or 'low'
        """
        # Default to medium risk if we can't determine
        score = 0.5
        risk_level = 'medium'
        
        try:
            # Convert value to appropriate type based on field
            if field_name in ['tax_appraisal', 'tax_value', 'property_value']:
                # Handle numeric values like currency
                if isinstance(field_value, str):
                    field_value = field_value.replace('$', '').replace(',', '')
                numeric_value = float(field_value)
                
                # Score based on thresholds
                if 'high' in thresholds and numeric_value >= float(thresholds['high']):
                    score = 1.0  # High risk
                    risk_level = 'high'
                elif 'medium' in thresholds and numeric_value >= float(thresholds['medium']):
                    score = 0.7  # Medium risk
                    risk_level = 'medium'
                elif 'low' in thresholds and numeric_value >= float(thresholds['low']):
                    score = 0.3  # Low risk
                    risk_level = 'low'
                else:
                    score = 0.1  # Very low risk
                    risk_level = 'low'
                    
            elif field_name in ['year_built', 'construction_year']:
                # For years, older properties are typically higher risk
                numeric_value = int(field_value)
                
                if 'high' in thresholds and numeric_value <= int(thresholds['high']):
                    score = 1.0  # High risk for very old buildings
                    risk_level = 'high'
                elif 'medium' in thresholds and numeric_value <= int(thresholds['medium']):
                    score = 0.7  # Medium risk
                    risk_level = 'medium'
                elif 'low' in thresholds and numeric_value <= int(thresholds['low']):
                    score = 0.3  # Low risk
                    risk_level = 'low'
                else:
                    score = 0.1  # Very low risk for new buildings
                    risk_level = 'low'
                    
            else:
                # Handle string-based categorical fields
                str_value = str(field_value).upper()
                
                if 'high' in thresholds and str_value == str(thresholds['high']).upper():
                    score = 1.0
                    risk_level = 'high'
                elif 'medium' in thresholds and str_value == str(thresholds['medium']).upper():
                    score = 0.7
                    risk_level = 'medium'
                else:
                    score = 0.3
                    risk_level = 'low'
                    
        except (ValueError, TypeError) as e:
            logger.warning(f"Error scoring field {field_name} with value {field_value}: {e}")
            # Default to medium risk if we can't determine
            score = 0.5
            risk_level = 'medium'
            
        return score, risk_level
    
    def _determine_risk_band(self, score: int) -> str:
        """Determine the risk band based on the numeric score.
        
        Args:
            score: Numeric risk score (0-100)
            
        Returns:
            str: Risk band ('High', 'Moderate', or 'Low')
        """
        if score >= self.risk_bands.get('high', 75):
            return "High"
        elif score >= self.risk_bands.get('moderate', 50):
            return "Moderate"
        else:
            return "Low"
