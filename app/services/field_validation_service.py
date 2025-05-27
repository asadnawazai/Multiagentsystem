from typing import Dict, List, Optional
from loguru import logger

class FieldValidationService:
    """Service for validating extracted fields and prompting for missing critical fields."""
    
    def __init__(self, config_path: str):
        """Initialize the field validation service.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.critical_fields = []
        self._load_config()
    
    def _load_config(self) -> None:
        """Load the field validation configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            self.critical_fields = config.get('critical_fields', [])
            logger.info(f"Loaded critical fields: {self.critical_fields}")
        except Exception as e:
            logger.error(f"Error loading field validation config: {e}")
            # Set defaults if config loading fails
            self.critical_fields = ["parcel_id", "tax_value", "property_address", "MLS_number"]
    
    def check_missing_fields(self, fields: Dict) -> Dict:
        """Check for missing critical fields in the extracted data.
        
        Args:
            fields: Dictionary of extracted fields
            
        Returns:
            Dict with missing_fields list and prompt_message if needed
        """
        missing_fields = []
        
        for field in self.critical_fields:
            if field not in fields or not fields[field]:
                missing_fields.append(field)
                
        result = {
            "missing_fields": missing_fields,
            "has_missing_fields": len(missing_fields) > 0,
            "prompt_message": ""
        }
        
        if missing_fields:
            formatted_fields = [field.replace('_', ' ').title() for field in missing_fields]
            field_list = ", ".join(formatted_fields)
            result["prompt_message"] = f"Some fields are missing. Please enter manually: {field_list}"
            logger.warning(f"Missing critical fields: {missing_fields}")
            
        return result
    
    def format_field_label(self, field_name: str) -> str:
        """Format a field name for display in the UI.
        
        Args:
            field_name: Raw field name from the system
            
        Returns:
            str: Formatted field name for display
        """
        return field_name.replace('_', ' ').title()
