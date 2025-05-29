import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from loguru import logger

class YAMLAdapter:
    """Adapter for managing YAML configuration files, particularly for scoring logic."""
    
    def __init__(self, yaml_path: str):
        """Initialize the YAML adapter.
        
        Args:
            yaml_path: Path to the YAML configuration file
        """
        self.yaml_path = yaml_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration from file.
        
        Returns:
            Dict containing the YAML configuration
        """
        try:
            if not os.path.exists(self.yaml_path):
                logger.warning(f"YAML file not found at {self.yaml_path}. Creating empty config.")
                return {}
                
            with open(self.yaml_path, 'r') as file:
                config = yaml.safe_load(file) or {}
            return config
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            return {}
            
    def load_config(self) -> Dict[str, Any]:
        """Public method to load or return the YAML configuration.
        
        Returns:
            Dict containing the YAML configuration
        """
        return self.config
    
    def _save_config(self) -> bool:
        """Save the current configuration to the YAML file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
            
            with open(self.yaml_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logger.error(f"Error saving YAML config: {e}")
            return False
    
    def update_fields(self, extracted_fields: Dict[str, str]) -> bool:
        """Update the YAML configuration with extracted fields.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize fields section if it doesn't exist
            if 'fields' not in self.config:
                self.config['fields'] = {}
            
            # Update each field with appropriate weights and thresholds
            for field_name, field_value in extracted_fields.items():
                # Skip empty values
                if not field_value:
                    continue
                    
                # Initialize field if it doesn't exist
                if field_name not in self.config['fields']:
                    self.config['fields'][field_name] = {
                        'weight': self._get_default_weight(field_name),
                        'thresholds': self._get_default_thresholds(field_name, field_value)
                    }
                else:
                    # Update thresholds based on new value
                    self._update_thresholds(field_name, field_value)
            
            # Save the updated configuration
            return self._save_config()
        except Exception as e:
            logger.error(f"Error updating fields in YAML config: {e}")
            return False
    
    def _get_default_weight(self, field_name: str) -> float:
        """Get the default weight for a field based on its importance.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Default weight value
        """
        # Define weights based on field importance in real estate domain
        weights = {
            'amount': 0.4,
            'tax_value': 0.3,
            'flood_zone': 0.3,
            'property_address': 0.2,
            'parcel_id': 0.2,
            'year_built': 0.15,
            'bedrooms': 0.1,
            'bathrooms': 0.1,
            'mls_number': 0.05
        }
        
        # Return default weight or fallback
        return weights.get(field_name, 0.1)
    
    def _get_default_thresholds(self, field_name: str, field_value: str) -> Dict[str, Any]:
        """Get the default thresholds for a field based on its type and value.
        
        Args:
            field_name: Name of the field
            field_value: Value of the field
            
        Returns:
            Default thresholds
        """
        # Process numeric fields
        if field_name in ['amount', 'tax_value']:
            # Extract numeric value from string (remove currency symbols, commas, etc.)
            try:
                numeric_value = self._extract_numeric_value(field_value)
                if numeric_value is not None:
                    return {
                        'high': numeric_value * 2,  # Double the current value
                        'medium': numeric_value,    # Current value
                        'low': numeric_value / 2    # Half the current value
                    }
            except:
                pass
                
            # Fallback thresholds for monetary values
            return {
                'high': 1000000,
                'medium': 500000,
                'low': 200000
            }
        
        # Process year fields
        elif field_name == 'year_built':
            try:
                year = int(field_value)
                return {
                    'new': max(year, 2000),
                    'medium': max(min(year, 2000), 1960),
                    'old': min(year, 1960)
                }
            except:
                return {
                    'new': 2000,
                    'medium': 1980,
                    'old': 1960
                }
        
        # Process bedroom/bathroom counts
        elif field_name in ['bedrooms', 'bathrooms']:
            try:
                count = float(field_value)
                return {
                    'high': count + 2,
                    'medium': count,
                    'low': max(count - 1, 1)
                }
            except:
                return {
                    'high': 4,
                    'medium': 3,
                    'low': 2
                }
        
        # Process flood zone
        elif field_name == 'flood_zone':
            # Flood zones use categorical data, not thresholds
            return {
                'categories': ['A', 'AE', 'AH', 'AO', 'X', 'X500'],
                'high_risk': ['A', 'AE', 'AH', 'AO'],
                'medium_risk': ['X500'],
                'low_risk': ['X']
            }
        
        # Generic thresholds for other fields
        return {
            'categories': [field_value] if field_value else []
        }
    
    def _update_thresholds(self, field_name: str, field_value: str) -> None:
        """Update thresholds for an existing field based on new data.
        
        Args:
            field_name: Name of the field
            field_value: New value for the field
        """
        # Skip if field doesn't exist in config
        if field_name not in self.config['fields']:
            return
        
        field_config = self.config['fields'][field_name]
        
        # Skip if no thresholds exist
        if 'thresholds' not in field_config:
            field_config['thresholds'] = self._get_default_thresholds(field_name, field_value)
            return
        
        # Update numeric thresholds
        if field_name in ['amount', 'tax_value']:
            try:
                numeric_value = self._extract_numeric_value(field_value)
                if numeric_value is None:
                    return
                    
                thresholds = field_config['thresholds']
                
                # Update thresholds based on the new value
                if numeric_value > thresholds.get('high', 0):
                    thresholds['high'] = numeric_value
                elif numeric_value < thresholds.get('low', float('inf')):
                    thresholds['low'] = numeric_value
            except:
                pass
        
        # Update categorical data
        elif 'categories' in field_config['thresholds']:
            # Add new category if not already present
            if field_value and field_value not in field_config['thresholds']['categories']:
                field_config['thresholds']['categories'].append(field_value)
    
    def _extract_numeric_value(self, value_str: str) -> Optional[float]:
        """Extract numeric value from a string.
        
        Args:
            value_str: String containing a numeric value
            
        Returns:
            Extracted numeric value or None if extraction fails
        """
        import re
        
        try:
            # Remove currency symbols, commas, etc.
            cleaned = re.sub(r'[^0-9.]', '', value_str)
            
            # Convert to float
            return float(cleaned) if cleaned else None
        except:
            return None
