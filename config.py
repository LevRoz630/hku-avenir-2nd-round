"""
Configuration module for reading OMS credentials and settings
"""

import os
from pathlib import Path
from typing import Dict, Optional

class Config:
    """Configuration manager for OMS credentials and settings"""
    
    def __init__(self, oms_file: str = "oms.txt"):
        self.oms_file = Path(oms_file)
        self.credentials = self._load_oms_credentials()
    
    def _load_oms_credentials(self) -> Dict[str, str]:
        """Load OMS credentials from oms.txt file"""
        credentials = {}
        
        if not self.oms_file.exists():
            print(f"Warning: {self.oms_file} not found. Using environment variables.")
            return {}
        
        try:
            with open(self.oms_file, 'r') as f:
                lines = f.readlines()
            
            # Parse the file format:
            # Line 1: empty
            # Line 2: test
            # Line 3: test_token
            # Line 4: empty  
            # Line 5: prod
            # Line 6: prod_token
            
            if len(lines) >= 6:
                credentials = {
                    'test_token': lines[2].strip(),
                    'prod_token': lines[5].strip()
                }
                print("OMS credentials loaded from file")
            else:
                print(f"Warning: {self.oms_file} format incorrect. Expected 6 lines.")
                
        except Exception as e:
            print(f"Error reading {self.oms_file}: {e}")
        
        return credentials
    
    def get_oms_config(self, mode: str = "test") -> Dict[str, str]:
        """Get OMS configuration for specified mode"""
        if mode not in ["test", "prod"]:
            raise ValueError("Mode must be 'test' or 'prod'")
        
        # Default URLs
        urls = {
            "test": "https://quant-competition-oms-test.yorkapp.com",
            "prod": ""  # Will be set by deployment
        }
        
        # Get token from file or environment
        token_key = f"{mode}_token"
        token = self.credentials.get(token_key) or os.getenv('OMS_ACCESS_TOKEN', '')
        
        if not token:
            raise ValueError(f"No {mode} token found in oms.txt or environment variables")
        
        return {
            'OMS_URL': urls[mode] if urls[mode] else os.getenv('OMS_URL', ''),
            'OMS_ACCESS_TOKEN': token
        }
    
    def setup_environment(self, mode: str = "test"):
        """Set up environment variables for OMS"""
        config = self.get_oms_config(mode)
        
        for key, value in config.items():
            if value:  # Only set if value exists
                os.environ[key] = value
                print(f"Set {key} for {mode} mode")
    
    def get_available_modes(self) -> list:
        """Get list of available modes based on loaded credentials"""
        modes = []
        if self.credentials.get('test_token'):
            modes.append('test')
        if self.credentials.get('prod_token'):
            modes.append('prod')
        return modes

# Global config instance
config = Config()

def get_oms_config(mode: str = "test") -> Dict[str, str]:
    """Convenience function to get OMS config"""
    return config.get_oms_config(mode)

def setup_environment(mode: str = "test"):
    """Convenience function to setup environment"""
    config.setup_environment(mode)
