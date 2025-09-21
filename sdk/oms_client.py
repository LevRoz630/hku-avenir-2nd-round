"""
OMS Client SDK

A requests-based OMS service client that provides complete API interface functionality.
Reads OMS_URL and OMS_ACCESS_TOKEN configuration from environment variables.

Usage example:
    >>> from sdk.oms_client import OmsClient
    >>> client = OmsClient()
    >>> balances = client.get_balance()
    >>> positions = client.get_position()
    >>> result = client.set_target_position(
    ...     instrument_name="BTC-USDT",
    ...     instrument_type="future", 
    ...     target_value=100,
    ...     position_side="LONG"
    ... )
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import requests


logger = logging.getLogger(__name__)


class OmsError(Exception):
    """OMS SDK base exception class"""
    pass


class AuthenticationError(OmsError):
    """Authentication error"""
    pass


class ApiError(OmsError):
    """API call error"""
    pass


class RateLimitError(OmsError):
    """Rate limit error"""
    pass


class ConfigurationError(OmsError):
    """Configuration error"""
    pass


class OmsClient:
    """
    OMS SDK Client
    
    Reads configuration from environment variables:
    - OMS_URL: OMS service base URL
    - OMS_ACCESS_TOKEN: Bearer authentication token
    
    Usage example:
        >>> client = OmsClient()
        >>> balances = client.get_balance()
        >>> positions = client.get_position()
        >>> client.set_target_position("BTC-USDT", "future", 100, "LONG")
    """
    
    def __init__(self, base_url: Optional[str] = None, access_token: Optional[str] = None, timeout: int = 30):
        """
        Initialize OMS client
        
        Args:
            base_url: OMS service base URL, if not provided, reads from OMS_URL environment variable
            access_token: Bearer authentication token, if not provided, reads from OMS_ACCESS_TOKEN environment variable
            timeout: Request timeout time (seconds)
            
        Raises:
            ConfigurationError: Missing or invalid configuration
        """

        # Get configuration from environment variables or parameters
        self.base_url = (base_url or os.getenv('OMS_URL', '')).rstrip('/')
        self.access_token = access_token or os.getenv('OMS_ACCESS_TOKEN', '')
        self.timeout = timeout
        
        # Validate configuration
        if not self.base_url:
            raise ConfigurationError("OMS_URL environment variable or base_url parameter is required")
        if not self.access_token:
            raise ConfigurationError("OMS_ACCESS_TOKEN environment variable or access_token parameter is required")
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"OmsClient initialized with base_url: {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None, raw=False) -> Dict[str, Any]:
        """
        Send HTTP request
        
        Args:
            method: HTTP method (GET/POST/PUT/DELETE)
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data dictionary
            
        Raises:
            AuthenticationError: Authentication failed
            RateLimitError: Rate limit exceeded
            ApiError: Other API errors
        """
        url = f"{self.base_url}{endpoint}"
        prepared = requests.Request(method=method, url=url, params=params, json=data, headers=self.session.headers).prepare()
        try:
            response = self.session.send(prepared, timeout=self.timeout)
            if raw:
                return response
        
            # Handle response status codes
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed. Please check your access token.")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded. Please try again later.")
            elif response.status_code >= 400:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        error_msg = f"{error_msg} - {error_data['message']}"
                    elif 'error' in error_data:
                        error_msg = f"{error_msg} - {error_data['error']}"
                except:
                    error_msg = f"{error_msg} - {response.text}"
                raise ApiError(error_msg)
            
            # Parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                raise ApiError(f"Invalid JSON response: {response.text}")
            
        except requests.exceptions.Timeout:
            raise ApiError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise ApiError(f"Connection error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise ApiError(f"Request error: {str(e)}")

    def set_target_position(self, instrument_name: str, instrument_type: str, 
                          target_value: float, position_side: str) -> Dict[str, Any]:
        """
        Set target position
        
        Args:
            instrument_name: Trading pair name, e.g. "BTC-USDT"
            instrument_type: Trading type, "future" or "spot"
            target_value: Target position value (USDT)
            position_side: Position direction, "LONG" or "SHORT"
            
        Returns:
            Setting result response dictionary, containing task ID and other information
            
        Raises:
            ApiError: API call failed
            RateLimitError: Operation too frequent
        """
        data = {
            "instrument_name": instrument_name,
            "instrument_type": instrument_type,
            "target_value": target_value,
            "position_side": position_side
        }

        try:
            response = self._make_request('POST', '/api/binance/set-target-position', data=data)
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to set target position: {error_msg}")
            
            logger.info(f"Target position set successfully for {instrument_name}")
            return response['message']
            
        except Exception as e:
            logger.error(f"Error setting target position for {instrument_name}: {str(e)}")
            raise

    def set_target_position_batch(self, elements: List) -> Dict[str, Any]:
        """
        Batch set target positions

        Args:
            elements: List of dictionaries containing multiple target position settings, each dictionary contains the following fields:
                - instrument_name: Trading pair name, e.g. "BTC-USDT"
                - instrument_type: Trading type, "future" or "spot"
                - target_value: Target position value (USDT)
                - position_side: Position direction, "LONG" or "SHORT"

        Returns:
            Setting result response dictionary, containing task ID and other information

        Raises:
            ApiError: API call failed
            RateLimitError: Operation too frequent
        """

        try:
            response = self._make_request('POST', '/api/binance/set-target-position-batch', data=elements)

            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to set target position batch: {error_msg}")

            logger.info(f"Target position batch set successfully")
            return response['message']

        except Exception as e:
            logger.error(f"Error setting target position batch: {str(e)}")
            raise

    def get_position(self) -> List[Dict[str, Any]]:
        """
        Get user position list
        
        Returns:
            List of positions; each includes instrument, side, quantity, value
            
        Raises:
            ApiError: API call failed
        """
        try:
            response = self._make_request('GET', '/api/binance/get-position')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get positions: {error_msg}")
            
            positions = response['message']
            logger.info(f"Retrieved {len(positions)} position records")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise

    def get_balance(self) -> List[Dict[str, Any]]:
        """
        Get user asset list
        
        Returns:
            Asset information list, each element contains asset type, balance and other information
            
        Raises:
            ApiError: API call failed
        """
        try:
            response = self._make_request('GET', '/api/binance/get-balance')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get balances: {error_msg}")
            
            balances = response['message']
            logger.info(f"Retrieved {len(balances)} balance records")
            return balances
            
        except Exception as e:
            logger.error(f"Error getting balances: {str(e)}")
            raise

    def get_asset_changes(self) -> List[Dict[str, Any]]:
        """
        Get user asset change history
        
        Returns:
            Asset change record list, containing the latest 100 change records
            
        Raises:
            ApiError: API call failed
        """
        try:
            response = self._make_request('GET', '/api/binance/get-asset-changes')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get asset changes: {error_msg}")
            
            changes = response['message']
            logger.info(f"Retrieved {len(changes)} asset change records")
            return changes
            
        except Exception as e:
            logger.error(f"Error getting asset changes: {str(e)}")
            raise

    def get_symbols(self) -> List[str]:
        """
        Get tradable contract list
        
        Returns:
            Tradable contract list

        Raises:
            ApiError: API call failed
        """
        try:
            response = self._make_request('GET', '/api/market/symbols')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get symbols: {error_msg}")

            symbols = response['symbols']
            logger.info(f"Retrieved {len(symbols)} symbol records")
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise

    def create_strategy_user(self, name: str) -> Dict[str, str]:
        """
        Create strategy user (requires qb-backend permissions)
        
        Args:
            name: Username
            
        Returns:
            Dictionary containing username and token
            
        Raises:
            ApiError: API call failed
            AuthenticationError: Insufficient permissions
        """
        data = {"name": name}
        
        try:
            response = self._make_request('POST', '/api/strategy', data)
            
            if 'error' in response:
                raise ApiError(f"Failed to create strategy user: {response['error']}")
            
            logger.info(f"Strategy user '{name}' created successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error creating strategy user '{name}': {str(e)}")
            raise


    def close(self):
        """Close client connection"""
        if self.session:
            self.session.close()
        logger.info("OmsClient closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function
def create_client(base_url: Optional[str] = None, access_token: Optional[str] = None) -> OmsClient:
    """
    Create OMS client instance
    
    Args:
        base_url: OMS service base URL, if not provided, reads from OMS_URL environment variable
        access_token: Bearer authentication token, if not provided, reads from OMS_ACCESS_TOKEN environment variable
        
    Returns:
        OmsClient instance
    """
    return OmsClient(base_url=base_url, access_token=access_token)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create client (read configuration from environment variables)
        client = OmsClient()
        
        # Get account information
        print("=== Get Asset Balances ===")
        balances = client.get_balance()
        for balance in balances:
            print(f"Asset: {balance['asset']}, Balance: {balance['balance']}")
        
        print("\n=== Get Position Information ===")
        positions = client.get_position()
        for position in positions:
            print(f"Trading Pair: {position['instrument_name']}, "
                  f"Direction: {position['position_side']}, "
                  f"Quantity: {position['quantity']}, "
                  f"Value: {position['value']}")
        
        print("\n=== Get Asset Change History ===")
        changes = client.get_asset_changes()
        for change in changes[:5]:  # Only show first 5 records
            print(f"Asset: {change['asset']}, "
                  f"Change: {change['change']}, "
                  f"Balance: {change['balance']}, "
                  f"Time: {change['create_time']}")
        
        # Set target position example (commented out to avoid accidental execution)
        print("\n=== Set Target Position ===")
        result = client.set_target_position(
            instrument_name="BTC-USDT",
            instrument_type="future",
            target_value=100,
            position_side="LONG"
        )
        print(f"Task ID: {result['id']}")
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print("Please set environment variables OMS_URL and OMS_ACCESS_TOKEN")
    except Exception as e:
        print(f"Error: {e}")
    
    import time
    while True:
        time.sleep(1)
