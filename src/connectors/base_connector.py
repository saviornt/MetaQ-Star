# ./connectors/base_connector.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from src.utils.logger import setup_logger

class BaseConnector(ABC):
    """
    Abstract base class for all database connectors.
    
    Provides a standardized interface for connecting, disconnecting, and executing CRUD operations.
    """
    
    def __init__(self):
        """
        Initializes the connector and sets up the logger.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.connection = None
    
    @abstractmethod
    async def connect(self):
        """
        Establishes a connection to the database.
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """
        Closes the connection to the database.
        """
        pass
    
    @abstractmethod
    async def create(self, table: str, data: Dict[str, Any]) -> Any:
        """
        Inserts a new record into the specified table.
        
        Args:
            table (str): The name of the table.
            data (Dict[str, Any]): The data to insert.
        
        Returns:
            Any: The identifier of the created record.
        """
        pass
    
    @abstractmethod
    async def read(self, table: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieves records from the specified table based on the query.
        
        Args:
            table (str): The name of the table.
            query (Dict[str, Any]): The filter criteria.
        
        Returns:
            List[Dict[str, Any]]: A list of matching records.
        """
        pass
    
    @abstractmethod
    async def update(self, table: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """
        Updates records in the specified table based on the query.
        
        Args:
            table (str): The name of the table.
            query (Dict[str, Any]): The filter criteria.
            update_data (Dict[str, Any]): The data to update.
        
        Returns:
            int: The number of records updated.
        """
        pass
    
    @abstractmethod
    async def delete(self, table: str, query: Dict[str, Any]) -> int:
        """
        Deletes records from the specified table based on the query.
        
        Args:
            table (str): The name of the table.
            query (Dict[str, Any]): The filter criteria.
        
        Returns:
            int: The number of records deleted.
        """
        pass
