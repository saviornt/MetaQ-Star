import asyncio
from typing import Any, Dict, List, Optional
from src.connectors.mongodb_connector import MongoDBConnector
from src.connectors.sqlite_connector import SQLiteConnector
from src.connectors.memory_connector import MemoryConnector
from src.connectors.redis_connector import RedisConnector

class CRUDManager:
    """
    Database manager for handling multiple database connections.

    Attributes:
        connector (Any): The current connector instance used for database operations.
    """

    CONNECTOR_CLASSES = {
                "mongodb": MongoDBConnector,
                "sqlite": SQLiteConnector,
                "memory": MemoryConnector,
                "redis": RedisConnector,
            }

    def __init__(self, db_type: str):
        connector_class = self.CONNECTOR_CLASSES.get(db_type)
        if not connector_class:
            raise ValueError(f"Invalid database type: {db_type}")
        self.connector = connector_class()

    async def connect(self):
        """
        Connect to the database.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.connect)
        await self.connector.connect()

    async def disconnect(self):
        """
        Disconnect from the database.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.disconnect)
        await self.connector.disconnect()
    
    async def create(self, collection: str, schema: Optional[Dict[Any, Any]]) -> Any:
        """
        Create a new collection or table in the database.
        """
        if isinstance(self.connector, MemoryConnector):
            return # TODO: Look into creating a block of system memory and return its ID to use as the "collection"
        return await self.connector.create(collection, schema)
    
    async def insert_one(self, collection: Optional[str], data: Dict[str, Any], **kwargs) -> Any:
        """
        Create a new record in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to insert the data into.
            data (Dict[str, Any]): The data to insert into the collection or table.
            **kwargs: Additional keyword arguments to pass to the connector's insert method.

        Returns:
            Any: The result of the insert operation.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.insert_one, data, **kwargs)
        return await self.connector.insert_one(collection, data, **kwargs)
    
    async def insert_many(self, collection: Optional[str], data_list: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Create multiple new records in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to insert the data into.
            data (List[Dict[str, Any]]): The list of data to insert into the collection or table.

        Returns:
            Any: The result of the insert operation.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.insert_many, data_list, **kwargs)
        return await self.connector.insert_many(collection, data_list, **kwargs)
    
    async def find_one(self, collection: Optional[str], key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single record in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to find the record in.
            key (Dict[str, Any]): The key to search for.

        Returns:
            Optional[Dict[str, Any]]: The record found in the collection or table, or None if no record is found.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.find_one, key)
        return await self.connector.find_one(collection, key)
    
    async def find_many(self, collection: Optional[str], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find multiple records in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to find the records in.
            filters (Dict[str, Any]): The filters to apply to the search.

        Returns:
            List[Dict[str, Any]]: The list of records found in the collection or table.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.find_many, list(filters.keys()))
        return await self.connector.find_many(collection, filters)
    
    async def update_one(self, collection: Optional[str], key: Dict[str, Any], data: Dict[str, Any], **kwargs) -> bool:
        """
        Update a single record in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to update the record in.
            key (Dict[str, Any]): The key to search for.
            data (Dict[str, Any]): The data to update the record with.
        
        Returns:
            bool: True if the record was updated, False otherwise.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.update_one, key, data, **kwargs)
        return await self.connector.update_one(collection, key, data, **kwargs)
    
    async def update_many(self, collection: Optional[str], data: List[Dict[str, Any]], **kwargs) -> int:
        """
        Update multiple records in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to update the records in.
            data (List[Dict[str, Any]]): The list of data to update the records with, each containing:
                - key (Dict[str, Any]): The key to identify the record being updated.
                - data (Dict[str, Any]): The data to update the record with.

        Returns:
            int: The number of records updated.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.update_many, data, **kwargs)
        return await self.connector.update_many(collection, data, **kwargs)
    
    async def delete_one(self, collection: Optional[str], key: Dict[str, Any]) -> bool:
        """
        Delete a single record in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to delete the record from.
            key (Dict[str, Any]): The key to identify the record to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.delete_one, key)
        return await self.connector.delete_one(collection, key)
    
    async def delete_many(self, collection: Optional[str], keys: List[Dict[str, Any]]) -> int:
        """
        Delete multiple records in the specified collection or table.

        Args:
            collection (Optional[str]): The name of the collection or table to delete the records from.
            keys (List[Dict[str, Any]]): The list of keys to identify the records to delete.

        Returns:
            int: The number of records deleted.
        """
        if isinstance(self.connector, MemoryConnector):
            return await asyncio.to_thread(self.connector.delete_many, keys)
        return await self.connector.delete_many(collection, keys)
    
    async def drop(self, collection: Optional[str]) -> None:
        """
        Drop a collection, table, or flush the in-memory storage.
        """
        if isinstance(self.connector, MemoryConnector):
            await asyncio.to_thread(self.connector.drop)
        await self.connector.drop(collection)
    
    def help(self) -> None:
        """
        Display information about the CRUDManager and its methods.
        """
        help_text = """
        CRUDManager Help
        ----------------
        The CRUDManager class provides a unified interface for performing CRUD 
        (Create, Read, Update, Delete) operations across various database connectors.

        Supported Database Types:
        - memory
        - mongodb
        - redis
        - sqlite

        Available Methods:
        ------------------
        1. connect()
           Connect to the database or storage.

        2. disconnect()
           Disconnect from the database or storage.

        3. create(self, collection: Optional[str], data: Dict[str, Any], **kwargs)
           Creates a new collection/table with the specified name and schema (in the case of tables)
           
        4. insert(collection: Optional[str], data: Dict[str, Any], **kwargs)
           Create a new record in the specified collection or table.

        5. insert_many(collection: Optional[str], data_list: List[Dict[str, Any]], **kwargs)
           Create multiple records in the specified collection or table.

        6. find_one(collection: Optional[str], key: Dict[str, Any])
           Find a single record in the specified collection or table.

        7. find_many(collection: Optional[str], filters: Dict[str, Any])
           Find multiple records in the specified collection or table.

        8. update_one(collection: Optional[str], key: Dict[str, Any], updates: Dict[str, Any], **kwargs)
           Update a single record in the specified collection or table.

        9. update_many(collection: Optional[str], updates: List[Dict[str, Any]], **kwargs)
           Update multiple records in the specified collection or table.

        10. delete_one(collection: Optional[str], key: Dict[str, Any])
           Delete a single record in the specified collection or table.

        11. delete_many(collection: Optional[str], keys: List[Dict[str, Any]])
           Delete multiple records in the specified collection or table.

        12. drop(collection: Optional[str])
           Drop a collection, table, or flush in-memory storage.

        Notes:
        -------
        - `collection` is optional for connectors like MemoryConnector.
        - Additional parameters like `ttl` can be passed using `**kwargs`.
        - Ensure the appropriate connector is configured and connected before invoking these methods.

        Usage Example:
        --------------
        crud = CRUDManager("mysql")
        await crud.connect()

        # Create a table or collection
        schema = {"id": generate_id, userName: userName, userPassword: userpassword}
        await crud.create("users", schema)
        
        # Insert a record
        await crud.insert_one("users", {"id": 1, "name": "John Doe"})

        # Find a record
        record = await crud.find_one("users", {"id": 1})
        print(record)

        # Disconnect
        await crud.disconnect()
        """
        print(help_text)