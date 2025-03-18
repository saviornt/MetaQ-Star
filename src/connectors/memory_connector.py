# ./connectors/memory_connector.py

from src.connectors.base_connector import BaseConnector
from src.settings.memory_settings import memory_settings
import time
from typing import Any, Dict, List, Optional
import logging

# Initialize logger
logger = logging.getLogger(__name__)


class MemoryConnector(BaseConnector):
    """
    In-memory storage connector using Python's dictionary.

    Attributes:
        _store (Dict[str, Any]): The in-memory key-value store.
        _expiry (Dict[str, float]): Tracks expiration times for keys.
    """

    def __init__(self):
        super().__init__()
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def _cleanup(self) -> None:
        """
        Remove expired keys from the store.
        """
        current_time = time.time()
        expired_keys = [key for key, expiry in self._expiry.items() if expiry < current_time]
        for key in expired_keys:
            del self._store[key]
            del self._expiry[key]
        if expired_keys:
            logger.info(f"Cleaned up expired keys: {expired_keys}")

    def connect(self) -> None:
        """
        Connect to the in-memory storage.
        """
        self.connected = True
        logger.info("Connected to in-memory storage.")

    def disconnect(self) -> None:
        """
        Disconnect from the in-memory storage.
        """
        self._store.clear()
        self._expiry.clear()
        self.connected = False
        logger.info("Disconnected from in-memory storage.")

    def insert_one(self, data: Dict[str, Any], **kwargs) -> None:
        """
        Insert a single key-value pair into memory.

        Args:
            data (Dict[str, Any]): The data to insert, as a dictionary with one key-value pair.
            **kwargs: Optional parameters, such as TTL.
        """
        if len(data) != 1:
            raise ValueError("MemoryConnector.insert_one expects a single key-value pair.")

        key, value = next(iter(data.items()))  # Extract the single key-value pair
        ttl = kwargs.get("ttl", memory_settings.default_ttl)  # Retrieve TTL if provided

        self._cleanup()
        self._store[key] = value
        self._expiry[key] = time.time() + ttl
        logger.info(f"Set key {key} with TTL {ttl}.")

    def insert_many(self, data_list: List[Dict[str, Any]], **kwargs) -> None:
        """
        Insert multiple key-value pairs into memory.

        Args:
            data_list (List[Dict[str, Any]]): A list of dictionaries with key-value pairs.
            **kwargs: Optional parameters, such as TTL.
        """
        if not data_list:
            logger.warning("No data provided for insert_many.")
            return

        self._cleanup()
        ttl = kwargs.get("ttl", memory_settings.default_ttl)  # Retrieve TTL if provided

        for data in data_list:
            if len(data) != 1:
                raise ValueError("Each item in data_list for insert_many must be a single key-value pair.")
            key, value = next(iter(data.items()))
            self._store[key] = value
            self._expiry[key] = time.time() + ttl

        logger.info(f"Inserted {len(data_list)} key-value pairs with TTL {ttl}.")

    def find_one(self, key: str) -> Optional[Any]:
        """
        Retrieve a single value by key.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The value if the key exists and has not expired, or None.
        """
        self._cleanup()
        if key in self._store:
            return self._store[key]
        logger.warning(f"Key {key} not found or expired.")
        return None

    def find_many(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """
        Retrieve multiple values by their keys.

        Args:
            keys (List[str]): The keys to retrieve.

        Returns:
            Dict[str, Optional[Any]]: A dictionary of key-value pairs.
        """
        self._cleanup()
        results = {key: self._store.get(key) for key in keys}
        logger.info(f"Retrieved {len(results)} keys from memory.")
        return results

    def update_one(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """
        Update a single key-value pair in memory.

        Args:
            key (str): The key to update.
            data (Dict[str, Any]): The data to update as a dictionary with one key-value pair.
            **kwargs: Optional parameters, such as TTL.

        Returns:
            bool: True if the key was updated, False otherwise.
        """
        if len(data) != 1:
            raise ValueError("MemoryConnector.update_one expects a single key-value pair in 'data'.")

        self._cleanup()
        if key in self._store:
            update_key, update_value = next(iter(data.items()))
            if key != update_key:
                raise ValueError("The key in 'data' must match the provided 'key' argument.")

            self._store[key] = update_value
            ttl = kwargs.get("ttl", memory_settings.default_ttl)
            self._expiry[key] = time.time() + ttl
            logger.info(f"Updated key {key} with new TTL {ttl}.")
            return True

        logger.warning(f"Key {key} not found for update.")
        return False

    def update_many(self, updates: List[Dict[str, Dict[str, Any]]], **kwargs) -> int:
        """
        Update multiple key-value pairs in memory.

        Args:
            updates (List[Dict[str, Dict[str, Any]]]): A list of updates, each containing:
                - "key": The key to identify the record.
                - "data": The updated values.

        Returns:
            int: The number of keys successfully updated.
        """
        if not updates:
            logger.warning("No updates provided for update_many.")
            return 0

        self._cleanup()

        updated_count = 0

        for update in updates:
            key = update.get("key")
            data = update.get("data")

            if not key or not data:
                logger.warning("Each update must include 'key' and 'data'. Skipping...")
                continue

            if key in self._store:
                self._store[key] = data
                ttl = kwargs.get("ttl", memory_settings.default_ttl)
                self._expiry[key] = time.time() + ttl
                updated_count += 1
                logger.info(f"Updated key {key} with new TTL {ttl}.")
            else:
                logger.warning(f"Key {key} not found. Skipping...")

        logger.info(f"Successfully updated {updated_count} keys in memory.")
        return updated_count

    def delete_one(self, key: str) -> bool:
        """
        Delete a single key from memory.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        self._cleanup()
        if key in self._store:
            del self._store[key]
            self._expiry.pop(key, None)
            logger.info(f"Deleted key {key}.")
            return True
        logger.warning(f"Key {key} not found for deletion.")
        return False

    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys from memory.

        Args:
            keys (List[str]): The keys to delete.

        Returns:
            int: The number of keys successfully deleted.
        """
        self._cleanup()
        delete_count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                self._expiry.pop(key, None)
                delete_count += 1
        logger.info(f"Deleted {delete_count} keys from memory.")
        return delete_count

    def drop(self) -> None:
        """
        Drop all keys from memory.
        """
        self._store.clear()
        self._expiry.clear()
        logger.info("All keys dropped from memory.")
