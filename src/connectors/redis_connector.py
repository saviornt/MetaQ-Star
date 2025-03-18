# ./connectors/redis_connector.py

from src.connectors.base_connector import BaseConnector
from src.settings.redis_settings import redis_settings
from src.settings.retry_settings import RetrySettings
from redis.asyncio import Redis
from typing import Any, Optional, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import logging

# Initialize retry settings and logger
retry_settings = RetrySettings()
logger = logging.getLogger(__name__)


class RedisConnector(BaseConnector):
    """
    Redis connector implementation.

    Attributes:
        client (Optional[Redis]): Redis client instance.
    """

    def __init__(self):
        super().__init__()
        self.client: Optional[Redis] = None

    @retry(
        stop=stop_after_attempt(retry_settings.max_retries),
        wait=wait_exponential(min=retry_settings.backoff_min, max=retry_settings.backoff_max),
    )
    async def connect(self) -> None:
        """
        Establish a connection to the Redis database with retry mechanisms.
        """
        try:
            self.client = Redis(
                host=redis_settings.host,
                port=redis_settings.port,
                password=redis_settings.password,
                db=redis_settings.db,
                max_connections=redis_settings.pool_size,
                decode_responses=True,  # Ensures responses are strings
            )
            self.connected = True
            logger.info("Successfully connected to the Redis database.")
        except RetryError as re:
            logger.error(f"RetryError occurred during Redis connection: {re}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close the Redis client connection.
        """
        if self.client:
            await self.client.close()
        self.connected = False
        logger.info("Disconnected from the Redis database.")

    async def insert_one(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a single key-value pair in Redis with an optional TTL.

        Args:
            key (str): The key.
            value (Any): The value to store.
            ttl (Optional[int]): Time-to-live in seconds.
        """
        try:
            ttl = ttl or redis_settings.ttl
            await self.client.set(key, value, ex=ttl)
            logger.info(f"Set key: {key} with TTL: {ttl}")
        except RetryError as re:
            logger.error(f"RetryError occurred during set_one for key {key}: {re}")
            raise
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            raise

    async def find_one(self, key: str) -> Optional[Any]:
        """
        Get the value of a single key from Redis.

        Args:
            key (str): The key.

        Returns:
            Optional[Any]: The value, or None if the key does not exist.
        """
        try:
            return await self.client.get(key)
        except RetryError as re:
            logger.error(f"RetryError occurred during get_one for key {key}: {re}")
            raise
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            raise

    async def insert_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Set multiple key-value pairs in Redis.

        Args:
            data (Dict[str, Any]): A dictionary of key-value pairs to set.
            ttl (Optional[int]): Time-to-live in seconds (applies to all keys).
        """
        try:
            await self.client.mset(data)
            logger.info(f"Set multiple keys: {', '.join(data.keys())}")
            if ttl:
                # Set TTL for each key
                for key in data.keys():
                    await self.client.expire(key, ttl)
        except RetryError as re:
            logger.error(f"RetryError occurred during set_many: {re}")
            raise
        except Exception as e:
            logger.error(f"Error setting multiple keys: {e}")
            raise

    async def find_many(self, keys: List[str]) -> List[Optional[Any]]:
        """
        Get the values of multiple keys from Redis.

        Args:
            keys (List[str]): A list of keys to retrieve.

        Returns:
            List[Optional[Any]]: A list of values corresponding to the keys.
        """
        try:
            return await self.client.mget(keys)
        except RetryError as re:
            logger.error(f"RetryError occurred during get_many for keys {keys}: {re}")
            raise
        except Exception as e:
            logger.error(f"Error getting multiple keys: {e}")
            raise

    async def delete_one(self, key: str) -> int:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete.

        Returns:
            int: The number of keys deleted.
        """
        try:
            return await self.client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            raise

    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys from Redis using a pipeline for batch deletion.

        Args:
            keys (List[str]): A list of keys to delete.

        Returns:
            int: The number of keys successfully deleted.
        """
        try:
            if not keys:
                logger.warning("delete_keys called with an empty list of keys.")
                return 0

            async with self.client.pipeline() as pipe:
                for key in keys:
                    pipe.delete(key)
                results = await pipe.execute()

            deleted_count = sum(1 for result in results if result)
            logger.info(f"Deleted {deleted_count} keys from Redis.")
            return deleted_count
        except RetryError as re:
            logger.error(f"RetryError occurred during delete_keys for keys {keys}: {re}")
            raise
        except Exception as e:
            logger.error(f"Error deleting keys {keys}: {e}")
            raise

    async def drop(self) -> None:
        """
        Flush the current Redis database.
        """
        try:
            await self.client.flushdb()
            logger.info("Flushed the Redis database.")
        except Exception as e:
            logger.error(f"Error flushing the Redis database: {e}")
            raise
