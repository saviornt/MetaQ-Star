# ./db_crud/connectors/mongodb_connector.py

from src.connectors.base_connector import BaseConnector
from src.settings.mongodb_settings import mongodb_settings
from src.settings.retry_settings import RetrySettings
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import logging


# Initialize retry settings and logger
retry_settings = RetrySettings()
logger = logging.getLogger(__name__)


class MongoDBConnector(BaseConnector):
    """
    MongoDB database connector implementation.

    Attributes:
        client (Optional[AsyncIOMotorClient]): MongoDB client instance.
        db (Optional[Any]): MongoDB database instance.
    """

    def __init__(self):
        super().__init__()
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[Any] = None

    @retry(
        stop=stop_after_attempt(retry_settings.max_retries),
        wait=wait_exponential(min=retry_settings.backoff_min, max=retry_settings.backoff_max),
    )
    async def connect(self) -> None:
        """
        Establish a connection to the MongoDB database with retry mechanisms.
        """
        try:
            self.client = AsyncIOMotorClient(
                mongodb_settings.uri, maxPoolSize=mongodb_settings.pool_size
            )
            self.db = self.client[mongodb_settings.database]
            self.connected = True
            logger.info("Successfully connected to the MongoDB database.")
        except RetryError as re:
            logger.error(f"RetryError occurred during MongoDB connection: {re}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close the MongoDB client connection.
        """
        if self.client:
            self.client.close()
        self.connected = False
        logger.info("Disconnected from the MongoDB database.")

    async def create(self, collection_name: str, schema: Optional[Dict[Any, Any]]) -> None:
        """
        Create a new collection in the MongoDB database.

        Args:
            collection_name (str): The name of the collection to create.
            schema (Optional[Dict[Any, Any]]): Passing schema for unified CRUD statement.

        Raises:
            Exception: If the collection cannot be created.
        """
        try:
            db: AsyncIOMotorDatabase = self.db  # Explicit type for database
            existing_collections = await db.list_collection_names()  # Explicitly tied to AsyncIOMotorDatabase
            if collection_name in existing_collections:
                logger.info(f"Collection {collection_name} already exists.")
            else:
                await db.create_collection(collection_name)
                logger.info(f"Successfully created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    async def insert_one(self, collection: str, document: Dict[str, Any]) -> Any:
        """
        Insert a single document into the specified collection.

        Args:
            collection (str): The collection name.
            document (Dict[str, Any]): The document to insert.

        Returns:
            Any: The inserted document ID.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: InsertOneResult = await col.insert_one(document)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error inserting document into {collection}: {e}")
            raise

    async def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[Any]:
        """
        Insert multiple documents into the specified collection.

        Args:
            collection (str): The collection name.
            documents (List[Dict[str, Any]]): The documents to insert.

        Returns:
            List[Any]: List of inserted document IDs.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: InsertManyResult = await col.insert_many(documents)
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Error inserting documents into {collection}: {e}")
            raise

    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document that matches the query.

        Args:
            collection (str): The collection name.
            query (Dict[str, Any]): The query criteria.

        Returns:
            Optional[Dict[str, Any]]: The matching document, or None if not found.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: Optional[Dict[str, Any]] = await col.find_one(query)
            return result
        except Exception as e:
            logger.error(f"Error finding document in {collection}: {e}")
            raise

    async def find_many(self, collection: str, query: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find multiple documents that match the query.

        Args:
            collection (str): The collection name.
            query (Optional[Dict[str, Any]]): The query criteria.
            limit (Optional[int]): Maximum number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of matching documents.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            cursor = col.find(query or {})
            if limit:
                cursor = cursor.limit(limit)
            return await cursor.to_list(length=limit or 0)
        except Exception as e:
            logger.error(f"Error finding documents in {collection}: {e}")
            raise

    async def update_one(self, collection: str, query: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Update a single document that matches the query.

        Args:
            collection (str): The collection name.
            query (Dict[str, Any]): The query criteria.
            updates (Dict[str, Any]): The update operations.

        Returns:
            int: The number of documents updated.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: UpdateResult = await col.update_one(query, {"$set": updates})
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating document in {collection}: {e}")
            raise

    async def update_many(self, collection: str, query: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Update multiple documents that match the query.

        Args:
            collection (str): The collection name.
            query (Dict[str, Any]): The query criteria.
            updates (Dict[str, Any]): The update operations.

        Returns:
            int: The number of documents updated.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: UpdateResult = await col.update_many(query, {"$set": updates})
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating documents in {collection}: {e}")
            raise

    async def delete_one(self, collection: str, query: Dict[str, Any]) -> int:
        """
        Delete a single document that matches the query.

        Args:
            collection (str): The collection name.
            query (Dict[str, Any]): The query criteria.

        Returns:
            int: The number of documents deleted.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: DeleteResult = await col.delete_one(query)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting document in {collection}: {e}")
            raise

    async def delete_many(self, collection: str, query: Dict[str, Any]) -> int:
        """
        Delete multiple documents that match the query.

        Args:
            collection (str): The collection name.
            query (Dict[str, Any]): The query criteria.

        Returns:
            int: The number of documents deleted.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            result: DeleteResult = await col.delete_many(query)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents in {collection}: {e}")
            raise

    async def drop(self, collection: str) -> None:
        """
        Drop the specified collection.

        Args:
            collection (str): The collection name.
        """
        try:
            col: AsyncIOMotorCollection = self.db[collection]
            await col.drop()
            logger.info(f"Dropped collection: {collection}.")
        except Exception as e:
            logger.error(f"Error dropping collection {collection}: {e}")
            raise
