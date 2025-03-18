# ./settings/mongodb_settings.py

from pydantic import Field
from src.settings.base_settings import BaseConfig


class MongoDBSettings(BaseConfig):
    """
    Configuration settings for MongoDB.

    Attributes:
        uri (str): MongoDB connection URI.
        database (str): MongoDB database name.
        pool_size (int): Connection pool size.
    """

    uri: str = Field("mongodb://localhost:27017", env="MONGODB_URI", description="MongoDB connection URI.")
    database: str = Field("testdb", env="MONGODB_DATABASE", description="MongoDB database name.")
    pool_size: int = Field(10, env="MONGODB_POOL_SIZE", description="Number of connections in the pool.")


# Instantiate MongoDB settings
mongodb_settings = MongoDBSettings()
