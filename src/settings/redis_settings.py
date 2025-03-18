# ./settings/redis_settings.py

from pydantic import Field
from typing import Optional
from src.settings.base_settings import BaseConfig

class RedisSettings(BaseConfig):
    """
    Configuration settings for Redis.

    Attributes:
        host (str): Redis server hostname.
        port (int): Redis server port.
        password (Optional[str]): Redis server password.
        db (int): Redis database index.
        ttl (int): Default time-to-live for cache entries in seconds.
        pool_size (int): Number of connections in the pool.
    """

    host: str = Field("localhost", env="REDIS_HOST", description="Hostname of the Redis server.")
    port: int = Field(6379, env="REDIS_PORT", description="Port number for the Redis server.")
    use_resp3: bool = Field(default=True, env="REDIS_USE_RESP3", description="Use RESP 3 standard. Default value is True")
    username: Optional[str] = Field(None, env="REDIS_USERNAME", description="Username for Redis authentication.")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD", description="Password for Redis authentication.")
    db: int = Field(0, env="REDIS_DB", description="Redis database index.")
    default_ttl: int = Field(3600, env="REDIS_TTL", description="Default TTL for cache entries (in seconds).")
    max_connections: int = Field(default=100, description="Maximum number of connections to the pool for concurrency")
    pool_size: int = Field(10, env="REDIS_POOL_SIZE", description="Number of connections in the pool.")


# Instantiate Redis settings
redis_settings = RedisSettings()
