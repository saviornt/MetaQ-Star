# ./settings/memory_settings.py

from src.settings.base_settings import BaseConfig
from pydantic import Field


class MemorySettings(BaseConfig):
    """
    Settings for in-memory storage.

    Attributes:
        default_ttl (int): Default time-to-live for keys in memory, in seconds.
    """
    default_ttl: int = Field(3600, env="MEMORY_DEFAULT_TTL", description="Default time-to-live for in-memory keys, in seconds (1 hour).")
    cache_maxsize: int = Field(default=5000, description="Maximum size for in-memory cache")

memory_settings = MemorySettings()
