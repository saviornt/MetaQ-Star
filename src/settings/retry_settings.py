# ./settings/retry_settings.py

from pydantic import Field
from src.settings.base_settings import BaseConfig


class RetrySettings(BaseConfig):
    """
    Configuration settings for retry logic.

    Attributes:
        max_retries (int): Maximum number of retry attempts.
        backoff_min (float): Minimum wait time (in seconds) for retries.
        backoff_max (float): Maximum wait time (in seconds) for retries.
    """

    max_retries: int = Field(5, env="MAX_RETRIES", description="Maximum number of retry attempts.")
    backoff_min: float = Field(0.5, env="EXPONENTIAL_BACKOFF_MIN", description="Minimum wait time for retries in seconds.")
    backoff_max: float = Field(60, env="EXPONENTIAL_BACKOFF_MAX", description="Maximum wait time for retries in seconds.")
