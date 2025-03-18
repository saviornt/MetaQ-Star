# ./settings/logging_settings.py

import os
from pydantic import Field
from src.settings.base_settings import BaseConfig
from dotenv import load_dotenv

load_dotenv()

if os.getenv("ENABLE_CLOUD") == "True":
    # import mongodb
    pass
else:
    # import sqlite3
    pass


class LoggingSettings(BaseConfig):
    """
    Configuration settings for logging.

    Attributes:
        level (str): The logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format (str): The format string for log messages.
        datefmt (str): The date format string for log messages.
    """

    level: str = Field("INFO", env="LOG_LEVEL", description="Logging level.")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Logging format string.",
    )
    datefmt: str = Field(
        "%Y-%m-%d %H:%M:%S",
        env="LOG_DATEFMT",
        description="Logging date format string.",
    )
