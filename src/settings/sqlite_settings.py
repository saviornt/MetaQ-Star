from pydantic import BaseModel, Field
from typing import Optional


class SQLiteSettings(BaseModel):
    """
    Configuration settings for SQLite database.
    """
    database: str = Field(default=":memory:", description="Path to SQLite database file or :memory: for in-memory database")
    echo: bool = Field(default=False, description="Echo SQL commands to stdout")
    timeout: float = Field(default=5.0, description="Timeout for database operations in seconds")
    check_same_thread: bool = Field(default=True, description="Check if operations are performed from the same thread that created the connection")
    isolation_level: Optional[str] = Field(default=None, description="Isolation level for database transactions")


# Create an instance of the settings
sqlite_settings = SQLiteSettings()
