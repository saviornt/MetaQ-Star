# ./connectors/sqlite_connector.py

from src.connectors.base_connector import BaseConnector
from src.settings.sqlite_settings import sqlite_settings
import aiosqlite
from typing import Any, Dict, List, Optional
import logging

# Initialize logger
logger = logging.getLogger(__name__)


class SQLiteConnector(BaseConnector):
    """
    SQLite connector implementation.

    Attributes:
        connection (Optional[aiosqlite.Connection]): The active SQLite connection.
    """

    def __init__(self):
        super().__init__()
        self.connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """
        Establish a connection to the SQLite database.
        """
        try:
            self.connection = await aiosqlite.connect(sqlite_settings.database)
            self.connected = True
            logger.info(f"Connected to SQLite database: {sqlite_settings.database}.")
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close the SQLite connection.
        """
        if self.connection:
            await self.connection.close()
            self.connection = None
            self.connected = False
            logger.info("Disconnected from SQLite database.")

    async def create(self, table: str, schema: str) -> None:
        """
        Create a table in the SQLite database.

        Args:
            table (str): The name of the table to create.
            schema (str): The SQL schema for the table.
        """
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
                await self.connection.commit()
                logger.info(f"Table {table} created successfully.")
        except Exception as e:
            logger.error(f"Error creating table {table}: {e}")
            raise

    async def insert_one(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a single row into a table.

        Args:
            table (str): The table name.
            data (Dict[str, Any]): The data to insert.

        Returns:
            int: The last inserted row ID.
        """
        keys = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data.values())
        values = tuple(data.values())

        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"INSERT INTO {table} ({keys}) VALUES ({placeholders})", values)
                await self.connection.commit()
                logger.info(f"Inserted row into {table}.")
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting row into table {table}: {e}")
            raise

    async def insert_many(self, table: str, rows: List[Dict[str, Any]]) -> None:
        """
        Insert multiple rows into a table.

        Args:
            table (str): The table name.
            rows (List[Dict[str, Any]]): The rows to insert.
        """
        if not rows:
            logger.warning("No rows provided for insert_many.")
            return

        keys = ", ".join(rows[0].keys())
        placeholders = ", ".join("?" for _ in rows[0].values())
        values = [tuple(row.values()) for row in rows]

        try:
            async with self.connection.cursor() as cursor:
                await cursor.executemany(f"INSERT INTO {table} ({keys}) VALUES ({placeholders})", values)
                await self.connection.commit()
                logger.info(f"Inserted {len(rows)} rows into {table}.")
        except Exception as e:
            logger.error(f"Error inserting rows into table {table}: {e}")
            raise

    async def find_one(self, table: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single row from a table.

        Args:
            table (str): The table name.
            conditions (Dict[str, Any]): Conditions to filter the row.

        Returns:
            Optional[Dict[str, Any]]: The retrieved row or None.
        """
        where_clause = " AND ".join(f"{key} = ?" for key in conditions.keys())
        values = tuple(conditions.values())

        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"SELECT * FROM {table} WHERE {where_clause}", values)
                result = await cursor.fetchone()
                if result:
                    return dict(result)
                return None
        except Exception as e:
            logger.error(f"Error retrieving row from table {table}: {e}")
            raise

    async def find_many(self, table: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple rows from a table.

        Args:
            table (str): The table name.
            conditions (Dict[str, Any]): Conditions to filter the rows.

        Returns:
            List[Dict[str, Any]]: A list of retrieved rows.
        """
        where_clause = " AND ".join(f"{key} = ?" for key in conditions.keys())
        values = tuple(conditions.values())

        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"SELECT * FROM {table} WHERE {where_clause}", values)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving rows from table {table}: {e}")
            raise

    async def update_one(self, table: str, conditions: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Update a single row in a table.

        Args:
            table (str): The table name.
            conditions (Dict[str, Any]): Conditions to filter the row.
            updates (Dict[str, Any]): The updates to apply.
        """
        set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
        where_clause = " AND ".join(f"{key} = ?" for key in conditions.keys())
        values = tuple(updates.values()) + tuple(conditions.values())

        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"UPDATE {table} SET {set_clause} WHERE {where_clause}", values)
                await self.connection.commit()
                logger.info(f"Updated row in table {table}.")
        except Exception as e:
            logger.error(f"Error updating row in table {table}: {e}")
            raise

    async def delete_one(self, table: str, conditions: Dict[str, Any]) -> None:
        """
        Delete a single row from a table.

        Args:
            table (str): The table name.
            conditions (Dict[str, Any]): Conditions to identify the row.
        """
        where_clause = " AND ".join(f"{key} = ?" for key in conditions.keys())
        values = tuple(conditions.values())

        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"DELETE FROM {table} WHERE {where_clause}", values)
                await self.connection.commit()
                logger.info(f"Deleted row from table {table}.")
        except Exception as e:
            logger.error(f"Error deleting row from table {table}: {e}")
            raise

    async def drop(self, table: str) -> None:
        """
        Drop a table from the SQLite database.

        Args:
            table (str): The table name.
        """
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"DROP TABLE IF EXISTS {table}")
                await self.connection.commit()
                logger.info(f"Dropped table {table}.")
        except Exception as e:
            logger.error(f"Error dropping table {table}: {e}")
            raise
