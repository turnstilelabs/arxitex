import sqlite3
import abc
import json
from typing import Dict

class BaseSQLiteIndex(abc.ABC):
    """
    An abstract base class for indices backed by tables in a shared SQLite database.
    """
    def __init__(self, db_path: str):
        """
        Initializes the index, pointing to a shared SQLite database file.
        """
        self.db_path = db_path

        self._create_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a new database connection. Called for each transaction."""
        # The timeout helps prevent deadlocks if multiple threads contend for the db.
        conn = sqlite3.connect(self.db_path, timeout=10)
        # Use Row factory to get dict-like rows, which is more convenient.
        conn.row_factory = sqlite3.Row
        return conn

    @abc.abstractmethod
    def _create_table(self):
        """
        Abstract method to be implemented by subclasses.
        Should contain the 'CREATE TABLE IF NOT EXISTS' statement for the index.
        """
        pass

    def _serialize(self, value: Dict) -> str:
        """Serializes a Python dict into a JSON string for storage."""
        return json.dumps(value)

    def _deserialize(self, value: str) -> Dict:
        """Deserializes a JSON string from the db back into a Python dict."""
        return json.loads(value)