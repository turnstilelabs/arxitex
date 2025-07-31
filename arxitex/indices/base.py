import os
import json
import abc
from threading import Lock
from typing import Dict, Any
from loguru import logger

class BaseIndex(abc.ABC):
    """
    An abstract base class for persistent, thread-safe, on-disk JSON indices.

    This class handles the common logic for loading, saving, and locking a
    JSON file that backs an in-memory dictionary.
    """
    def __init__(self, output_dir: str, filename: str):
        if not filename.endswith('.json'):
            raise ValueError("Index filename must end with .json")
            
        self.index_file_path = os.path.join(output_dir, filename)
        self._lock = Lock()
        self.data: Dict[str, Any] = self._load()

    def _get_default_data(self) -> Dict:
        return {}

    def _load(self) -> Dict[str, Any]:
        """Loads the index from the JSON file into self.data."""
        with self._lock:
            if not os.path.exists(self.index_file_path):
                return self._get_default_data()
            try:
                with open(self.index_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(
                    f"Could not load or parse index '{self.index_file_path}', "
                    f"starting with default data structure: {e}"
                )
                return self._get_default_data()

    def _save(self):
        """
        Saves the current self.data dictionary to disk.
        """
        try:
            sorted_data = dict(sorted(self.data.items()))
            with open(self.index_file_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, indent=2)
        except IOError as e:
            logger.error(f"CRITICAL: Could not save index to disk at '{self.index_file_path}': {e}")

    def __len__(self) -> int:
        with self._lock:
            return len(self.data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self.data