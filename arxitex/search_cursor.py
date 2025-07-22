# arxitex/search_cursor.py

import os
import json
from threading import Lock
from loguru import logger
from urllib.parse import quote_plus
from datetime import datetime

class SearchCursorManager:
    """
    Manages persistent search cursors to avoid re-fetching the same query results.
    This class is thread-safe.
    """

    def __init__(self, output_dir: str):
        self.cursor_file_path = os.path.join(output_dir, "search_cursors.json")
        self._lock = Lock()
        self.cursors = self._load_cursors()
        logger.info(f"Search cursor manager initialized. Loaded {len(self.cursors)} cursors from '{self.cursor_file_path}'.")

    def _load_cursors(self) -> dict:
        """Loads the cursors from the JSON file."""
        with self._lock:
            if not os.path.exists(self.cursor_file_path):
                return {}
            try:
                with open(self.cursor_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not load cursors file '{self.cursor_file_path}', starting fresh: {e}")
                return {}

    def _save(self):
        """Saves the current cursors to disk. Assumes lock is already held."""
        try:
            with open(self.cursor_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.cursors, f, indent=2)
        except IOError as e:
            logger.error(f"CRITICAL: Could not save search cursors to disk: {e}")

    def _get_query_key(self, search_query: str) -> str:
        """Creates a stable, filename-safe key from a search query."""
        return quote_plus(search_query.lower().strip())

    def get_query_with_cursor(self, search_query: str) -> str:
        """
        Returns the ArXiv search query modified with the last known date cursor.
        If no cursor exists for the query, returns the original query.
        """
        key = self._get_query_key(search_query)
        cursor_date_str = self.cursors.get(key)

        if not cursor_date_str:
            logger.info(f"No cursor found for query key '{key}'. Starting search from the beginning.")
            return search_query

        # ArXiv API format for date range is [YYYYMMDDHHMM TO YYYYMMDDHHMM]
        try:
            # The 'Z' indicates UTC, which fromisoformat handles in Python 3.11+
            # For broader compatibility, we can replace it.
            if cursor_date_str.endswith('Z'):
                 cursor_date_str = cursor_date_str[:-1] + '+00:00'
            dt_object = datetime.fromisoformat(cursor_date_str)
            
            # Format for ArXiv API: YYYYMMDDHHMMSS
            # We use a date far in the past as the start of our range.
            arxiv_format_date = dt_object.strftime('%Y%m%d%H%M%S')
            date_filter = f" AND submittedDate:[20000101000000 TO {arxiv_format_date}]"
            
            modified_query = f"({search_query}){date_filter}"
            logger.info(f"Using date cursor '{dt_object.isoformat()}' for query. New query fragment: ...{date_filter}")
            return modified_query

        except ValueError as e:
            logger.error(f"Could not parse cursor date '{cursor_date_str}' for query '{key}'. Starting fresh. Error: {e}")
            return search_query


    def update_cursor(self, search_query: str, entries: list, ns: dict):
        """
        Finds the oldest submission date from a list of entries and updates the cursor.
        The date stored is the one from the <published> tag, in ISO 8601 format.
        """
        key = self._get_query_key(search_query)

        timestamps = [
            elem.text for entry in entries
            if (elem := entry.find('atom:published', ns)) is not None and elem.text
        ]

        if not timestamps:
            return

        oldest_in_batch = min(timestamps)

        current_cursor = self.cursors.get(key)
        # We only update if the new "oldest" date is older than our current cursor,
        # or if no cursor exists yet. This makes the process idempotent.
        if not current_cursor or oldest_in_batch < current_cursor:
            logger.info(f"Updating cursor for query '{key}' to '{oldest_in_batch}'.")
            with self._lock:
                self.cursors[key] = oldest_in_batch
                self._save()