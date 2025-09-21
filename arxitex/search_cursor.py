import json
import os
from datetime import datetime
from threading import Lock
from urllib.parse import quote_plus

from loguru import logger


class SearchCursorManager:
    """
    Manages persistent search cursors to avoid re-fetching the same query results.
    This class is thread-safe.
    """

    def __init__(self, output_dir: str):
        self.cursor_file_path = os.path.join(output_dir, "search_cursors.json")
        self._lock = Lock()
        self.cursors = self._load_cursors()
        logger.info(
            f"Search cursor manager initialized. Loaded {len(self.cursors)} cursors from '{self.cursor_file_path}'."
        )

    def _load_cursors(self) -> dict:
        """Loads the cursors from the JSON file."""
        with self._lock:
            if not os.path.exists(self.cursor_file_path):
                return {}
            try:
                with open(self.cursor_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(
                    f"Could not load cursors file '{self.cursor_file_path}', starting fresh: {e}"
                )
                return {}

    def _save(self):
        """Saves the current cursors to disk. Assumes lock is already held."""
        try:
            with open(self.cursor_file_path, "w", encoding="utf-8") as f:
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
            logger.info(
                f"No cursor found for query key '{key}'. Starting search from the beginning."
            )
            return search_query

        # ArXiv API format for date range is [YYYYMMDDHHMM TO YYYYMMDDHHMM]
        try:
            # The 'Z' indicates UTC, which fromisoformat handles in Python 3.11+
            # For broader compatibility, we can replace it.
            if cursor_date_str.endswith("Z"):
                cursor_date_str = cursor_date_str[:-1] + "+00:00"
            dt_object = datetime.fromisoformat(cursor_date_str)

            # Format for ArXiv API: YYYYMMDDHHMMSS
            # We use a date far in the past as the start of our range.
            arxiv_format_date = dt_object.strftime("%Y%m%d%H%M%S")
            date_filter = f" AND submittedDate:[20000101000000 TO {arxiv_format_date}]"

            modified_query = f"({search_query}){date_filter}"
            logger.info(
                f"Using date cursor '{dt_object.isoformat()}' for query. New query:{modified_query}"
            )
            return modified_query

        except ValueError as e:
            logger.error(
                f"Could not parse cursor date '{cursor_date_str}' for query '{key}'. Starting fresh. Error: {e}"
            )
            return search_query

    def update_cursor(self, search_query: str, entries: list, ns: dict):
        """
        Finds the oldest submission date from a list of entries and updates the cursor.
        The date stored is the one from the <published> tag, in ISO 8601 format.
        """
        key = self._get_query_key(search_query)

        timestamps = [
            elem.text
            for entry in entries
            if (elem := entry.find("atom:published", ns)) is not None and elem.text
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


class BackfillStateManager:
    """
    Manages the state of a monthly historical backfill.
    """

    def __init__(self, output_dir: str):
        self.state_file_path = os.path.join(output_dir, "backfill_state.json")
        self._lock = Lock()
        self.states = self._load_state()
        logger.info(
            f"Backfill state manager initialized. Loaded {len(self.states)} states."
        )

    def _load_state(self) -> dict:
        if not os.path.exists(self.state_file_path):
            return {}
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error("Could not parse backfill state file. Starting fresh.")
            return {}

    def _save_state(self):
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(self.states, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save backfill state: {e}")

    def _get_query_key(self, search_query: str) -> str:
        return quote_plus(search_query.lower().strip())

    def get_next_interval(self, search_query: str) -> tuple[int, int]:
        """Gets the next (year, month) to process."""
        key = self._get_query_key(search_query)

        # Quick check without a lock for performance.
        if key in self.states:
            state = self.states[key]
            return state["year"], state["month"]

        with self._lock:
            # Re-check inside the lock to prevent a race condition.
            if key not in self.states:
                now = datetime.now()
                self.states[key] = {"year": now.year, "month": now.month}
                self._save_state()

            state = self.states[key]
            return state["year"], state["month"]

    def complete_interval(self, search_query: str, year: int, month: int):
        """Marks the current month as complete and moves to the previous one."""
        key = self._get_query_key(search_query)
        logger.success(
            f"Completed processing for query '{key}' for {year}-{month:02d}."
        )

        with self._lock:
            current_year = self.states.get(key, {}).get("year")
            current_month = self.states.get(key, {}).get("month")

            if current_year == year and current_month == month:
                new_month = current_month - 1
                new_year = current_year
                if new_month == 0:
                    new_month = 12
                    new_year -= 1

                self.states[key] = {"year": new_year, "month": new_month}
                self._save_state()
                logger.info(
                    f"State updated. Next run will process {new_year}-{new_month:02d}."
                )
