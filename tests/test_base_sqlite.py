import sqlite3

from arxitex.indices.base_sqlite import BaseSQLiteIndex


class DummyIndex(BaseSQLiteIndex):
    def _create_table(self):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS dummy_index (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.commit()
        conn.close()

    def insert(self, key: str, value: dict):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO dummy_index (key, value) VALUES (?, ?)",
            (key, self._serialize(value)),
        )
        conn.commit()
        conn.close()

    def get(self, key: str):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM dummy_index WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row:
            return self._deserialize(row["value"])
        return None


def test_insert_and_get_serialization(tmp_path):
    db_file = tmp_path / "test.db"
    idx = DummyIndex(str(db_file))

    sample = {"a": 1, "b": "text"}
    idx.insert("k1", sample)

    fetched = idx.get("k1")
    assert fetched == sample

    # Ensure raw sqlite file exists and contains JSON text
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT value FROM dummy_index WHERE key = 'k1'")
    raw = cur.fetchone()[0]
    conn.close()
    assert '"a": 1' in raw
    assert '"b": "text"' in raw
