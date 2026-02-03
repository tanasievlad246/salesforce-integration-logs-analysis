"""Database schema and connection helpers for action metrics."""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent.parent / "logs_analysis.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS action_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    app_version TEXT,
    level TEXT,
    flow TEXT,
    name TEXT NOT NULL,
    duration TEXT,
    duration_ms INTEGER,
    handler TEXT,
    request_id TEXT,
    operation_id TEXT,
    client TEXT,
    operation TEXT,
    source_file TEXT,
    UNIQUE(timestamp, request_id, name)
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON action_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_operation_id ON action_metrics(operation_id);
CREATE INDEX IF NOT EXISTS idx_request_id ON action_metrics(request_id);
CREATE INDEX IF NOT EXISTS idx_client ON action_metrics(client);
CREATE INDEX IF NOT EXISTS idx_handler ON action_metrics(handler);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize the database with the schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
