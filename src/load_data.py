"""Load CSV log files into SQLite database."""

import json
import re
from pathlib import Path

import pandas as pd

from schema import DEFAULT_DB_PATH, get_connection, init_db

DATA_DIR = Path(__file__).parent.parent / "data"


def get_sorted_csv_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Get CSV files sorted naturally: base file first, then (1), (2), etc."""
    files = list(data_dir.glob("logs-insights-results*.csv"))

    def sort_key(path: Path) -> tuple[int, int]:
        name = path.name
        # Base file without number
        if name == "logs-insights-results.csv":
            return (0, 0)
        # Extract number from parentheses
        match = re.search(r"\((\d+)\)", name)
        if match:
            return (1, int(match.group(1)))
        return (2, 0)

    return sorted(files, key=sort_key)


def extract_app_version(app_str: str) -> str | None:
    """Extract version from app string like 'auchan.salesforce-integration@1.9.11'."""
    if not app_str:
        return None
    match = re.search(r"@([\d.]+)", app_str)
    return match.group(1) if match else None


def parse_message(message: str) -> dict | None:
    """Parse JSON message and extract relevant fields."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return None

    # Extract nested data.data fields
    inner_data = data.get("data", {}).get("data", {})
    if not inner_data:
        return None

    return {
        "level": data.get("level"),
        "app_version": extract_app_version(data.get("app", "")),
        "flow": data.get("data", {}).get("flow"),
        "name": inner_data.get("name"),
        "duration": inner_data.get("duration"),
        "duration_ms": inner_data.get("durationMs"),
        "handler": inner_data.get("handler"),
        "request_id": inner_data.get("requestId"),
        "operation_id": inner_data.get("operationId"),
        "client": inner_data.get("client"),
        "operation": inner_data.get("operation"),
    }


def load_csv_file(csv_path: Path, conn) -> tuple[int, int]:
    """Load a single CSV file into the database.

    Returns:
        Tuple of (inserted_count, skipped_count)
    """
    df = pd.read_csv(csv_path)
    source_file = csv_path.name

    inserted = 0
    skipped = 0

    cursor = conn.cursor()

    for _, row in df.iterrows():
        timestamp = row.get("@timestamp")
        message = row.get("@message")

        parsed = parse_message(message)
        if not parsed or not parsed.get("name"):
            skipped += 1
            continue

        try:
            cursor.execute(
                """
                INSERT INTO action_metrics (
                    timestamp, app_version, level, flow, name, duration,
                    duration_ms, handler, request_id, operation_id, client,
                    operation, source_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    parsed["app_version"],
                    parsed["level"],
                    parsed["flow"],
                    parsed["name"],
                    parsed["duration"],
                    parsed["duration_ms"],
                    parsed["handler"],
                    parsed["request_id"],
                    parsed["operation_id"],
                    parsed["client"],
                    parsed["operation"],
                    source_file,
                ),
            )
            inserted += 1
        except Exception:
            # Duplicate or other error
            skipped += 1

    conn.commit()
    return inserted, skipped


def main():
    """Main entry point for data loading."""
    print("Initializing database...")
    init_db()

    csv_files = get_sorted_csv_files()
    print(f"Found {len(csv_files)} CSV files to process")

    conn = get_connection()
    total_inserted = 0
    total_skipped = 0

    for csv_path in csv_files:
        inserted, skipped = load_csv_file(csv_path, conn)
        total_inserted += inserted
        total_skipped += skipped
        print(f"  {csv_path.name}: {inserted} inserted, {skipped} skipped")

    conn.close()

    print(f"\nComplete!")
    print(f"  Total records inserted: {total_inserted}")
    print(f"  Total records skipped: {total_skipped}")
    print(f"  Database location: {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
