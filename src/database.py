from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .utils import DB_PATH, ensure_directories, relative_or_absolute


class VehicleLogDB:
    """SQLite storage for vehicle entry and exit events."""

    def __init__(self, db_path: str | Path = DB_PATH):
        ensure_directories()
        self.db_path = Path(db_path)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        """Create the vehicle_logs table when it does not exist."""
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_number TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    source_type TEXT NOT NULL,
                    image_path TEXT,
                    confidence_score REAL,
                    status TEXT NOT NULL CHECK(status IN ('inside', 'exited'))
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_logs_number ON vehicle_logs(vehicle_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_logs_status ON vehicle_logs(status)")

    def get_active_log(self, vehicle_number: str) -> sqlite3.Row | None:
        """Return the active inside record for a vehicle, if any."""
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM vehicle_logs
                WHERE vehicle_number = ? AND status = 'inside'
                ORDER BY entry_time DESC
                LIMIT 1
                """,
                (vehicle_number,),
            ).fetchone()

    def recently_logged(self, vehicle_number: str, duplicate_window_seconds: int) -> bool:
        """Prevent repeated detections from creating noisy logs."""
        cutoff = datetime.now() - timedelta(seconds=duplicate_window_seconds)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT entry_time, exit_time FROM vehicle_logs
                WHERE vehicle_number = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (vehicle_number,),
            ).fetchone()
        if not row:
            return False
        last_time = row["exit_time"] or row["entry_time"]
        try:
            return datetime.fromisoformat(last_time) > cutoff
        except ValueError:
            return False

    def record_detection(
        self,
        vehicle_number: str,
        source_type: str,
        image_path: str | Path | None,
        confidence_score: float,
        cooldown_seconds: int = 60,
        duplicate_window_seconds: int = 10,
    ) -> tuple[str, int | None]:
        """
        Apply entry/exit logic.

        New plate: create an inside entry. Existing inside plate after cooldown: mark exited.
        Detections inside the duplicate window are ignored.
        """
        vehicle_number = vehicle_number.strip().upper()
        now = datetime.now()
        image_value = relative_or_absolute(image_path)

        active = self.get_active_log(vehicle_number)
        if active:
            entry_time = datetime.fromisoformat(active["entry_time"])
            if now - entry_time < timedelta(seconds=cooldown_seconds):
                return "duplicate", active["id"]
            with self.connect() as conn:
                conn.execute(
                    """
                    UPDATE vehicle_logs
                    SET exit_time = ?, status = 'exited', image_path = COALESCE(?, image_path),
                        confidence_score = MAX(COALESCE(confidence_score, 0), ?)
                    WHERE id = ?
                    """,
                    (now.isoformat(timespec="seconds"), image_value, confidence_score, active["id"]),
                )
            return "exit", active["id"]

        if self.recently_logged(vehicle_number, duplicate_window_seconds):
            return "duplicate", None

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vehicle_logs
                (vehicle_number, entry_time, exit_time, source_type, image_path, confidence_score, status)
                VALUES (?, ?, NULL, ?, ?, ?, 'inside')
                """,
                (
                    vehicle_number,
                    now.isoformat(timespec="seconds"),
                    source_type,
                    image_value,
                    confidence_score,
                ),
            )
        return "entry", int(cursor.lastrowid)

    def fetch_logs(self, search: str | None = None, status: str | None = None) -> pd.DataFrame:
        """Return logs as a DataFrame for dashboard display and CSV export."""
        query = "SELECT * FROM vehicle_logs"
        clauses: list[str] = []
        params: list[str] = []
        if search:
            clauses.append("vehicle_number LIKE ?")
            params.append(f"%{search.strip().upper()}%")
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC"
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def export_csv(self, search: str | None = None, status: str | None = None) -> bytes:
        """Export filtered logs as UTF-8 CSV bytes."""
        return self.fetch_logs(search=search, status=status).to_csv(index=False).encode("utf-8")
