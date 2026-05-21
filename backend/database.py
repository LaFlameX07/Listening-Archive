"""
SQLite — the local database.
Stores Spotify tokens, listening events, top items, and user notes.
All data stays on this machine. The file lives at data/archive.db.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS spotify_auth (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    user_id TEXT
);

CREATE TABLE IF NOT EXISTS plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    track_name TEXT NOT NULL,
    artist_name TEXT NOT NULL,
    artist_id TEXT,
    played_at TEXT NOT NULL,
    duration_ms INTEGER,
    UNIQUE(track_id, played_at)
);
CREATE INDEX IF NOT EXISTS idx_plays_played_at ON plays(played_at);
CREATE INDEX IF NOT EXISTS idx_plays_track ON plays(track_id);

CREATE TABLE IF NOT EXISTS top_items (
    snapshot_date TEXT NOT NULL,
    kind TEXT NOT NULL,             -- 'artist' or 'track'
    time_range TEXT NOT NULL,        -- 'short_term' / 'medium_term' / 'long_term'
    rank INTEGER NOT NULL,
    spotify_id TEXT NOT NULL,
    name TEXT NOT NULL,
    extra TEXT,                      -- JSON: artist for tracks, genres for artists, etc.
    PRIMARY KEY (snapshot_date, kind, time_range, rank)
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True, parents=True)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- Auth ---

    def save_auth(self, access_token: str, refresh_token: str, expires_at: int, user_id: Optional[str] = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO spotify_auth (id, access_token, refresh_token, expires_at, user_id) VALUES (1, ?, ?, ?, ?)",
                (access_token, refresh_token, expires_at, user_id),
            )

    def get_auth(self) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM spotify_auth WHERE id = 1").fetchone()
            return dict(row) if row else None

    # --- Plays ---

    def insert_play(self, *, track_id: str, track_name: str, artist_name: str,
                    artist_id: str, played_at: str, duration_ms: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO plays (track_id, track_name, artist_name, artist_id, played_at, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (track_id, track_name, artist_name, artist_id, played_at, duration_ms),
            )

    def recent_plays(self, limit: int = 50) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM plays ORDER BY played_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    # --- Notes ---

    def add_note(self, track_id: str, text: str) -> int:
        with self.connect() as conn:
            cur = conn.execute("INSERT INTO notes (track_id, text) VALUES (?, ?)", (track_id, text))
            return cur.lastrowid

    def all_notes(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()]
