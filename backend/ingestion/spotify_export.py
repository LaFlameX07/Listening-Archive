"""
Spotify GDPR export ingester.

The free, no-Premium path to your own listening data.

How to get the export:
1. Go to https://www.spotify.com/account/privacy/
2. Scroll to "Download your data"
3. Request "Account data" (5-day wait) and/or "Extended streaming history" (30-day wait)
4. Spotify emails you a ZIP file with JSON inside

Relevant files in the ZIP:
- StreamingHistory0.json, StreamingHistory1.json, …  (last 12 months)
- Streaming_History_Audio_*.json                      (extended, years of data)
- YourLibrary.json                                    (saved tracks/albums)

Usage:
    uv run python -m backend.ingestion.spotify_export /path/to/extracted/folder

This populates SQLite (plays table) and memory_rag (embedded with time context).
After ingestion, the magazine page and chat work with your real data — no Web API needed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.config import settings
from backend.database import Database
from backend.rag.embeddings import Embedder
from backend.rag.memory_rag import MemoryRAG


def find_history_files(export_dir: Path) -> list[Path]:
    """Locate every StreamingHistory*.json (basic) or Streaming_History_Audio_*.json (extended)."""
    patterns = ["StreamingHistory*.json", "Streaming_History_Audio_*.json"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(export_dir.rglob(pat))
    return sorted(set(files))


def parse_play_record(rec: dict) -> dict | None:
    """Normalise the two slightly different record formats Spotify uses."""
    # Basic format (StreamingHistory*.json)
    if "trackName" in rec and "artistName" in rec:
        return {
            "track_id": f"export::{rec['artistName']}::{rec['trackName']}",
            "track_name": rec["trackName"],
            "artist_name": rec["artistName"],
            "artist_id": "",
            "played_at": rec.get("endTime", "") + ":00Z" if rec.get("endTime") else "",
            "duration_ms": int(rec.get("msPlayed", 0)),
        }
    # Extended format (Streaming_History_Audio_*.json)
    if "master_metadata_track_name" in rec and rec.get("master_metadata_track_name"):
        track = rec["master_metadata_track_name"]
        artist = rec.get("master_metadata_album_artist_name", "")
        return {
            "track_id": rec.get("spotify_track_uri", f"export::{artist}::{track}"),
            "track_name": track,
            "artist_name": artist,
            "artist_id": "",
            "played_at": rec.get("ts", ""),
            "duration_ms": int(rec.get("ms_played", 0)),
        }
    return None


def ingest(export_dir: Path, min_ms: int = 30_000) -> dict:
    """
    Walk the export, insert plays into SQLite, embed into memory_rag.
    `min_ms`: filter out skipped plays (< 30s by default).
    """
    files = find_history_files(export_dir)
    if not files:
        raise FileNotFoundError(
            f"No streaming history files found under {export_dir}. "
            "Expected StreamingHistory*.json or Streaming_History_Audio_*.json."
        )

    print(f"→ Found {len(files)} history file(s).")
    db = Database(settings.DB_PATH)
    db.init_schema()
    embedder = Embedder()
    memory_rag = MemoryRAG(embedder, settings.CHROMA_PATH, db)

    total_records = 0
    inserted = 0
    skipped_short = 0
    play_counts: Counter = Counter()

    for f in files:
        print(f"  · reading {f.name}")
        with open(f) as fp:
            records = json.load(fp)
        for rec in records:
            total_records += 1
            play = parse_play_record(rec)
            if not play:
                continue
            if play["duration_ms"] < min_ms:
                skipped_short += 1
                continue
            db.insert_play(**play)
            inserted += 1
            play_counts[(play["track_name"], play["artist_name"])] += 1

    print(f"\n→ Embedding {inserted} plays into memory_rag (this is the slow part)…")
    memory_rag.reindex_all_plays()

    # Quick summary stats
    top_5 = play_counts.most_common(5)
    print(f"\n✓ Done.")
    print(f"  records seen:    {total_records:>7}")
    print(f"  skipped (<{min_ms}ms): {skipped_short:>7}")
    print(f"  inserted:        {inserted:>7}")
    print(f"\nTop 5 tracks by play count:")
    for (track, artist), n in top_5:
        print(f"  {n:>4}× {track} — {artist}")
    return {"total": total_records, "inserted": inserted, "skipped": skipped_short, "top_5": top_5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a Spotify GDPR data export.")
    parser.add_argument("path", help="Path to the extracted Spotify export folder.")
    parser.add_argument("--min-ms", type=int, default=30_000, help="Skip plays shorter than this (ms). Default 30000.")
    args = parser.parse_args()
    ingest(Path(args.path), min_ms=args.min_ms)


def from_zip(zip_path: Path, min_ms: int = 30_000) -> dict:
    """
    Extract a Spotify GDPR ZIP into a temp folder and run ingest on it.
    Called by the /api/ingest/upload endpoint.
    """
    import tempfile
    import zipfile
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / "extracted"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        return ingest(extract_dir, min_ms=min_ms)
