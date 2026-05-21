"""Settings loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).parent.parent


class Settings:
    # --- Spotify ---
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    SPOTIFY_REDIRECT_URI: str = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:3000/api/callback")
    SPOTIFY_SCOPES: str = "user-top-read user-read-recently-played user-library-read"

    # --- Google AI Studio (Gemma / Gemini) ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma-3-27b-it")  # swap to "gemini-2.5-flash" if tool-calling falters

    # --- Optional Genius API for lyrics ingestion ---
    GENIUS_TOKEN: str = os.getenv("GENIUS_TOKEN", "")

    # --- Local paths ---
    DATA_DIR: Path = ROOT / "data"
    DB_PATH: Path = ROOT / "data" / "archive.db"
    CHROMA_PATH: Path = ROOT / "data" / "chroma"

    # --- Embedding model (local, CPU-friendly, ~80MB) ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True, parents=True)
settings.CHROMA_PATH.mkdir(exist_ok=True, parents=True)
