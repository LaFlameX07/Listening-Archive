"""
Spotify Web API client.
Handles OAuth (Authorization Code flow) + the read endpoints we care about.
Tokens persist in SQLite so re-auth isn't needed every run.
"""
from __future__ import annotations

import base64
import time
import urllib.parse
from typing import Any, Optional

import httpx

from backend.config import settings
from backend.database import Database


AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"


class SpotifyClient:
    def __init__(self, db: Database):
        self.db = db
        self._client_id = settings.SPOTIFY_CLIENT_ID
        self._client_secret = settings.SPOTIFY_CLIENT_SECRET
        self._redirect = settings.SPOTIFY_REDIRECT_URI
        self._scopes = settings.SPOTIFY_SCOPES

    # --- OAuth ---

    def build_authorize_url(self) -> str:
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "scope": self._scopes,
            "redirect_uri": self._redirect,
            "show_dialog": "true",
        }
        return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str) -> dict:
        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        async with httpx.AsyncClient() as c:
            r = await c.post(
                TOKEN_URL,
                data={"grant_type": "authorization_code", "code": code, "redirect_uri": self._redirect},
                headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            )
        r.raise_for_status()
        tok = r.json()
        expires_at = int(time.time()) + int(tok["expires_in"])
        self.db.save_auth(
            access_token=tok["access_token"],
            refresh_token=tok["refresh_token"],
            expires_at=expires_at,
        )
        return tok

    async def _refresh_if_needed(self) -> str:
        auth = self.db.get_auth()
        if not auth:
            raise RuntimeError("No Spotify auth — visit /api/auth/login first.")
        if auth["expires_at"] - 30 > int(time.time()):
            return auth["access_token"]

        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        async with httpx.AsyncClient() as c:
            r = await c.post(
                TOKEN_URL,
                data={"grant_type": "refresh_token", "refresh_token": auth["refresh_token"]},
                headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            )
        r.raise_for_status()
        tok = r.json()
        new_refresh = tok.get("refresh_token", auth["refresh_token"])
        expires_at = int(time.time()) + int(tok["expires_in"])
        self.db.save_auth(tok["access_token"], new_refresh, expires_at)
        return tok["access_token"]

    # --- API calls ---

    async def _get(self, path: str, **params: Any) -> dict:
        token = await self._refresh_if_needed()
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{API_BASE}{path}", params=params, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()

    async def get_top_artists(self, time_range: str = "long_term", limit: int = 10) -> list[dict]:
        data = await self._get("/me/top/artists", time_range=time_range, limit=limit)
        return [
            {
                "id": a["id"],
                "name": a["name"],
                "genres": a.get("genres", []),
                "popularity": a.get("popularity"),
                "image": (a.get("images") or [{}])[0].get("url"),
            }
            for a in data.get("items", [])
        ]

    async def get_top_tracks(self, time_range: str = "long_term", limit: int = 10) -> list[dict]:
        data = await self._get("/me/top/tracks", time_range=time_range, limit=limit)
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                "album": t.get("album", {}).get("name"),
                "duration_ms": t.get("duration_ms"),
                "release_date": t.get("album", {}).get("release_date"),
            }
            for t in data.get("items", [])
        ]

    async def get_recently_played(self, limit: int = 50) -> list[dict]:
        data = await self._get("/me/player/recently-played", limit=limit)
        plays = []
        for item in data.get("items", []):
            t = item["track"]
            play = {
                "track_id": t["id"],
                "track_name": t["name"],
                "artist_name": ", ".join(a["name"] for a in t.get("artists", [])),
                "artist_id": (t.get("artists") or [{}])[0].get("id", ""),
                "played_at": item["played_at"],
                "duration_ms": t.get("duration_ms", 0),
            }
            self.db.insert_play(**play)
            plays.append(play)
        return plays
