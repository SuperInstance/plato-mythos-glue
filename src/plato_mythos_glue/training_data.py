"""Export tiles from the server as a Hugging-Face-style JSONL dataset (no torch, no hf)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _fetch_json(url: str, timeout: float = 30.0) -> Any:
    """Generic JSON GET helper using stdlib urllib."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_tiles(room: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull tile list from a room dict, normalising assorted field names."""
    if not isinstance(room, dict):
        return []
    tiles = room.get("tiles", [])
    if not isinstance(tiles, list):
        tiles = []
    return tiles


def export_tiles_as_hf(server_url: str, output_dir: str) -> str:
    """Fetch all rooms from *server_url*, flatten their tiles, and write JSONL to *output_dir*.

    Returns the path to the written JSONL file.
    """
    base = server_url.rstrip("/")
    rooms_url = base + "/rooms"

    try:
        data = _fetch_json(rooms_url)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error fetching rooms: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc

    rooms = data if isinstance(data, list) else data.get("rooms", []) if isinstance(data, dict) else []

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.jsonl")

    with open(out_path, "w", encoding="utf-8") as fh:
        for room in rooms:
            for tile in _extract_tiles(room):
                if not isinstance(tile, dict):
                    continue
                question = str(tile.get("question", tile.get("q", "")))
                answer = str(tile.get("answer", tile.get("a", "")))
                domain = str(tile.get("domain", tile.get("category", "general")))
                confidence = tile.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                record = {
                    "input": question,
                    "output": answer,
                    "domain": domain,
                    "confidence": round(float(confidence), 6),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path
