"""Room loader: fetch rooms from a Plato/Mythos Glue server and build expert configs."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def load_rooms(server_url: str = "http://localhost:8847") -> list[dict[str, Any]]:
    """GET /rooms from *server_url* and return a list of room dicts."""
    url = server_url.rstrip("/") + "/rooms"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to load rooms: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc.reason}") from exc

    if isinstance(data, dict):
        rooms = data.get("rooms", [])
    elif isinstance(data, list):
        rooms = data
    else:
        rooms = []
    return rooms


def room_to_expert_config(room: dict[str, Any]) -> dict[str, Any]:
    """Convert a room's tiles into a lightweight expert-routing configuration."""
    tiles = room.get("tiles", []) if isinstance(room, dict) else []
    if not tiles:
        return {"room_id": room.get("id") if isinstance(room, dict) else None, "weights": []}

    domain_counts: dict[str, int] = {}
    confidences: dict[str, list[float]] = {}

    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        domain = tile.get("domain") or tile.get("category") or "general"
        confidence = tile.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        confidences.setdefault(domain, []).append(float(confidence))

    total = sum(domain_counts.values())
    weights = []
    for domain, count in domain_counts.items():
        avg_conf = sum(confidences[domain]) / len(confidences[domain])
        weight = (count / total) * avg_conf
        weights.append({
            "domain": domain,
            "weight": round(weight, 6),
            "count": count,
            "avg_confidence": round(avg_conf, 6),
        })

    weights.sort(key=lambda w: w["weight"], reverse=True)
    return {
        "room_id": room.get("id") if isinstance(room, dict) else None,
        "weights": weights,
    }
