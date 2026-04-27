"""Tile encoder/decoder: word-level tokenisation and list-based 'tensors'."""

from __future__ import annotations

import math
from typing import Any

_VOCAB = {
    "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "<mask>": 4,
    "what": 5, "is": 6, "the": 7, "of": 8, "a": 9, "how": 10, "to": 11,
    "in": 12, "and": 13, "or": 14, "not": 15, "true": 16, "false": 17,
    "yes": 18, "no": 19, "why": 20, "when": 21, "where": 22, "who": 23,
    "can": 24, "do": 25, "does": 26, "are": 27, "be": 28, "being": 29,
    "become": 30, "one": 31, "many": 32, "all": 33, "none": 34, "some": 35,
    "form": 36, "idea": 37, "soul": 38, "body": 39, "mind": 40,
    "knowledge": 41, "virtue": 42, "justice": 43, "beauty": 44,
    "good": 45, "evil": 46, "love": 47, "wisdom": 48, "reality": 49,
    "truth": 50, "opinion": 51, "dialectic": 52, "myth": 53, "logos": 54,
    "ethics": 55, "politics": 56, "metaphysics": 57, "epistemology": 58,
    "cosmos": 59, "nature": 60, "cause": 61, "effect": 62, "unity": 63,
    "plurality": 64, "eternal": 65, "temporal": 66, "finite": 67,
    "infinite": 68, "absolute": 69, "relative": 70, "necessary": 71,
    "possible": 72, "actual": 73, "potential": 74, "substance": 75,
    "attribute": 76, "essence": 77, "existence": 78, "perception": 79,
    "reason": 80, "argument": 81, "premise": 82, "conclusion": 83,
    "syllogism": 84, "dialogue": 85, "rhetoric": 86, "poetry": 87,
    "art": 88, "science": 89, "mathematics": 90, "geometry": 91,
    "number": 92, "measure": 93, "proportion": 94, "harmony": 95,
    "music": 96, "motion": 97, "rest": 98,
}
_ID2TOKEN = {v: k for k, v in _VOCAB.items()}
_DOMAINS = [
    "metaphysics", "epistemology", "ethics", "politics", "aesthetics",
    "logic", "rhetoric", "poetics", "physics", "mathematics",
    "theology", "general",
]


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return [t for t in cleaned.split() if t]


def _text_to_ids(text: str, max_len: int = 32) -> list[int]:
    tokens = ["<bos>"] + _tokenize(text) + ["<eos>"]
    ids = [_VOCAB.get(t, _VOCAB["<unk>"]) for t in tokens]
    if len(ids) < max_len:
        ids += [_VOCAB["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]


def _ids_to_text(ids: list[int]) -> str:
    skip = {_VOCAB["<pad>"], _VOCAB["<bos>"], _VOCAB["<eos>"], _VOCAB["<mask>"]}
    return " ".join(_ID2TOKEN.get(i, "<unk>") for i in ids if i not in skip)


def encode_tile(tile_dict: dict[str, Any], max_len: int = 32) -> dict[str, Any]:
    """Encode a tile dict into token ids, domain id, and confidence vector."""
    t = tile_dict if isinstance(tile_dict, dict) else {}
    question = str(t.get("question", t.get("q", "")))
    answer = str(t.get("answer", t.get("a", "")))
    domain = str(t.get("domain", t.get("category", "general"))).lower()
    confidence = float(t.get("confidence", 0.5))
    domain_id = _DOMAINS.index(domain) if domain in _DOMAINS else _DOMAINS.index("general")
    return {
        "token_ids": {"question": _text_to_ids(question, max_len), "answer": _text_to_ids(answer, max_len)},
        "domain_id": domain_id,
        "confidence": [math.tanh(confidence)] * max_len,
    }


def decode_output(output_tensor: dict[str, Any]) -> dict[str, Any]:
    """Decode an output dict back to a human-readable tile dict."""
    o = output_tensor if isinstance(output_tensor, dict) else {}
    tok = o.get("token_ids", {})
    q_ids = tok.get("question", []) if isinstance(tok, dict) else []
    a_ids = tok.get("answer", []) if isinstance(tok, dict) else []
    did = o.get("domain_id", _DOMAINS.index("general"))
    if isinstance(did, (list, tuple)):
        did = int(did[0]) if did else _DOMAINS.index("general")
    domain = _DOMAINS[int(did)] if 0 <= int(did) < len(_DOMAINS) else "general"
    cvec = o.get("confidence", [0.5])
    if isinstance(cvec, (list, tuple)):
        avg = sum(float(v) for v in cvec) / max(len(cvec), 1)
    else:
        avg = float(cvec)
    # Inverse of tanh, clamped to avoid math domain errors.
    at = math.atanh(max(-0.99, min(0.99, avg)))
    confidence = max(0.0, min(1.0, (at + 1.0) / 2.0))
    return {
        "question": _ids_to_text(q_ids),
        "answer": _ids_to_text(a_ids),
        "domain": domain,
        "confidence": round(confidence, 6),
    }
