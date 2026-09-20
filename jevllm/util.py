from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Parse strict JSON first, then recover a single JSON object/array from fenced/noisy output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    starts = [(text.find("{"), "{"), (text.find("["), "[")]
    starts = [(idx, ch) for idx, ch in starts if idx >= 0]
    if not starts:
        raise ValueError("No JSON object/array found in model output")
    start, opener = min(starts, key=lambda x: x[0])
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("Incomplete JSON in model output")
    return json.loads(text[start : end + 1])


def recover_candidate_strings(text: str, limit: int) -> list[str]:
    """Best-effort recovery from strict JSON, loose JSON, object candidates, or labelled lines."""
    raw = (text or "").strip()
    if not raw:
        return []

    parsed: Any = None
    try:
        parsed = extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        parsed = None

    def item_text(item: Any) -> str | None:
        if isinstance(item, str):
            return item.strip() or None
        if isinstance(item, dict):
            for key in ("text", "content", "candidate", "continuation", "value"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    pool: list[Any] = []
    if isinstance(parsed, dict):
        for key in ("candidates", "options", "continuations", "choices"):
            value = parsed.get(key)
            if isinstance(value, list):
                pool.extend(value)
                break
            if isinstance(value, str):
                pool.append(value)
                break
        if not pool:
            maybe = item_text(parsed)
            if maybe:
                pool.append(maybe)
    elif isinstance(parsed, list):
        pool.extend(parsed)
    elif isinstance(parsed, str):
        pool.append(parsed)

    cleaned: list[str] = []
    for item in pool:
        value = item_text(item)
        if value and value not in cleaned:
            cleaned.append(value)
        if len(cleaned) >= limit:
            return cleaned

    if cleaned:
        return cleaned[:limit]

    # Recover common malformed JSON object snippets.
    for match in re.finditer(r'"(?:text|content|candidate|continuation)"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.I):
        try:
            value = json.loads('"' + match.group(1) + '"').strip()
        except Exception:
            value = match.group(1).strip()
        if value and value not in cleaned:
            cleaned.append(value)
        if len(cleaned) >= limit:
            return cleaned

    # Recover C0:, 1., -, * labelled candidates.
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        value = re.sub(r"^(?:C\d+\s*[:.)-]|\d+\s*[:.)-]|[-*•])\s*", "", line, flags=re.I).strip()
        if value and value != line or re.match(r"^(?:C\d+|\d+|[-*•])", line, flags=re.I):
            if value and value not in cleaned:
                cleaned.append(value)
        if len(cleaned) >= limit:
            return cleaned

    return cleaned[:limit]


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_probability_map(answer: dict[str, Any]) -> dict[str, float]:
    probs = answer.get("probabilities") or {}
    out: dict[str, float] = {}
    if isinstance(probs, dict):
        for key, value in probs.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def score_level(answer: dict[str, Any], level_count: int) -> int:
    """Use the most-probable Jev score level. Do not round expected score unless probabilities are absent."""
    probs = normalize_probability_map(answer)
    numeric: list[tuple[int, float]] = []
    for key, probability in probs.items():
        try:
            idx = int(float(key))
        except ValueError:
            continue
        if 0 <= idx < level_count:
            numeric.append((idx, probability))
    if numeric:
        return max(numeric, key=lambda x: x[1])[0]

    try:
        raw_score = float(answer.get("score", 0))
    except (TypeError, ValueError):
        raw_score = 0.0
    return clamp_int(round(raw_score), 0, level_count - 1)


def answer_confidence(answer: dict[str, Any]) -> float:
    try:
        return float(answer.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def score_expectation(answer: dict[str, Any], level_count: int = 6) -> float:
    """Return normalized expected Jev score in [0,1]. Falls back to raw score."""
    probs = normalize_probability_map(answer)
    weighted = 0.0
    total = 0.0
    for key, probability in probs.items():
        try:
            idx = int(float(key))
            p = max(0.0, float(probability))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < level_count:
            weighted += idx * p
            total += p
    if total > 0:
        return max(0.0, min(1.0, (weighted / total) / max(1, level_count - 1)))
    try:
        raw = float(answer.get("score", 0.0))
    except (TypeError, ValueError):
        raw = 0.0
    return max(0.0, min(1.0, raw / max(1, level_count - 1)))


def noul_probability(answer: dict[str, Any], default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(answer.get("noul", default))))
    except (TypeError, ValueError):
        return default
