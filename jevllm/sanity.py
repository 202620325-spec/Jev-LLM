from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SanityIssue:
    code: str
    message: str
    evidence: str


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER_TOKEN = r"(?:\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")"
_COUNT_RE = re.compile(
    rf"\b(?P<n>{_NUMBER_TOKEN})\s+(?:(?:test|evaluation|data)\s+)?"
    r"(?P<kind>items?|examples?|samples?|cases?|trials?|observations?|predictions?)\b",
    re.IGNORECASE,
)
_METRIC_PATTERNS = (
    re.compile(
        r"\b(?P<label>accuracy|error(?:\s+rate)?|correct(?:ness)?|incorrect(?:ness)?)"
        r"\s*(?:=|:|is|of)?\s*(?P<pct>\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*"
        r"(?P<label>accuracy|error(?:\s+rate)?|correct(?:ness)?|incorrect(?:ness)?)\b",
        re.IGNORECASE,
    ),
)
_FRACTION_PERCENT_RE = re.compile(
    r"(?P<num>\d+)\s*/\s*(?P<den>\d+)"
    r"[^\n.;]{0,48}?(?:=|is|means|→|->)?\s*"
    r"(?P<pct>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_APPROX_RE = re.compile(r"(?:about|approx(?:imately)?|roughly|around|circa|~|≈)\s*$", re.IGNORECASE)


def _parse_count(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _is_approximate(text: str, start: int) -> bool:
    prefix = text[max(0, start - 18):start]
    return bool(_APPROX_RE.search(prefix))


def _excerpt(text: str, start: int, end: int, radius: int = 48) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return " ".join(text[lo:hi].split())


def hard_sanity_issues(text: str) -> list[SanityIssue]:
    """Return only deterministic contradictions that are safe to use as a STOP veto.

    This intentionally avoids semantic/model-based judging. The first validators
    cover discrete-count arithmetic because those failures can be proven directly
    from the text (for example, exact 75% accuracy over exactly 5 test items).
    """
    if not text or not text.strip():
        return [SanityIssue("EMPTY", "Candidate text is empty.", "")]

    issues: list[SanityIssue] = []

    # Explicit fraction/percentage arithmetic, e.g. "3/5 = 75%".
    for match in _FRACTION_PERCENT_RE.finditer(text):
        if _is_approximate(text, match.start("pct")):
            continue
        num = int(match.group("num"))
        den = int(match.group("den"))
        pct_text = match.group("pct")
        if den <= 0:
            issues.append(
                SanityIssue(
                    "ZERO_DENOMINATOR",
                    f"Fraction {num}/{den} has a zero denominator.",
                    _excerpt(text, match.start(), match.end()),
                )
            )
            continue
        expected = 100.0 * num / den
        stated = float(pct_text)
        tolerance = 0.051 if "." in pct_text else 0.000001
        if abs(expected - stated) > tolerance:
            issues.append(
                SanityIssue(
                    "FRACTION_PERCENT_MISMATCH",
                    f"{num}/{den} is {expected:.6g}%, not {stated:g}%.",
                    _excerpt(text, match.start(), match.end()),
                )
            )

    # If the candidate declares one unambiguous discrete evaluation-set size,
    # exact integer accuracy/error percentages must correspond to an integer
    # number of outcomes. Decimal percentages may simply be rounded, so they are
    # intentionally not vetoed here.
    counts = []
    for match in _COUNT_RE.finditer(text):
        n = _parse_count(match.group("n"))
        if n is not None and n > 0:
            counts.append((n, match))

    unique_counts = sorted({n for n, _ in counts})
    if len(unique_counts) == 1:
        n = unique_counts[0]
        for pattern in _METRIC_PATTERNS:
            for match in pattern.finditer(text):
                pct_text = match.group("pct")
                if "." in pct_text or _is_approximate(text, match.start("pct")):
                    continue
                pct = float(pct_text)
                if pct < 0.0 or pct > 100.0:
                    issues.append(
                        SanityIssue(
                            "PERCENT_OUT_OF_RANGE",
                            f"Percentage {pct:g}% is outside 0..100%.",
                            _excerpt(text, match.start(), match.end()),
                        )
                    )
                    continue
                discrete = n * pct / 100.0
                if abs(discrete - round(discrete)) > 1e-9:
                    label = " ".join(match.group("label").lower().split())
                    issues.append(
                        SanityIssue(
                            "IMPOSSIBLE_DISCRETE_PERCENT",
                            f"Exact {label} {pct:g}% is impossible over {n} discrete items.",
                            _excerpt(text, match.start(), match.end()),
                        )
                    )

    # Deduplicate repeated regex views of the same contradiction.
    deduped: list[SanityIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (issue.code, issue.message)
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
    return deduped
