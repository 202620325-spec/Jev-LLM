from __future__ import annotations

import re
from typing import Any


_SIMPLE_DEFINITION_PATTERNS = (
    r"\bwhat\s+is\b",
    r"\bwhat(?:'s| is)\s+.+\bmean\b",
    r"\bdefine\b",
    r"\bmeaning\s+of\b",
    r"(?:뭐냐|뭐야|뭐임|뭔데|무슨\s*뜻|뜻이\s*뭐|의미가\s*뭐|뭔\s*뜻)",
)

_REASONING_MARKERS = (
    r"\bprove\b",
    r"\bproof\b",
    r"\bderive\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bcounterexample\b",
    r"\bminimality\b",
    r"\brank\b",
    r"\bparity\b",
    r"\binvariant\b",
    r"\bbfs\b",
    r"\bwhy\b",
    r"\bhow\s+many\b",
    r"(?:증명|계산|유도|반례|최소성|랭크|불변량|패리티|경우의\s*수|가능하면\s*방법|불가능하면)",
)


def classify_query_mode(user_text: str, plan: dict[str, Any] | None = None) -> str:
    """Cheap deterministic routing before expensive search.

    This intentionally detects only high-confidence simple-definition requests.
    Ambiguous cases fall back to normal search rather than being over-compressed.
    """
    text = " ".join((user_text or "").strip().split())
    lowered = text.casefold()
    plan = plan or {}
    joined_plan = " ".join(
        [
            str(plan.get("intent") or ""),
            str(plan.get("answer_shape") or ""),
            " ".join(str(x) for x in (plan.get("route") or []) if x is not None),
        ]
    ).casefold()
    combined = lowered + " " + joined_plan

    if any(re.search(pattern, combined, flags=re.I) for pattern in _REASONING_MARKERS):
        return "reasoning"

    simple_hit = any(re.search(pattern, lowered, flags=re.I) for pattern in _SIMPLE_DEFINITION_PATTERNS)
    if simple_hit and len(text) <= 180 and text.count("?") <= 1 and "\n" not in text:
        return "simple_definition"

    return "normal"


def needs_claim_audit(user_text: str, plan: dict[str, Any] | None = None) -> bool:
    text = " ".join((user_text or "").strip().split())
    plan = plan or {}
    combined = " ".join(
        [
            text,
            str(plan.get("intent") or ""),
            str(plan.get("answer_shape") or ""),
            " ".join(str(x) for x in (plan.get("route") or []) if x is not None),
            " ".join(str(x) for x in (plan.get("verification_needs") or []) if x is not None),
        ]
    )
    return any(re.search(pattern, combined, flags=re.I) for pattern in _REASONING_MARKERS)
