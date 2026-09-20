from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HORIZONS = [
    ("character", "1-3 characters; exact spelling/symbol/local continuation"),
    ("subword", "a tiny token-like fragment, roughly 2-8 characters"),
    ("word", "one lexical word or compact code/token unit"),
    ("phrase", "a short phrase; one small semantic unit"),
    ("clause", "one clause; a medium semantic continuation"),
    ("sentence", "one complete sentence; a high-level semantic continuation"),
]

COGNITIVE_MODES = [
    "predict",
    "options",
    "reason_low",
    "reason_medium",
    "reason_high",
    "reason_extreme",
]

MODE_RANK = {name: idx for idx, name in enumerate(COGNITIVE_MODES)}

MODE_TO_REASONING = {
    "predict": "low",
    "options": "low",
    "reason_low": "low",
    "reason_medium": "medium",
    "reason_high": "high",
    "reason_extreme": "high",
}

BREADTH_LEVELS = [1, 2, 3, 4, 6]

# How much context should be considered as one decision field.
REASONING_SCOPES = [
    ("local", "only the immediate next fragment and local correctness"),
    ("sentence", "the current sentence or semantic move"),
    ("subproblem", "the current subproblem/section and its constraints"),
    ("whole_answer", "the whole answer and all requirements in the current user request"),
    ("conversation", "the whole answer plus relevant recent conversation context"),
    ("multi_angle", "global answer with cross-checking across multiple plausible interpretations/angles"),
]

# Final answer surface budget. These are target ceilings, not promises to fill every token.
RESPONSE_LENGTHS = [
    # name, desired surface tokens, API ceiling (includes reasoning headroom), description
    ("micro", 80, 512, "one or a few compact sentences"),
    ("short", 220, 768, "a concise answer with only the necessary support"),
    ("medium", 550, 1400, "a normal complete answer with useful explanation"),
    ("long", 1200, 2600, "a detailed answer covering multiple parts"),
    ("extended", 2400, 4800, "an extensive answer for broad or deeply technical requests"),
]


@dataclass
class ControlProfile:
    horizon_index: int
    horizon: str
    cognitive_mode: str
    breadth: int
    reasoning_scope_index: int = 0
    reasoning_scope: str = "local"
    response_length_index: int = 2
    response_length: str = "medium"
    ready_to_finalize: float = 0.0
    confidence: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveProfile:
    horizon_index: int
    horizon: str
    cognitive_mode: str
    breadth: int
    reasoning_scope_index: int
    reasoning_scope: str
    response_length_index: int
    response_length: str
    response_target_tokens: int
    response_max_tokens: int
    reasoning_effort: str
    deliberation_budget: float
    uncertainty: float
    desired_guided_steps: int
    finalize_threshold: float


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


@dataclass
class SolarResult:
    text: str
    reasoning: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)
