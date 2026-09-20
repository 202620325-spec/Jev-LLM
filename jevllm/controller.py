from __future__ import annotations

from dataclasses import dataclass

from .types import (
    BREADTH_LEVELS,
    COGNITIVE_MODES,
    HORIZONS,
    MODE_TO_REASONING,
    REASONING_SCOPES,
    RESPONSE_LENGTHS,
    AdaptiveProfile,
    ControlProfile,
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _norm(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return _clip(index / float(count - 1))


def _index(value: float, count: int) -> int:
    return max(0, min(count - 1, round(_clip(value) * (count - 1))))


@dataclass
class _LatentState:
    horizon: float = 0.5
    mode: float = 0.4
    breadth: float = 0.35
    scope: float = 0.5
    length: float = 0.5
    initialized: bool = False


class AdaptiveController:
    """A tiny recurrent routing network over Jev decisions.

    This is deliberately not called a trained neural network: there are no learned
    weights. It behaves *like* a small recurrent controller: Jev outputs become
    activations, activations are smoothed over time, then cross-coupled to derive
    an execution budget for the next generation step.
    """

    def __init__(self, *, run_mode: str, fast_base_steps: int, full_base_steps: int, max_steps: int = 6):
        self.run_mode = run_mode
        self.fast_base_steps = max(1, fast_base_steps)
        self.full_base_steps = max(1, full_base_steps)
        self.max_steps = max(1, max_steps)
        self.state = _LatentState()
        self.momentum = 0.55

    @staticmethod
    def _uncertainty(profile: ControlProfile) -> float:
        values = [v for v in profile.confidence.values() if isinstance(v, (int, float)) and v > 0]
        if not values:
            return 0.45
        mean_conf = sum(max(0.0, min(1.0, float(v))) for v in values) / len(values)
        return _clip(1.0 - mean_conf)

    def update(self, profile: ControlProfile) -> AdaptiveProfile:
        breadth_idx = min(range(len(BREADTH_LEVELS)), key=lambda i: abs(BREADTH_LEVELS[i] - profile.breadth))
        raw = {
            "horizon": _norm(profile.horizon_index, len(HORIZONS)),
            "mode": _norm(COGNITIVE_MODES.index(profile.cognitive_mode), len(COGNITIVE_MODES)),
            "breadth": _norm(breadth_idx, len(BREADTH_LEVELS)),
            "scope": _norm(profile.reasoning_scope_index, len(REASONING_SCOPES)),
            "length": _norm(profile.response_length_index, len(RESPONSE_LENGTHS)),
        }
        uncertainty = self._uncertainty(profile)

        if not self.state.initialized:
            for key, value in raw.items():
                setattr(self.state, key, value)
            self.state.initialized = True
        else:
            for key, value in raw.items():
                old = getattr(self.state, key)
                setattr(self.state, key, self.momentum * old + (1.0 - self.momentum) * value)

        mode_signal = _clip(self.state.mode + 0.12 * self.state.scope + 0.10 * uncertainty)
        breadth_signal = _clip(self.state.breadth + 0.16 * uncertainty + 0.10 * mode_signal)
        scope_signal = _clip(self.state.scope + 0.08 * mode_signal)
        length_signal = _clip(self.state.length + 0.05 * scope_signal)
        horizon_signal = _clip(self.state.horizon + 0.08 * (1.0 - uncertainty) - 0.08 * mode_signal)

        mode_idx = _index(mode_signal, len(COGNITIVE_MODES))
        horizon_idx = _index(horizon_signal, len(HORIZONS))
        scope_idx = _index(scope_signal, len(REASONING_SCOPES))
        length_idx = _index(length_signal, len(RESPONSE_LENGTHS))
        breadth_level_idx = _index(breadth_signal, len(BREADTH_LEVELS))

        mode = COGNITIVE_MODES[mode_idx]
        breadth = BREADTH_LEVELS[breadth_level_idx]

        minimums = {"predict": 1, "options": 2, "reason_low": 2, "reason_medium": 3, "reason_high": 4, "reason_extreme": 5}
        maximums = {"predict": 1, "options": 4, "reason_low": 3, "reason_medium": 4, "reason_high": 5, "reason_extreme": 6}
        breadth = max(minimums[mode], min(breadth, maximums[mode]))

        deliberation = _clip(
            0.34 * mode_signal
            + 0.24 * scope_signal
            + 0.16 * breadth_signal
            + 0.16 * uncertainty
            + 0.10 * (1.0 - horizon_signal)
        )

        if self.run_mode == "fast":
            cap = min(self.max_steps, self.fast_base_steps + 1)
            base = self.fast_base_steps
        else:
            cap = self.max_steps
            base = self.full_base_steps
        desired_steps = min(cap, max(1, round(base + deliberation * max(0, cap - base))))

        finalize_threshold = _clip(0.58 + 0.24 * deliberation, 0.58, 0.86)

        length_name, target_tokens, max_tokens, _ = RESPONSE_LENGTHS[length_idx]
        return AdaptiveProfile(
            horizon_index=horizon_idx,
            horizon=HORIZONS[horizon_idx][0],
            cognitive_mode=mode,
            breadth=breadth,
            reasoning_scope_index=scope_idx,
            reasoning_scope=REASONING_SCOPES[scope_idx][0],
            response_length_index=length_idx,
            response_length=length_name,
            response_target_tokens=target_tokens,
            response_max_tokens=max_tokens,
            reasoning_effort=MODE_TO_REASONING[mode],
            deliberation_budget=deliberation,
            uncertainty=uncertainty,
            desired_guided_steps=desired_steps,
            finalize_threshold=finalize_threshold,
        )
