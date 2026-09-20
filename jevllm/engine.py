from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any, Callable

from .audit import ConversationJSONLogger
from .config import Config
from .controller import AdaptiveController
from .jev import JevClient
from .jevnet import JevDecisionNetwork
from .solar import SolarClient
from .types import HORIZONS, MODE_RANK


EventSink = Callable[[str, dict[str, Any]], None]


class JevLLM:
    def __init__(self, config: Config, event_sink: EventSink | None = None):
        self.config = config
        self.jev = JevClient(config)
        self.solar = SolarClient(config)
        self.history: list[dict[str, str]] = []
        self.event_sink = event_sink or (lambda _event, _data: None)
        self.audit_events: list[dict[str, Any]] = []
        self.conversation_logger = ConversationJSONLogger(
            config.conversation_log_dir,
            enabled=config.conversation_log_enabled,
        )
        self.last_log_path: str | None = None

        self.pipeline_mode = "net"       # net | solar | legacy
        self.intensity = "auto"          # auto | fast | full | max
        self.width_override: int | None = None
        self.layers_override: int | None = None
        self.drafts_override: int | None = None
        self.mutation_override: int | None = None  # max refill per action in v1.3
        self.seed_override: int | None = None
        self.generated_override: int | None = None
        self.run_mode = config.default_run_mode  # legacy only
        self.last_stats: dict[str, Any] = {}

        self.network = JevDecisionNetwork(config, self.jev, self.solar, self._capture_event)

    def reset(self) -> None:
        self.history.clear()
        self.audit_events.clear()
        self.conversation_logger = ConversationJSONLogger(
            self.config.conversation_log_dir,
            enabled=self.config.conversation_log_enabled,
        )
        self.last_log_path = None

    def clear_overrides(self) -> None:
        self.width_override = None
        self.layers_override = None
        self.drafts_override = None
        self.mutation_override = None
        self.seed_override = None
        self.generated_override = None
        self.intensity = "auto"

    def _capture_event(self, event: str, data: dict[str, Any]) -> None:
        self.audit_events.append({
            "ts_ns": time.time_ns(),
            "event": event,
            "data": data,
        })
        self.event_sink(event, data)

    def _emit(self, event: str, **data: Any) -> None:
        self._capture_event(event, data)

    def audit_cursor(self) -> dict[str, int]:
        return {
            "solar": len(getattr(self.solar, "audit_calls", [])),
            "jev": len(getattr(self.jev, "audit_calls", [])),
            "events": len(self.audit_events),
        }

    def _audit_slice(self, cursor: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        solar_calls = list(getattr(self.solar, "audit_calls", []))[cursor.get("solar", 0):]
        jev_calls = list(getattr(self.jev, "audit_calls", []))[cursor.get("jev", 0):]
        calls = solar_calls + jev_calls
        events = self.audit_events[cursor.get("events", 0):]
        return calls, events

    def record_turn_log(
        self,
        *,
        question: str,
        answer: Any,
        pipeline: str,
        cursor: dict[str, int],
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        calls, events = self._audit_slice(cursor)
        path = self.conversation_logger.append_turn(
            question=question,
            answer=answer,
            pipeline=pipeline,
            calls=calls,
            events=events,
            stats=dict(self.last_stats),
            extra=extra,
        )
        self.last_log_path = str(path) if path is not None else None
        if self.last_log_path is not None:
            self.last_stats["conversation_log"] = self.last_log_path
        return self.last_log_path

    def _remember(self, user_text: str, answer: str) -> None:
        self.history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
        max_messages = self.config.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def answer(self, user_text: str, *, log_turn: bool = True) -> str:
        if not user_text.strip():
            return ""
        audit_cursor = self.audit_cursor()

        if self.pipeline_mode == "solar":
            start_solar = getattr(self.solar, "call_count", 0)
            answer = self.solar.direct_answer(user_text=user_text, history=self.history, reasoning_effort="high", response_length="medium")
            self.last_stats = {"pipeline": "solar", "solar_calls": getattr(self.solar, "call_count", start_solar) - start_solar, "jev_calls": 0}
            self._remember(user_text, answer)
            if log_turn:
                self.record_turn_log(
                    question=user_text,
                    answer=answer,
                    pipeline="solar",
                    cursor=audit_cursor,
                )
            return answer

        if self.pipeline_mode == "legacy":
            answer = self._answer_legacy(user_text)
            if log_turn:
                self.record_turn_log(
                    question=user_text,
                    answer=answer,
                    pipeline="legacy",
                    cursor=audit_cursor,
                )
            return answer

        start_solar = getattr(self.solar, "call_count", 0)
        start_jev = getattr(self.jev, "call_count", 0)
        self.network.jev = self.jev
        self.network.solar = self.solar
        plan = self.solar.plan(user_text, self.history)
        self._emit("plan", plan=plan)
        answer = self.network.run(
            user_text=user_text,
            history=self.history,
            plan=plan,
            intensity=self.intensity,
            width_override=self.width_override,
            layers_override=self.layers_override,
            drafts_override=self.drafts_override,
            mutation_override=self.mutation_override,
            seed_override=self.seed_override,
            generated_override=self.generated_override,
        )
        self.last_stats = dict(self.network.last_stats)
        self.last_stats.update({
            "pipeline": "net",
            "solar_calls_this_turn": getattr(self.solar, "call_count", start_solar) - start_solar,
            "jev_calls_this_turn": getattr(self.jev, "call_count", start_jev) - start_jev,
        })
        self._remember(user_text, answer)
        if log_turn:
            self.record_turn_log(
                question=user_text,
                answer=answer,
                pipeline="net",
                cursor=audit_cursor,
            )
        return answer

    @staticmethod
    def _append_fragment(prefix: str, fragment: str, horizon: str) -> str:
        if not prefix:
            return fragment
        if not fragment:
            return prefix
        if fragment[0].isspace() or prefix[-1].isspace():
            return prefix + fragment
        if horizon in {"character", "subword"}:
            return prefix + fragment
        if re.match(r"[\w가-힣]", prefix[-1]) and re.match(r"[\w가-힣]", fragment[0]):
            return prefix + " " + fragment
        return prefix + fragment

    def _hard_step_cap(self) -> int:
        if self.run_mode == "fast":
            return min(self.config.adaptive_max_guided_steps, self.config.fast_guided_steps + 1)
        return self.config.adaptive_max_guided_steps

    def _answer_legacy(self, user_text: str) -> str:
        start_solar = getattr(self.solar, "call_count", 0)
        start_jev = getattr(self.jev, "call_count", 0)
        plan = self.solar.plan(user_text, self.history)
        self._emit("plan", plan=plan)
        prefix = ""
        highest_mode = "predict"
        final_adaptive = None
        hard_cap = self._hard_step_cap()
        controller = AdaptiveController(
            run_mode=self.run_mode,
            fast_base_steps=self.config.fast_guided_steps,
            full_base_steps=self.config.full_guided_steps,
            max_steps=self.config.adaptive_max_guided_steps,
        )
        for step in range(hard_cap):
            raw_profile = self.jev.controls(user_text=user_text, history=self.history, plan=plan, prefix=prefix, step=step)
            adaptive = controller.update(raw_profile); final_adaptive = adaptive
            if MODE_RANK[adaptive.cognitive_mode] > MODE_RANK[highest_mode]:
                highest_mode = adaptive.cognitive_mode
            horizon_rule = HORIZONS[adaptive.horizon_index][1]
            self._emit("control", profile=asdict(raw_profile), adaptive=asdict(adaptive), effective_breadth=adaptive.breadth)
            candidates = self.solar.propose_continuations(
                user_text=user_text, history=self.history, plan=plan, prefix=prefix,
                horizon=adaptive.horizon, horizon_rule=horizon_rule, cognitive_mode=adaptive.cognitive_mode,
                breadth=adaptive.breadth, reasoning_effort=adaptive.reasoning_effort,
                reasoning_scope=adaptive.reasoning_scope, response_length=adaptive.response_length,
            )
            self._emit("candidates", candidates=candidates)
            if len(candidates) == 1:
                chosen_index = 0
            else:
                chosen_index, selection_raw = self.jev.choose_candidate(
                    user_text=user_text, plan=plan, prefix=prefix, candidates=candidates,
                    horizon=adaptive.horizon, reasoning_scope=adaptive.reasoning_scope,
                )
                self._emit("selection", index=chosen_index, raw=selection_raw)
            prefix = self._append_fragment(prefix, candidates[chosen_index], adaptive.horizon)
            self._emit("prefix", prefix=prefix)
            completed_steps = step + 1
            if completed_steps >= adaptive.desired_guided_steps and raw_profile.ready_to_finalize >= adaptive.finalize_threshold:
                break
        if final_adaptive is None:
            raise RuntimeError("adaptive controller produced no profile")
        answer = self.solar.finalize(
            user_text=user_text, history=self.history, plan=plan, selected_prefix=prefix,
            reasoning_effort=final_adaptive.reasoning_effort, extreme=(highest_mode == "reason_extreme"),
            reasoning_scope=final_adaptive.reasoning_scope, response_length=final_adaptive.response_length,
            response_target_tokens=final_adaptive.response_target_tokens, response_max_tokens=final_adaptive.response_max_tokens,
        )
        self.last_stats = {
            "pipeline": "legacy", "solar_calls": getattr(self.solar, "call_count", start_solar) - start_solar,
            "jev_calls": getattr(self.jev, "call_count", start_jev) - start_jev,
        }
        self._remember(user_text, answer)
        return answer

    def doctor(self) -> tuple[bool, str]:
        jev = self.jev.doctor()
        jev_choice = (jev.get("answers") or {}).get("route", {}).get("choice")
        solar = self.solar.chat(
            [{"role": "system", "content": "Connection test. Reply exactly OK."}, {"role": "user", "content": "ping"}],
            reasoning_effort="low", max_tokens=32,
        )
        ok = jev_choice == "ok" and "OK" in solar.text.upper()
        return ok, f"Jev={jev_choice!r}, Solar={solar.text.strip()!r}"
