from __future__ import annotations

import json
from typing import Any

import requests

from .config import Config
from .types import RESPONSE_LENGTHS, SolarResult, Usage
from .util import extract_json, recover_candidate_strings


class SolarError(RuntimeError):
    pass


class SolarClient:
    def __init__(self, config: Config):
        self.config = config
        self.call_count = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.upstage_api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> SolarResult:
        payload: dict[str, Any] = {
            "model": self.config.solar_model,
            "messages": messages,
            "stream": False,
        }
        if reasoning_effort in {"low", "medium", "high"}:
            payload["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        self.call_count += 1
        try:
            response = self.session.post(self.config.solar_url, json=payload, timeout=self.config.request_timeout)
        except requests.RequestException as exc:
            raise SolarError(f"Solar request failed: {exc}") from exc

        if not response.ok:
            body = response.text[:2000]
            raise SolarError(f"Solar HTTP {response.status_code}: {body}")

        data = response.json()
        try:
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise SolarError(f"Unexpected Solar response: {data}") from exc

        usage_raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0),
            output_tokens=int(usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0),
        )
        return SolarResult(text=text, reasoning=message.get("reasoning"), usage=usage, raw=data)

    def plan(self, user_text: str, history: list[dict[str, str]]) -> dict[str, Any]:
        recent = history[-8:]
        system = (
            "You are the route planner inside a hybrid Jev+Solar language model. "
            "Design a compact answer path, not chain-of-thought. Return JSON only. "
            "The route contains high-level answer operations, required constraints, and answer shape."
        )
        prompt = {
            "user_request": user_text,
            "recent_conversation": recent,
            "output_schema": {
                "intent": "short string",
                "route": ["3-6 short high-level operations"],
                "required_points": ["facts/actions that must appear, may be empty"],
                "answer_shape": "short description",
                "risk_or_uncertainty": ["only material uncertainties, may be empty"],
                "language": "target response language",
            },
        }
        result = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            reasoning_effort="medium",
            max_tokens=700,
        )
        try:
            parsed = extract_json(result.text)
        except (ValueError, json.JSONDecodeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        parsed.setdefault("intent", user_text[:160])
        parsed.setdefault("route", ["Answer the request directly"])
        parsed.setdefault("required_points", [])
        parsed.setdefault("answer_shape", "direct answer")
        parsed.setdefault("risk_or_uncertainty", [])
        return parsed

    @staticmethod
    def _candidate_token_budget(horizon: str, breadth: int) -> int:
        base = {
            "character": 384,
            "subword": 384,
            "word": 448,
            "phrase": 512,
            "clause": 640,
            "sentence": 800,
        }.get(horizon, 220)
        return min(1800, base + max(0, breadth - 1) * 160)

    def _repair_single_continuation(
        self,
        *,
        user_text: str,
        plan: dict[str, Any],
        prefix: str,
        horizon: str,
        horizon_rule: str,
        reasoning_scope: str,
        reasoning_effort: str,
    ) -> str:
        payload = {
            "user_request": user_text,
            "solar_plan": plan,
            "response_prefix": prefix,
            "prediction_horizon": horizon,
            "horizon_rule": horizon_rule,
            "reasoning_scope": reasoning_scope,
        }
        result = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly ONE literal next response continuation and nothing else. "
                        "No JSON, no label, no explanation. Respect the requested horizon."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=self._candidate_token_budget(horizon, 1),
        )
        value = result.text.strip()
        if not value:
            raise SolarError("Solar returned an empty continuation even after fallback repair")
        return value

    def propose_continuations(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        prefix: str,
        horizon: str,
        horizon_rule: str,
        cognitive_mode: str,
        breadth: int,
        reasoning_effort: str,
        reasoning_scope: str = "whole_answer",
        response_length: str = "medium",
    ) -> list[str]:
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-6:],
            "solar_plan": plan,
            "response_prefix": prefix,
            "prediction_horizon": horizon,
            "horizon_rule": horizon_rule,
            "cognitive_mode": cognitive_mode,
            "reasoning_scope": reasoning_scope,
            "final_response_length": response_length,
            "candidate_count": breadth,
            "requirements": [
                "Candidates are literal next output fragments, never meta-commentary.",
                "Respect the requested language and format.",
                "Keep each candidate at the requested horizon scale.",
                "Do not repeat response_prefix.",
            ],
        }

        # The common failure case was choices=1 + a fragile JSON wrapper. Avoid that wrapper entirely.
        if breadth <= 1:
            return [
                self._repair_single_continuation(
                    user_text=user_text,
                    plan=plan,
                    prefix=prefix,
                    horizon=horizon,
                    horizon_rule=horizon_rule,
                    reasoning_scope=reasoning_scope,
                    reasoning_effort=reasoning_effort,
                )
            ]

        system = (
            "Generate materially distinct literal continuations for the response. Do not explain reasoning. "
            "Prefer strict JSON {\"candidates\":[\"...\"]}. Each item may also be {\"text\":\"...\"}."
        )
        result = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=self._candidate_token_budget(horizon, breadth),
        )

        candidates = recover_candidate_strings(result.text, breadth)
        if candidates:
            return candidates[:breadth]

        # A valid non-empty Solar response should not kill the whole chat because its JSON shape drifted.
        # Fall back to one direct literal continuation; Jev then has a deterministic C0.
        fallback = self._repair_single_continuation(
            user_text=user_text,
            plan=plan,
            prefix=prefix,
            horizon=horizon,
            horizon_rule=horizon_rule,
            reasoning_scope=reasoning_scope,
            reasoning_effort=reasoning_effort,
        )
        return [fallback]

    def finalize(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        selected_prefix: str,
        reasoning_effort: str,
        extreme: bool,
        reasoning_scope: str = "whole_answer",
        response_length: str = "medium",
        response_target_tokens: int = 550,
        response_max_tokens: int = 1400,
    ) -> str:
        length_desc = next((desc for name, _, _, desc in RESPONSE_LENGTHS if name == response_length), "normal complete answer")
        system = (
            "You are the language surface generator in a Jev-controlled hybrid model. "
            "Answer directly and naturally. Use the supplied route as high-level guidance. "
            "Do not reveal hidden reasoning, internal scoring, or chain-of-thought."
        )
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "solar_plan": plan,
            "jev_selected_opening_or_scaffold": selected_prefix,
            "reasoning_scope": reasoning_scope,
            "response_length": {
                "level": response_length,
                "desired_surface_tokens": response_target_tokens,
                "target": length_desc,
            },
            "instructions": [
                "Preserve the useful semantic direction chosen by Jev.",
                "If the scaffold is a natural literal opening, start with it exactly when practical.",
                "Think over the declared reasoning scope but expose only the user-facing answer.",
                "Respect the response-length target; do not fill the token budget unnecessarily.",
                "Complete the answer; do not mention this pipeline unless asked.",
            ],
        }
        first = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=response_max_tokens,
        ).text.strip()

        if not first:
            raise SolarError("Solar finalizer returned an empty answer")
        if not extreme:
            return first

        audit_system = (
            "Verify and revise the draft for correctness, completeness, instruction-following, and clarity. "
            "Check globally across the declared reasoning scope. Do not expose private reasoning. "
            "Return only the revised user-facing answer and preserve the requested answer length."
        )
        audit_payload = {
            "user_request": user_text,
            "solar_plan": plan,
            "jev_scaffold": selected_prefix,
            "reasoning_scope": reasoning_scope,
            "response_length": response_length,
            "draft": first,
        }
        revised = self.chat(
            [
                {"role": "system", "content": audit_system},
                {"role": "user", "content": json.dumps(audit_payload, ensure_ascii=False)},
            ],
            reasoning_effort="high",
            max_tokens=response_max_tokens,
        ).text.strip()
        return revised or first

    def expand_reasoning_paths(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        count: int,
        stage: str,
        reasoning_effort: str = "high",
    ) -> list[str]:
        """Generate compact answer blueprints, not hidden chain-of-thought."""
        count = max(1, min(32, count))
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "stage": stage,
            "candidate_count": count,
            "requirements": [
                "Generate materially different answer blueprints/solution approaches.",
                "Each candidate must be self-contained enough for a decision model to judge.",
                "Use concise claims, steps, assumptions, or answer structure; do not reveal private chain-of-thought.",
                "Include alternative interpretations when ambiguity is material.",
                "Do not optimize wording; optimize semantic diversity and correctness potential.",
            ],
        }
        system = (
            "You are the proposal population generator inside an inference-time decision network. "
            "Return strict JSON {\"candidates\":[\"...\"]}. Generate distinct compact answer blueprints, not final prose and not chain-of-thought."
        )
        result = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            reasoning_effort=reasoning_effort,
            max_tokens=min(6400, 900 + count * 210),
        )
        out = recover_candidate_strings(result.text, count)
        # One supplementation pass if the model under-produces badly.
        if len(out) < max(2, count // 2):
            missing = count - len(out)
            supplement = self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({**payload, "candidate_count": missing, "avoid_duplicates_of": out}, ensure_ascii=False)},
                ],
                reasoning_effort=reasoning_effort,
                max_tokens=min(4800, 700 + missing * 210),
            )
            for item in recover_candidate_strings(supplement.text, missing):
                if item not in out:
                    out.append(item)
                if len(out) >= count:
                    break
        if not out:
            raise SolarError("Solar hypothesis expansion returned no recoverable candidates")
        return out[:count]

    def mutate_reasoning_paths(
        self,
        *,
        user_text: str,
        plan: dict[str, Any],
        parents: list[str],
        count: int,
        layer_index: int,
        reasoning_effort: str = "high",
    ) -> list[str]:
        count = max(0, min(16, count))
        if count == 0 or not parents:
            return []
        payload = {
            "user_request": user_text,
            "route": plan,
            "layer": layer_index + 1,
            "surviving_parent_blueprints": parents,
            "child_count": count,
            "operations": [
                "repair a likely weak assumption without copying wording",
                "combine complementary strengths from different parents",
                "invert a key assumption when plausible and see if it yields a stronger route",
                "make an implicit constraint explicit",
                "produce a more robust route that survives obvious counterexamples",
            ],
            "rule": "Return concise child blueprints only; no hidden chain-of-thought.",
        }
        result = self.chat(
            [
                {"role": "system", "content": "Recombine and mutate surviving answer blueprints. Return strict JSON {\"candidates\":[\"...\"]}."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=min(5200, 700 + count * 240),
        )
        return recover_candidate_strings(result.text, count)

    def adaptive_reasoning_operation(
        self,
        *,
        action: str,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        parents: list[str],
        existing: list[str],
        count: int,
        round_index: int,
        reasoning_effort: str = "high",
    ) -> list[str]:
        """Generate only the candidates requested by Jev's adaptive search action."""
        action = action.upper()
        count = max(1, min(24, count))
        operation_instructions = {
            "REFILL": [
                "Generate additional strong alternatives near the currently promising solution region.",
                "Do not merely paraphrase existing candidates; vary assumptions, structure, or implementation details when useful.",
            ],
            "DIVERSE_REFILL": [
                "Generate approaches that are deliberately different from the existing pool.",
                "Explore different decompositions, interpretations, algorithms, or solution families when plausible.",
                "Avoid cosmetic wording diversity; seek semantic diversity.",
            ],
            "DEEPEN": [
                "Take the strongest supplied parent blueprints and make their missing logic/technical structure explicit.",
                "Resolve important gaps, edge cases, and dependencies without turning the result into hidden chain-of-thought.",
            ],
            "MUTATE": [
                "Repair weak assumptions, constraint misses, or brittle steps in the parent blueprints.",
                "Create variants that preserve strengths while changing the likely failure point.",
            ],
            "MERGE": [
                "Combine complementary strengths from multiple parent blueprints into new coherent candidates.",
                "Do not concatenate blindly; reconcile conflicts and produce one integrated route per candidate.",
            ],
            "CHALLENGE": [
                "Attack the strongest parent routes with plausible counterexamples, alternative interpretations, or failure modes.",
                "Turn those challenges into rival/repaired candidate blueprints that could beat the current leader.",
            ],
        }
        instructions = operation_instructions.get(action, operation_instructions["REFILL"])
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "adaptive_round": round_index + 1,
            "action": action,
            "parent_blueprints": parents,
            "existing_pool": existing[:24],
            "candidate_count": count,
            "requirements": [
                *instructions,
                "Return compact semantic answer blueprints/solution approaches, not final polished prose.",
                "Do not reveal private chain-of-thought; provide only concise externally judgeable claims/steps/structure.",
                "Avoid duplicates of the existing pool.",
            ],
        }
        system = (
            "You are Solar acting as the proposal generator for an adaptive Jev decision network. "
            "Perform exactly the requested search operation. Return strict JSON {\"candidates\":[\"...\"]}."
        )
        result = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=min(7200, 850 + count * 260),
        )
        out = recover_candidate_strings(result.text, count)
        if out:
            return out[:count]

        # Do not abort the whole network because one action response drifted from JSON.
        # Fall back to a tiny generic expansion, still respecting the requested count cap.
        fallback = self.expand_reasoning_paths(
            user_text=user_text,
            history=history,
            plan=plan,
            count=min(count, 3),
            stage=f"fallback after adaptive action {action}",
            reasoning_effort=reasoning_effort,
        )
        return fallback[:count]

    def render_answer_drafts(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        chosen_blueprint: str,
        supporting_blueprints: list[str],
        count: int,
        response_length: str,
        reasoning_effort: str,
    ) -> list[str]:
        count = max(1, min(6, count))
        length_data = next(((target, ceiling, desc) for name, target, ceiling, desc in RESPONSE_LENGTHS if name == response_length), (550, 1400, "normal complete answer"))
        target, ceiling, desc = length_data
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "winning_blueprint": chosen_blueprint,
            "supporting_survivors": supporting_blueprints,
            "draft_count": count,
            "response_length": {"name": response_length, "target_tokens": target, "description": desc},
            "requirements": [
                "Answer the user directly.",
                "Preserve the semantic strengths of the winning blueprint.",
                "Use supporting survivors only when they improve correctness or completeness.",
                "Do not mention the internal network, candidates, scores, or hidden reasoning unless asked.",
                "Drafts should be independently phrased but semantically strong, not cosmetic paraphrases.",
            ],
        }
        if count == 1:
            text = self.chat(
                [
                    {"role": "system", "content": "Render the selected blueprint into the best final user-facing answer. Return only the answer."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                reasoning_effort=reasoning_effort,
                max_tokens=ceiling,
            ).text.strip()
            return [text] if text else []

        result = self.chat(
            [
                {"role": "system", "content": "Render multiple strong final answers. Return strict JSON {\"candidates\":[\"full answer\", ...]}."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=min(8000, ceiling * count),
        )
        drafts = recover_candidate_strings(result.text, count)
        if drafts:
            return drafts[:count]
        fallback = self.chat(
            [
                {"role": "system", "content": "Render the selected blueprint into the best final user-facing answer. Return only the answer."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=ceiling,
        ).text.strip()
        return [fallback] if fallback else []

    def direct_answer(self, *, user_text: str, history: list[dict[str, str]], reasoning_effort: str = "high", response_length: str = "medium") -> str:
        _target, ceiling, _desc = next(((target, max_tokens, desc) for name, target, max_tokens, desc in RESPONSE_LENGTHS if name == response_length), (550, 1400, "normal"))
        messages = [
            {"role": "system", "content": "Answer the user directly, accurately, and naturally."},
            *history[-8:],
            {"role": "user", "content": user_text},
        ]
        out = self.chat(messages, reasoning_effort=reasoning_effort, max_tokens=ceiling).text.strip()
        if not out:
            raise SolarError("Solar direct answer was empty")
        return out
