from __future__ import annotations

import json
import re
from typing import Any

import requests

from .audit import make_call_record
from .config import Config
from .types import RESPONSE_LENGTHS, SolarResult, Usage
from .util import extract_json, recover_candidate_strings


class SolarError(RuntimeError):
    pass


class SolarClient:
    def __init__(self, config: Config):
        self.config = config
        self.call_count = 0
        self.audit_calls: list[dict[str, Any]] = []
        self.last_render_stats: dict[str, Any] = {}
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
        request_log = {
            "messages": messages,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
        }
        try:
            response = self.session.post(self.config.solar_url, json=payload, timeout=self.config.request_timeout)
        except requests.RequestException as exc:
            self.audit_calls.append(make_call_record(
                provider="solar",
                model=self.config.solar_model,
                request=request_log,
                response=None,
                usage={"reported": False, "input_tokens": 0, "output_tokens": 0},
                error={"type": type(exc).__name__, "message": str(exc)},
            ))
            raise SolarError(f"Solar request failed: {exc}") from exc

        if not response.ok:
            body = response.text[:2000]
            self.audit_calls.append(make_call_record(
                provider="solar",
                model=self.config.solar_model,
                request=request_log,
                response={"http_status": response.status_code, "body": body},
                usage={"reported": False, "input_tokens": 0, "output_tokens": 0},
                error={"type": "HTTPError", "message": f"HTTP {response.status_code}"},
            ))
            raise SolarError(f"Solar HTTP {response.status_code}: {body}")

        data = response.json()
        try:
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            self.audit_calls.append(make_call_record(
                provider="solar",
                model=self.config.solar_model,
                request=request_log,
                response={"raw": data},
                usage={"reported": False, "input_tokens": 0, "output_tokens": 0},
                error={"type": type(exc).__name__, "message": "Unexpected Solar response shape"},
            ))
            raise SolarError(f"Unexpected Solar response: {data}") from exc

        usage_raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0),
            output_tokens=int(usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0),
        )
        usage_reported = isinstance(data.get("usage"), dict) and any(
            key in usage_raw for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens")
        )
        result = SolarResult(text=text, reasoning=message.get("reasoning"), usage=usage, raw=data)
        self.audit_calls.append(make_call_record(
            provider="solar",
            model=self.config.solar_model,
            request=request_log,
            response={
                "content": text,
                "reasoning": message.get("reasoning"),
                "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
                "raw": data,
            },
            usage={
                "reported": usage_reported,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "raw": usage_raw,
            },
        ))
        return result

    def plan(self, user_text: str, history: list[dict[str, str]]) -> dict[str, Any]:
        """Build a compact route without silently degrading on reasoning-only output.

        Solar reasoning models can spend the whole completion budget in the
        provider-side reasoning field and leave content empty. A generic
        "Answer directly" fallback erased exactly the constraints JevNet needed
        in hard problems, so planning now gets one bounded repair attempt.
        """
        recent = history[-8:]
        system = (
            "You are the route planner inside a hybrid Jev+Solar language model. "
            "Return ONLY a compact JSON route, not a solution and not chain-of-thought. "
            "Name important verification obligations when the answer could hinge on "
            "a construction, invariant, counterexample, calculation, or executable claim."
        )
        prompt = {
            "user_request": user_text,
            "recent_conversation": recent,
            "output_schema": {
                "intent": "short string",
                "route": ["3-6 short high-level operations"],
                "required_points": ["facts/actions that must appear, may be empty"],
                "verification_needs": [
                    "claims that must be tested rather than merely scored, may be empty"
                ],
                "answer_shape": "short description",
                "risk_or_uncertainty": ["only material uncertainties, may be empty"],
                "language": "target response language",
            },
        }

        parsed: dict[str, Any] = {}
        attempts = [
            ("low", 900, system),
            (
                "low",
                1600,
                system
                + " The previous attempt did not yield usable JSON. Keep reasoning minimal "
                  "and put the JSON in the visible content field.",
            ),
        ]
        for effort, max_tokens, attempt_system in attempts:
            result = self.chat(
                [
                    {"role": "system", "content": attempt_system},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                reasoning_effort=effort,
                max_tokens=max_tokens,
            )
            if not (result.text or "").strip():
                continue
            try:
                candidate = extract_json(result.text)
            except (ValueError, json.JSONDecodeError):
                candidate = {}
            if isinstance(candidate, dict) and candidate:
                parsed = candidate
                break

        parsed.setdefault("intent", user_text[:160])
        parsed.setdefault(
            "route",
            [
                "Identify materially different conclusions or solution families",
                "Test answer-changing claims with explicit evidence",
                "Answer only after unresolved contradictions are handled",
            ],
        )
        parsed.setdefault("required_points", [])
        parsed.setdefault("verification_needs", [])
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

    @staticmethod
    def _dedupe_texts(items: list[str], limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = " ".join((item or "").strip().split())
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= limit:
                break
        return out

    @classmethod
    def _recover_blueprints_lenient(cls, text: str, limit: int) -> list[str]:
        """Recover semantic candidates without making JSON format a hard dependency.

        The normal parser remains first. If Solar returns valid prose instead of the
        requested wrapper, that prose is still usable as one candidate rather than
        crashing the entire inference search.
        """
        raw = (text or "").strip()
        if not raw:
            return []

        recovered = recover_candidate_strings(raw, limit)
        if recovered:
            return cls._dedupe_texts(recovered, limit)

        # Explicit fallback protocols used by repair prompts.
        marker_parts: list[str] = []
        for marker in ("CANDIDATE::", "<CANDIDATE>", "===CANDIDATE==="):
            if marker in raw:
                marker_parts = [part.strip() for part in raw.split(marker) if part.strip()]
                if marker_parts:
                    break
        if marker_parts:
            return cls._dedupe_texts(marker_parts, limit)

        # Some models ignore the wrapper and emit paragraph-separated alternatives.
        paragraphs = [part.strip() for part in raw.replace("\r\n", "\n").split("\n\n") if part.strip()]
        if 1 < len(paragraphs) <= limit * 2:
            return cls._dedupe_texts(paragraphs, limit)

        # Last parser-level fallback: any non-empty successful Solar completion is
        # still an externally judgeable proposal. Jev can score/prune it later.
        return cls._dedupe_texts([raw], limit)

    @staticmethod
    def _route_fallback_blueprint(user_text: str, plan: dict[str, Any], stage: str) -> str:
        """Build a minimal non-model fallback from the already available route."""
        intent = str(plan.get("intent") or user_text[:240] or "Answer the user request").strip()
        route = plan.get("route") or []
        if not isinstance(route, list):
            route = [str(route)]
        route = [str(x).strip() for x in route if str(x).strip()][:6]
        required = plan.get("required_points") or []
        if not isinstance(required, list):
            required = [str(required)]
        required = [str(x).strip() for x in required if str(x).strip()][:6]
        pieces = [f"Intent: {intent}"]
        if route:
            pieces.append("Route: " + " -> ".join(route))
        if required:
            pieces.append("Must cover: " + "; ".join(required))
        pieces.append(f"Fallback stage: {stage}")
        return " | ".join(pieces)

    def _repair_blueprint_plain(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        stage: str,
        avoid: list[str],
        reasoning_effort: str,
    ) -> str | None:
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "stage": stage,
            "avoid_duplicates_of": avoid[-12:],
        }
        result = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Return exactly ONE compact semantic answer blueprint as plain text. "
                        "No JSON, no label, no preface, no chain-of-thought. "
                        "It must be directly judgeable for correctness and constraint fit."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=2048,
        )
        value = (result.text or "").strip()
        return value or None

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
        """Generate compact answer blueprints without format-fragile failure.

        Solar is allowed to drift away from JSON. A successful but oddly formatted
        completion is recovered as a candidate; repeated empty completions degrade
        to the already-computed route instead of aborting JevNet.
        """
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
            max_tokens=min(10000, max(4096, 1400 + count * 360)),
        )
        out = self._recover_blueprints_lenient(result.text, count)

        # Second pass deliberately changes the output protocol. If JSON is what
        # caused the failure, repeating the same JSON request is not a real repair.
        if len(out) < max(2, count // 2):
            missing = count - len(out)
            repair_payload = {**payload, "candidate_count": missing, "avoid_duplicates_of": out}
            repair = self.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Generate additional compact answer blueprints. Do NOT use JSON. "
                            "Write each candidate after the exact marker CANDIDATE:: and nothing else."
                        ),
                    },
                    {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)},
                ],
                reasoning_effort=reasoning_effort,
                max_tokens=min(10000, max(4096, 1400 + max(1, missing) * 360)),
            )
            out = self._dedupe_texts(out + self._recover_blueprints_lenient(repair.text, missing), count)

        # Plain-text single-candidate repairs remove all list/wrapper parsing from
        # the equation. Cap retries so robustness does not become an infinite loop.
        repair_attempts = min(3, max(0, count - len(out)))
        for attempt in range(repair_attempts):
            candidate = self._repair_blueprint_plain(
                user_text=user_text,
                history=history,
                plan=plan,
                stage=f"{stage}; plain repair {attempt + 1}",
                avoid=out,
                reasoning_effort=reasoning_effort,
            )
            if candidate:
                out = self._dedupe_texts(out + [candidate], count)
            if len(out) >= count:
                break

        if not out:
            # This is intentionally deterministic: parser/format failure must not
            # crash the network after the route planner already succeeded.
            out = [self._route_fallback_blueprint(user_text, plan, stage)]
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
            max_tokens=min(10000, max(4096, 1300 + count * 360)),
        )
        out = self._recover_blueprints_lenient(result.text, count)
        if out:
            return out
        return parents[:1]

    def discriminate_hypotheses(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        candidates: list[dict[str, str]],
        reasoning_effort: str = "high",
    ) -> dict[str, Any]:
        """Produce evidence that discriminates mutually incompatible hypotheses.

        This is deliberately different from semantic scoring. The verifier must
        identify an answer-changing claim and try to *test* it: replay a proposed
        construction, substitute into equations, check a counterexample, derive an
        invariant, enumerate a genuinely small finite case, or state that an
        external deterministic tool is required. Unsupported plausibility is not
        evidence.
        """
        compact = [
            {"id": str(item.get("id", "")), "hypothesis": str(item.get("text", ""))}
            for item in candidates[:8]
            if str(item.get("id", "")).strip() and str(item.get("text", "")).strip()
        ]
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-6:],
            "route": plan,
            "competing_hypotheses": compact,
            "requirements": [
                "Do not choose by style, confidence, familiarity, or majority vote.",
                "Find the smallest decisive test that separates the incompatible claims.",
                "Actually perform every check that can be performed from the prompt itself.",
                "For a claimed construction, replay it against the stated rules.",
                "For a claimed numeric/rank/parity/algebraic fact, derive or calculate it rather than trusting it.",
                "If a real external tool/runtime is required and unavailable, mark EXTERNAL_REQUIRED instead of pretending it was verified.",
                "PASS means the tested claim survived a concrete check; FAIL means a concrete contradiction/counterexample was found; UNCERTAIN means the check was not decisive.",
            ],
            "output_schema": {
                "material_disagreement": "boolean",
                "test": "short description of the decisive check actually attempted",
                "test_kind": "DETERMINISTIC | DERIVATION | COUNTEREXAMPLE | EXTERNAL_REQUIRED | INCONCLUSIVE",
                "resolved": "boolean; true only if evidence actually separates the hypotheses",
                "winner_id": "candidate id only when resolved, otherwise null",
                "results": [
                    {
                        "candidate_id": "candidate id",
                        "verdict": "PASS | FAIL | UNCERTAIN",
                        "confidence": "0..1",
                        "evidence": "brief externally inspectable result, not hidden reasoning",
                    }
                ],
            },
        }
        system = (
            "You are the evidence verifier inside an inference-time search system. "
            "The candidates may be confidently wrong. Run a discriminating check; "
            "do not merely rate plausibility. Return strict JSON only."
        )
        attempts = [
            (reasoning_effort, 3600),
            ("medium", 2800),
        ]
        last_text = ""
        for effort, max_tokens in attempts:
            result = self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                reasoning_effort=effort,
                max_tokens=max_tokens,
            )
            last_text = (result.text or "").strip()
            if not last_text:
                continue
            try:
                parsed = extract_json(last_text)
            except (ValueError, json.JSONDecodeError):
                parsed = None
            if not isinstance(parsed, dict):
                continue

            valid_ids = {item["id"] for item in compact}
            cleaned_results: list[dict[str, Any]] = []
            for item in parsed.get("results") or []:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("candidate_id", "")).strip()
                if cid not in valid_ids:
                    continue
                verdict = str(item.get("verdict", "UNCERTAIN")).upper()
                if verdict not in {"PASS", "FAIL", "UNCERTAIN"}:
                    verdict = "UNCERTAIN"
                try:
                    confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    confidence = 0.5
                cleaned_results.append({
                    "candidate_id": cid,
                    "verdict": verdict,
                    "confidence": confidence,
                    "evidence": str(item.get("evidence", "")).strip(),
                })

            winner = parsed.get("winner_id")
            winner = str(winner).strip() if winner is not None else None
            if winner not in valid_ids:
                winner = None
            test_kind = str(parsed.get("test_kind", "INCONCLUSIVE")).upper()
            winner_pass = any(
                item["candidate_id"] == winner
                and item["verdict"] == "PASS"
                and item["confidence"] >= 0.60
                for item in cleaned_results
            )
            separated_rival = any(
                item["candidate_id"] != winner
                and item["verdict"] == "FAIL"
                and item["confidence"] >= 0.60
                for item in cleaned_results
            )
            resolved = (
                bool(parsed.get("resolved"))
                and winner is not None
                and winner_pass
                and separated_rival
                and test_kind not in {"INCONCLUSIVE", "EXTERNAL_REQUIRED"}
            )
            return {
                "material_disagreement": bool(parsed.get("material_disagreement", True)),
                "test": str(parsed.get("test", "")).strip(),
                "test_kind": test_kind,
                "resolved": resolved,
                "winner_id": winner if resolved else None,
                "results": cleaned_results,
                "raw": parsed,
            }

        return {
            "material_disagreement": True,
            "test": "Verifier did not return a usable evidence report.",
            "test_kind": "INCONCLUSIVE",
            "resolved": False,
            "winner_id": None,
            "results": [],
            "raw_text": last_text,
        }

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
                "Treat the supplied parents as competing falsifiable hypotheses, not as a consensus to refine.",
                "Attack the strongest answer-changing claim with a concrete counterexample, direct substitution, construction replay, invariant, or other checkable test.",
                "Preserve a materially different rival when the current leader is not actually verified.",
                "Turn the result into rival/repaired candidate blueprints that include the decisive evidence, not merely stronger-sounding prose.",
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
            max_tokens=min(10000, max(4096, 1500 + count * 420)),
        )
        out = self._recover_blueprints_lenient(result.text, count)
        if out:
            return out[:count]

        # Remove structured-output dependency completely for one repair attempt.
        repaired = self._repair_blueprint_plain(
            user_text=user_text,
            history=history,
            plan=plan,
            stage=f"adaptive {action} round {round_index + 1}",
            avoid=existing,
            reasoning_effort=reasoning_effort,
        )
        if repaired:
            return [repaired]

        # If Solar returns successful-but-empty content twice, do not kill a long
        # Jev search. Reuse an already judged survivor so the controller can
        # continue to VERIFY/STOP on the next round.
        survivor = next((x.strip() for x in parents if x and x.strip()), None)
        if survivor is None:
            survivor = next((x.strip() for x in existing if x and x.strip()), None)
        if survivor:
            return [survivor]
        return [self._route_fallback_blueprint(user_text, plan, f"adaptive {action}")]

    @staticmethod
    def _finish_reason(result: SolarResult) -> str | None:
        raw = result.raw if isinstance(result.raw, dict) else {}
        try:
            value = raw["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        return str(value) if value is not None else None

    @staticmethod
    def _dedupe_final_drafts(items: list[str], limit: int) -> list[str]:
        """Deduplicate complete final answers without flattening Markdown/newlines."""
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = (item or "").strip()
            if not value:
                continue
            key = " ".join(value.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= limit:
                break
        return out

    @classmethod
    def _recover_final_drafts(cls, text: str, limit: int) -> tuple[list[str], str]:
        """Recover whole final answers, never bullet/heading fragments."""
        raw = (text or "").strip()
        if not raw:
            return [], "empty"

        parsed: Any = None
        try:
            parsed = extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            parsed = None

        def item_text(item: Any) -> str | None:
            if isinstance(item, str):
                return item.strip() or None
            if isinstance(item, dict):
                for key in ("text", "content", "answer", "draft", "candidate"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            return None

        pool: list[Any] = []
        if isinstance(parsed, dict):
            for key in ("candidates", "drafts", "answers", "options", "choices"):
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

        values: list[str] = []
        for item in pool:
            value = item_text(item)
            if value:
                values.append(value)
        structured = cls._dedupe_final_drafts(values, limit)
        if structured:
            return structured, "structured"

        stripped = raw.lstrip()
        if (
            stripped.startswith("{")
            or stripped.startswith("[")
            or stripped.lower().startswith(chr(96) * 3 + "json")
        ):
            return [], "malformed_structured"

        # Unlike blueprint recovery, Markdown prose is one complete draft.
        return [raw], "raw_prose"

    @staticmethod
    def _surface_answer_problem(
        text: str,
        *,
        user_text: str,
        plan: dict[str, Any],
        finish_reason: str | None = None,
    ) -> str | None:
        value = (text or "").strip()
        if not value:
            return "empty final answer"
        if finish_reason and finish_reason.lower() in {"length", "max_tokens", "max_output_tokens"}:
            return f"generation stopped by {finish_reason}"

        compact = " ".join(value.split())
        required = plan.get("required_points") or []
        answer_shape = str(plan.get("answer_shape") or "")
        complex_request = (
            len(required) >= 2
            or bool(re.search(
                r"\b(prove|proof|minimal|minimality|construction|construct|derive|explain|justify|counterexample)\b",
                user_text + " " + answer_shape,
                flags=re.I,
            ))
        )

        if complex_request and len(compact) <= 96:
            return "final answer is too short for the requested multi-part reasoning task"
        if len(compact) <= 120 and compact.count("**") % 2 == 1:
            return "unbalanced Markdown in a short final fragment"
        if compact.count(chr(96) * 3) % 2 == 1:
            return "unclosed fenced code block"
        return None

    def _repair_final_surface(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        chosen_blueprint: str,
        supporting_blueprints: list[str],
        broken_output: str,
        reason: str,
        ceiling: int,
    ) -> str | None:
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "winning_blueprint": chosen_blueprint,
            "supporting_survivors": supporting_blueprints,
            "broken_or_incomplete_output": broken_output[-5000:],
            "repair_reason": reason,
            "requirements": [
                "Return ONE complete user-facing answer as plain text/Markdown, not JSON.",
                "Answer every explicitly requested part.",
                "Do not return only a heading, label, outline fragment, or sentence stub.",
                "Preserve the winning blueprint semantics; repair surface completeness/formatting.",
                "Do not mention this repair process or internal candidates.",
            ],
        }
        for effort, max_tokens in [
            ("medium", max(2048, ceiling * 2)),
            ("low", max(3072, ceiling * 2)),
        ]:
            result = self.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Repair the incomplete final response. Return exactly one complete "
                            "user-facing answer. No JSON wrapper and no preface."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                reasoning_effort=effort,
                max_tokens=max_tokens,
            )
            candidate = (result.text or "").strip()
            problem = self._surface_answer_problem(
                candidate,
                user_text=user_text,
                plan=plan,
                finish_reason=self._finish_reason(result),
            )
            if candidate and problem is None:
                return candidate
            payload["broken_or_incomplete_output"] = candidate[-5000:]
            payload["repair_reason"] = problem or reason
        return None

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
        length_data = next(
            ((target, ceiling, desc) for name, target, ceiling, desc in RESPONSE_LENGTHS if name == response_length),
            (550, 1400, "normal complete answer"),
        )
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
                "Each draft must independently answer the whole request.",
            ],
        }

        if count == 1:
            result = self.chat(
                [
                    {
                        "role": "system",
                        "content": "Render the selected blueprint into the best complete final user-facing answer. Return only the answer.",
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                reasoning_effort=reasoning_effort,
                max_tokens=ceiling,
            )
            text = (result.text or "").strip()
            problem = self._surface_answer_problem(
                text,
                user_text=user_text,
                plan=plan,
                finish_reason=self._finish_reason(result),
            )
            if text and problem is None:
                self.last_render_stats = {
                    "mode": "single", "protocol": "plain", "repaired": False, "drafts": 1
                }
                return [text]

            repaired = self._repair_final_surface(
                user_text=user_text,
                history=history,
                plan=plan,
                chosen_blueprint=chosen_blueprint,
                supporting_blueprints=supporting_blueprints,
                broken_output=text,
                reason=problem or "empty final surface",
                ceiling=ceiling,
            )
            self.last_render_stats = {
                "mode": "single",
                "protocol": "plain",
                "repaired": bool(repaired),
                "drafts": 1 if repaired else 0,
                "problem": problem,
            }
            return [repaired or chosen_blueprint]

        result = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Render multiple strong COMPLETE final answers. "
                        "Return strict JSON with a candidates array of full answers. "
                        "Each candidate must independently answer the whole user request. "
                        "Never put headings or bullet fragments in the candidates array."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort=reasoning_effort,
            max_tokens=min(10000, max(4096, ceiling * count)),
        )
        finish_reason = self._finish_reason(result)
        drafts, protocol = self._recover_final_drafts(result.text, count)

        valid: list[str] = []
        rejected: list[dict[str, str]] = []
        for draft in drafts:
            problem = self._surface_answer_problem(
                draft,
                user_text=user_text,
                plan=plan,
                finish_reason=finish_reason,
            )
            if problem is None:
                valid.append(draft)
            else:
                rejected.append({"problem": problem, "preview": draft[:180]})

        if valid:
            self.last_render_stats = {
                "mode": "multi",
                "protocol": protocol,
                "repaired": False,
                "drafts": len(valid),
                "rejected": len(rejected),
                "finish_reason": finish_reason,
            }
            return self._dedupe_final_drafts(valid, count)

        broken = (result.text or "").strip()
        reason = rejected[0]["problem"] if rejected else f"malformed final-draft protocol ({protocol})"
        repaired = self._repair_final_surface(
            user_text=user_text,
            history=history,
            plan=plan,
            chosen_blueprint=chosen_blueprint,
            supporting_blueprints=supporting_blueprints,
            broken_output=broken,
            reason=reason,
            ceiling=ceiling,
        )
        self.last_render_stats = {
            "mode": "multi",
            "protocol": protocol,
            "repaired": bool(repaired),
            "drafts": 1 if repaired else 0,
            "rejected": len(rejected),
            "finish_reason": finish_reason,
            "problem": reason,
        }
        return [repaired or chosen_blueprint]

    def direct_answer(self, *, user_text: str, history: list[dict[str, str]], reasoning_effort: str = "high", response_length: str = "medium") -> str:
        """Solar-only baseline with empty-surface recovery.

        Keep the normal baseline as a single direct completion. Only when the API
        succeeds but returns empty `content` do we retry with an explicit surface-
        answer instruction and a larger output budget. This avoids turning ordinary
        baseline runs into a multi-stage pipeline while making transient/reasoning-
        only completions recoverable.
        """
        _target, ceiling, _desc = next(
            ((target, max_tokens, desc) for name, target, max_tokens, desc in RESPONSE_LENGTHS if name == response_length),
            (550, 1400, "normal"),
        )
        normal_system = "Answer the user directly, accurately, and naturally."
        recovery_system = (
            "Answer the user directly, accurately, and naturally. "
            "Return a complete user-facing final answer in the content field. "
            "Do not return only hidden reasoning and do not leave the answer empty."
        )

        attempts = [
            (reasoning_effort, ceiling, normal_system),
            ("medium", max(2048, ceiling * 2), recovery_system),
            ("low", max(3072, ceiling * 2), recovery_system),
        ]

        last_result: SolarResult | None = None
        for effort, max_tokens, system in attempts:
            messages = [
                {"role": "system", "content": system},
                *history[-8:],
                {"role": "user", "content": user_text},
            ]
            last_result = self.chat(messages, reasoning_effort=effort, max_tokens=max_tokens)
            out = (last_result.text or "").strip()
            if out:
                return out

        finish_reason = None
        if last_result is not None and isinstance(last_result.raw, dict):
            try:
                finish_reason = last_result.raw["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError, AttributeError):
                finish_reason = None
        detail = f"; finish_reason={finish_reason}" if finish_reason else ""
        raise SolarError(f"Solar direct answer remained empty after {len(attempts)} attempts{detail}")
