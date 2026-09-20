from __future__ import annotations

import json
import re
from typing import Any

import requests

from .audit import make_call_record
from .config import Config
from .routing import classify_query_mode
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
        mode = classify_query_mode(user_text)
        if mode == "simple_definition":
            return {
                "intent": "simple_definition",
                "route": ["identify conventional meaning", "state it directly"],
                "required_points": [],
                "verification_needs": [],
                "answer_shape": "1-2 sentence concrete definition",
                "risk_or_uncertainty": [],
                "language": "match_user",
                "query_mode": "simple_definition",
            }

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
        parsed.setdefault("query_mode", mode)
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
        generation_mode: str = "normal",
    ) -> list[str]:
        """Generate compact answer blueprints without format-fragile failure.

        Solar is allowed to drift away from JSON. A successful but oddly formatted
        completion is recovered as a candidate; repeated empty completions degrade
        to the already-computed route instead of aborting JevNet.
        """
        count = max(1, min(32, count))
        simple_definition = generation_mode == "simple_definition"
        if simple_definition:
            requirements = [
                "Generate only conventional, directly relevant meanings of the term in the user's stated context.",
                "Do not invent speculative alternate senses, certifications, brands, incidents, organizations, or unrelated domain interpretations.",
                "Each candidate must be a compact semantic definition, not a job description or encyclopedia expansion.",
                "A candidate may add at most one nearby clarifying example when it materially helps identify the meaning.",
                "Prefer semantic precision over diversity. If three genuinely different grounded meanings do not exist, return fewer candidates.",
                "Do not reveal private chain-of-thought.",
            ]
            system = (
                "You generate compact grounded definition blueprints. Stay inside the user's local semantic context. "
                "Never manufacture diversity. Return strict JSON {\"candidates\":[\"...\"]}."
            )
            max_tokens = min(1800, max(900, 420 + count * 260))
            effective_effort = "low"
        else:
            requirements = [
                "Generate materially different answer blueprints/solution approaches.",
                "Each candidate must be self-contained enough for a decision model to judge.",
                "Use concise claims, steps, assumptions, or answer structure; do not reveal private chain-of-thought.",
                "Include alternative interpretations only when ambiguity is material and grounded in the request.",
                "Do not optimize wording; optimize semantic diversity and correctness potential.",
            ]
            system = (
                "You are the proposal population generator inside an inference-time decision network. "
                "Return strict JSON {\"candidates\":[\"...\"]}. Generate distinct compact answer blueprints, not final prose and not chain-of-thought."
            )
            max_tokens = min(10000, max(4096, 1400 + count * 360))
            effective_effort = reasoning_effort

        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "stage": stage,
            "generation_mode": generation_mode,
            "candidate_count": count,
            "requirements": requirements,
        }

        result = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            reasoning_effort=effective_effort,
            max_tokens=max_tokens,
        )
        out = self._recover_blueprints_lenient(result.text, count)

        # Second pass deliberately changes the output protocol. If JSON is what
        # caused the failure, repeating the same JSON request is not a real repair.
        if len(out) < max(2, count // 2) and not simple_definition:
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
        repair_attempts = 0 if simple_definition and out else min(3, max(0, count - len(out)))
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

    def audit_claims(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        text: str,
        stage: str,
        verification_evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Audit atomic support claims independently of the selected conclusion.

        A correct conclusion with a false proof is a failed audit. Exact counts,
        parity/invariant claims, ranks, state-space sizes, exhaustive-search claims,
        constructions, and algebraic identities must be recomputed or downgraded.
        """
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-4:],
            "route": plan,
            "stage": stage,
            "text_to_audit": text,
            "prior_verified_evidence": verification_evidence or [],
            "rules": [
                "Separate the final conclusion from the claims used to support it.",
                "A correct conclusion does NOT excuse a false supporting claim.",
                "Extract every answer-changing or proof-supporting factual/logical claim.",
                "For exact counts, arithmetic, parity, ranks, dimensions, state-space sizes, BFS/exhaustive-search claims, and constructions: recompute or derive them from the original problem instead of trusting the text.",
                "For a construction, replay it against the actual rules.",
                "For an invariant, explicitly check one allowed operation preserves it.",
                "For a rank/dimension claim, require an actual derivation/certificate; otherwise mark UNCERTAIN and remove it from the proof.",
                "PASS only when the check is externally inspectable. Plausibility is not verification.",
                "If a claim is wrong or unnecessary and unverifiable, produce a repaired compact answer/proof that removes or replaces it.",
                "Do not reveal private chain-of-thought; evidence must be concise, checkable results only.",
            ],
            "output_schema": {
                "conclusion": "short statement of the text's conclusion",
                "conclusion_status": "PASS | FAIL | UNCERTAIN",
                "claims": [
                    {
                        "claim": "atomic supporting claim",
                        "importance": "ANSWER_CHANGING | SUPPORTING | OPTIONAL",
                        "verdict": "PASS | FAIL | UNCERTAIN",
                        "check_kind": "ARITHMETIC | DERIVATION | INVARIANT | CONSTRUCTION | ENUMERATION | EXTERNAL_REQUIRED | OTHER",
                        "check": "what was actually checked",
                        "evidence": "concise checkable result",
                    }
                ],
                "repaired_text": "compact corrected answer/proof, or empty string if no repair is needed",
            },
        }
        system = (
            "You are a claim-level proof auditor. Audit the support structure, not the prose style. "
            "The selected conclusion may be right while its proof is wrong. Return strict JSON only."
        )
        last_text = ""
        for effort, max_tokens in [("high", 4200), ("medium", 3000)]:
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

            cleaned: list[dict[str, str]] = []
            for item in parsed.get("claims") or []:
                if not isinstance(item, dict):
                    continue
                claim = " ".join(str(item.get("claim", "")).split())
                if not claim:
                    continue
                importance = str(item.get("importance", "SUPPORTING")).upper()
                if importance not in {"ANSWER_CHANGING", "SUPPORTING", "OPTIONAL"}:
                    importance = "SUPPORTING"
                verdict = str(item.get("verdict", "UNCERTAIN")).upper()
                if verdict not in {"PASS", "FAIL", "UNCERTAIN"}:
                    verdict = "UNCERTAIN"
                check_kind = str(item.get("check_kind", "OTHER")).upper()
                check = str(item.get("check", "")).strip()
                evidence = str(item.get("evidence", "")).strip()

                # A bare PASS with no actual check/evidence is not verification.
                if verdict == "PASS" and (len(check) < 4 or len(evidence) < 3):
                    verdict = "UNCERTAIN"
                if check_kind == "EXTERNAL_REQUIRED" and verdict == "PASS":
                    verdict = "UNCERTAIN"

                cleaned.append({
                    "claim": claim,
                    "importance": importance,
                    "verdict": verdict,
                    "check_kind": check_kind,
                    "check": check,
                    "evidence": evidence,
                })

            conclusion_status = str(parsed.get("conclusion_status", "UNCERTAIN")).upper()
            if conclusion_status not in {"PASS", "FAIL", "UNCERTAIN"}:
                conclusion_status = "UNCERTAIN"

            material = [x for x in cleaned if x["importance"] in {"ANSWER_CHANGING", "SUPPORTING"}]
            failed = [x for x in material if x["verdict"] == "FAIL"]
            uncertain = [x for x in material if x["verdict"] == "UNCERTAIN"]
            if failed:
                status = "FAIL"
            elif uncertain or (not material and stage in {"blueprint", "surface"}):
                status = "UNCERTAIN"
            elif conclusion_status == "FAIL":
                status = "FAIL"
            elif conclusion_status == "UNCERTAIN":
                status = "UNCERTAIN"
            else:
                status = "PASS"

            repaired = str(parsed.get("repaired_text") or "").strip()
            return {
                "status": status,
                "conclusion": str(parsed.get("conclusion") or "").strip(),
                "conclusion_status": conclusion_status,
                "claims": cleaned,
                "failed_claims": failed,
                "uncertain_claims": uncertain,
                "repaired_text": repaired,
                "raw": parsed,
            }

        return {
            "status": "UNCERTAIN",
            "conclusion": "",
            "conclusion_status": "UNCERTAIN",
            "claims": [],
            "failed_claims": [],
            "uncertain_claims": [],
            "repaired_text": "",
            "raw_text": last_text,
        }

    def repair_from_claim_audit(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        text: str,
        audit: dict[str, Any],
        verification_evidence: list[dict[str, Any]] | None = None,
        response_length: str = "medium",
    ) -> str:
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-4:],
            "route": plan,
            "text_to_repair": text,
            "claim_audit": {
                "status": audit.get("status"),
                "failed_claims": audit.get("failed_claims") or [],
                "uncertain_claims": audit.get("uncertain_claims") or [],
                "conclusion_status": audit.get("conclusion_status"),
            },
            "verified_evidence": verification_evidence or [],
            "requirements": [
                "Keep a conclusion only if it remains supported.",
                "Remove every FAIL claim.",
                "Remove or replace every material UNCERTAIN claim instead of presenting it as fact.",
                "Prefer the shortest proof/argument whose key steps can actually be checked.",
                "Do not introduce new exact counts, ranks, dimensions, BFS/state-space claims, invariants, or constructions unless you explicitly derive/check them.",
                "Return only the repaired user-facing answer.",
            ],
        }
        _target, ceiling, _desc = next(
            ((target, max_tokens, desc) for name, target, max_tokens, desc in RESPONSE_LENGTHS if name == response_length),
            (550, 1400, "normal"),
        )
        result = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Repair the answer using the claim audit. Correctness of the proof has priority over "
                        "coverage or sophistication. Return only the repaired answer."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort="high",
            max_tokens=max(2200, ceiling * 2),
        )
        return (result.text or "").strip() or str(audit.get("repaired_text") or "").strip() or text

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
            "REVIVE": [
                "Generate clean-room hypotheses from the original user request only.",
                "Do not inherit the current leader, current rival, their terminology, their constructions, or their claimed invariants.",
                "Re-derive the problem independently and seek solution families that the existing pool may have missed.",
                "Use this after a discriminating test failed to resolve an entrenched disagreement.",
            ],
        }
        instructions = operation_instructions.get(action, operation_instructions["REFILL"])
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "adaptive_round": round_index + 1,
            "action": action,
            "parent_blueprints": [] if action == "REVIVE" else parents,
            "existing_pool": [] if action == "REVIVE" else existing[:24],
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
            "Perform exactly the requested search operation. For REVIVE, solve from the original request clean-room "
            "and ignore prior hypotheses completely. Return strict JSON {\"candidates\":[\"...\"]}."
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
    def _infer_register(user_text: str, plan: dict[str, Any]) -> str:
        text = (user_text or "").strip()
        language = str(plan.get("language") or "").lower()
        if re.search(r"(뭐냐|뭐임|뭔데|뭐야|ㅋㅋ|야\??$)", text):
            return "casual_korean"
        if re.search(r"(무엇인가요|뭔가요|알려주세요|설명해주세요|요\??$)", text):
            return "polite_korean"
        if "korean" in language or "ko" == language:
            return "neutral_korean"
        return "natural_user_register"

    @staticmethod
    def _clean_state_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = " ".join(str(item or "").strip().split())
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    def _sanitize_state_lock(
        self,
        raw: dict[str, Any],
        *,
        user_text: str,
        plan: dict[str, Any],
        chosen_blueprint: str,
        query_mode: str,
    ) -> dict[str, Any]:
        simple = query_mode == "simple_definition"
        if simple:
            # The state compiler is not allowed to strengthen or broaden the
            # semantic claim selected by Jev.
            required = [chosen_blueprint.strip()]
        else:
            required = self._clean_state_list(raw.get("required_claims"), 12)
            if not required:
                required = [chosen_blueprint.strip()]

        active = self._clean_state_list(raw.get("active_concepts"), 6 if simple else 14)
        optional = self._clean_state_list(raw.get("optional_concepts"), 2 if simple else 6)
        suppressed = self._clean_state_list(raw.get("suppressed_concepts"), 10 if simple else 16)

        if simple:
            grounding_source = (user_text + " " + chosen_blueprint).casefold()
            source_tokens = {
                token.rstrip("s")
                for token in re.findall(r"[0-9a-zA-Z가-힣_-]{2,}", grounding_source)
            }

            def grounded(concept: str) -> bool:
                tokens = {
                    token.rstrip("s")
                    for token in re.findall(r"[0-9a-zA-Z가-힣_-]{2,}", concept.casefold())
                }
                return bool(tokens & source_tokens)

            active = [concept for concept in active if grounded(concept)]
            optional = [concept for concept in optional if grounded(concept)]

        try:
            max_sentences = int(raw.get("max_sentences", 2 if simple else 8))
        except (TypeError, ValueError):
            max_sentences = 2 if simple else 8
        max_sentences = max(1, min(2 if simple else 12, max_sentences))

        try:
            max_chars = int(raw.get("max_chars", 220 if simple else 2400))
        except (TypeError, ValueError):
            max_chars = 220 if simple else 2400
        max_chars = max(80, min(320 if simple else 6000, max_chars))

        return {
            "schema": "jev_state_lock_v1",
            "intent": str(raw.get("intent") or ("simple_definition" if simple else plan.get("intent") or "answer")).strip(),
            "active_concepts": active,
            "optional_concepts": optional,
            "required_claims": required,
            "suppressed_concepts": suppressed,
            "register": str(raw.get("register") or self._infer_register(user_text, plan)).strip(),
            "abstraction": str(raw.get("abstraction") or ("concrete_definition" if simple else "task_appropriate")).strip(),
            "max_sentences": max_sentences,
            "max_chars": max_chars,
            "allow_new_factual_concepts": False,
            "query_mode": query_mode,
        }

    def build_state_lock(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        chosen_blueprint: str,
        verification_evidence: list[dict[str, Any]] | None = None,
        query_mode: str = "normal",
    ) -> dict[str, Any]:
        """Compile the selected semantic route into a closed surface-generation state."""
        simple = query_mode == "simple_definition"
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-6:],
            "route": plan,
            "selected_blueprint": chosen_blueprint,
            "verification_evidence": verification_evidence or [],
            "query_mode": query_mode,
            "output_schema": {
                "intent": "short label",
                "active_concepts": ["concepts explicitly supported by the request/selected blueprint/evidence"],
                "optional_concepts": ["at most nearby clarification already supported by those sources"],
                "required_claims": ["claims that the answer must communicate"],
                "suppressed_concepts": ["likely tangents or expansions that must not enter the final answer"],
                "register": "user-matching register",
                "abstraction": "surface abstraction level",
                "max_sentences": 2 if simple else 8,
                "max_chars": 220 if simple else 2400,
            },
            "rules": [
                "This is semantic compression, not brainstorming.",
                "ACTIVE, OPTIONAL, and REQUIRED content must be grounded in the user request, selected blueprint, or verification evidence.",
                "Do not import facts from losing candidates or general world knowledge into ACTIVE/OPTIONAL/REQUIRED.",
                "Put tempting but unnecessary expansions in SUPPRESSED instead of adding them to the answerable state.",
                "For a simple definition, lock onto the conventional meaning and at most one nearby clarifier.",
                "Do not write final prose. Return JSON only.",
            ],
        }
        system = (
            "Compile a CLOSED semantic state for a constrained answer verbalizer. "
            "Your job is to reduce the state space, not expand it. Return JSON only."
        )
        result = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort="low" if simple else "medium",
            max_tokens=900 if simple else 1600,
        )
        parsed: dict[str, Any] = {}
        try:
            maybe = extract_json(result.text)
            if isinstance(maybe, dict):
                parsed = maybe
        except (ValueError, json.JSONDecodeError):
            parsed = {}
        return self._sanitize_state_lock(
            parsed,
            user_text=user_text,
            plan=plan,
            chosen_blueprint=chosen_blueprint,
            query_mode=query_mode,
        )

    @staticmethod
    def _state_lock_problem(text: str, state_lock: dict[str, Any] | None) -> str | None:
        if not state_lock:
            return None
        value = (text or "").strip()
        if not value:
            return "empty state-locked answer"
        max_chars = int(state_lock.get("max_chars", 0) or 0)
        if max_chars and len(value) > max_chars:
            return f"state-lock length exceeded ({len(value)}>{max_chars})"
        max_sentences = int(state_lock.get("max_sentences", 0) or 0)
        if max_sentences:
            sentence_count = len([x for x in re.split(r"(?<=[.!?。！？])\s+|\n+", value) if x.strip()])
            if sentence_count > max_sentences:
                return f"state-lock sentence budget exceeded ({sentence_count}>{max_sentences})"
        lower = value.casefold()
        for phrase in state_lock.get("suppressed_concepts") or []:
            p = str(phrase).strip().casefold()
            if len(p) >= 3 and p in lower:
                return f"suppressed concept surfaced: {phrase}"
        return None

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
        verification_evidence: list[dict[str, Any]] | None = None,
        state_lock: dict[str, Any] | None = None,
    ) -> str | None:
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-8:],
            "route": plan,
            "winning_blueprint": chosen_blueprint,
            "supporting_survivors": supporting_blueprints,
            "verification_evidence": verification_evidence or [],
            "state_lock": state_lock or {},
            "broken_or_incomplete_output": broken_output[-5000:],
            "repair_reason": reason,
            "requirements": [
                "Return ONE complete user-facing answer as plain text/Markdown, not JSON.",
                "Answer every explicitly requested part.",
                "Do not return only a heading, label, outline fragment, or sentence stub.",
                "Preserve the winning blueprint semantics except where verification evidence explicitly overrides it.",
                "Never resurrect a claim marked FAIL by verification evidence.",
                "When state_lock is present, use ONLY its required_claims, active_concepts, and optional_concepts.",
                "Never introduce a new factual concept outside state_lock. Prefer omission over expansion.",
                "Respect state_lock max_sentences/max_chars and never surface suppressed_concepts.",
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
            lock_problem = self._state_lock_problem(candidate, state_lock)
            if candidate and problem is None and lock_problem is None:
                return candidate
            problem = lock_problem or problem
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
        verification_evidence: list[dict[str, Any]] | None = None,
        state_lock: dict[str, Any] | None = None,
    ) -> list[str]:
        count = max(1, min(6, count))
        length_data = next(
            ((target, ceiling, desc) for name, target, ceiling, desc in RESPONSE_LENGTHS if name == response_length),
            (550, 1400, "normal complete answer"),
        )
        target, ceiling, desc = length_data
        if state_lock:
            # STATE LOCK: the surface model never sees losing blueprints. It is a
            # verbalizer, not a second semantic search.
            count = 1
            payload = {
                "user_request": user_text,
                "recent_conversation": history[-4:],
                "state_lock": state_lock,
                "verification_evidence": verification_evidence or [],
                "draft_count": 1,
                "requirements": [
                    "Verbalize the locked state; do not perform new semantic search.",
                    "Use ONLY required_claims, active_concepts, and optional_concepts from state_lock.",
                    "Do not add factual concepts, duties, examples, organizations, history, implications, or qualifications absent from state_lock.",
                    "Never surface suppressed_concepts.",
                    "Respect register, abstraction, max_sentences, and max_chars.",
                    "Prefer a shorter sufficient answer over a broader answer.",
                    "Do not mention the state lock, candidates, scores, or hidden reasoning.",
                ],
            }
        else:
            payload = {
                "user_request": user_text,
                "recent_conversation": history[-8:],
                "route": plan,
                "winning_blueprint": chosen_blueprint,
                "supporting_survivors": supporting_blueprints,
                "verification_evidence": verification_evidence or [],
                "draft_count": count,
                "response_length": {"name": response_length, "target_tokens": target, "description": desc},
                "requirements": [
                    "Answer the user directly.",
                    "Preserve the semantic strengths of the winning blueprint.",
                    "Use supporting survivors only when they improve correctness or completeness.",
                    "Do not mention the internal network, candidates, scores, or hidden reasoning unless asked.",
                    "Each draft must independently answer the whole request.",
                    "When verification evidence is present, it overrides unsupported or failed claims from the original blueprint.",
                    "Do not resurrect a claim marked FAIL by the evidence report.",
                ],
            }

        if count == 1:
            result = self.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a constrained verbalizer. If state_lock is supplied, you MUST stay inside it: "
                            "no new factual concepts, no semantic expansion, no extra duties/examples unless explicitly allowed. "
                            "Return only the complete user-facing answer."
                        ),
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
            lock_problem = self._state_lock_problem(text, state_lock)
            if text and problem is None and lock_problem is None:
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
                reason=lock_problem or problem or "empty final surface",
                ceiling=ceiling,
                verification_evidence=verification_evidence,
                state_lock=state_lock,
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
            verification_evidence=verification_evidence,
            state_lock=state_lock,
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

    def repair_state_locked_answer(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        state_lock: dict[str, Any],
        answer: str,
        audit: dict[str, Any],
    ) -> str:
        payload = {
            "user_request": user_text,
            "recent_conversation": history[-4:],
            "state_lock": state_lock,
            "answer_to_repair": answer,
            "conformance_audit": {
                "state_violation": audit.get("state_violation"),
                "unsupported_expansion": audit.get("unsupported_expansion"),
                "register_match": audit.get("register_match"),
                "length_match": audit.get("length_match"),
            },
            "rules": [
                "Return only the repaired user-facing answer.",
                "Use ONLY required_claims, active_concepts, and optional_concepts from state_lock.",
                "Delete every extra factual concept. Do not replace it with a different extra concept.",
                "Never surface suppressed_concepts.",
                "Match register and abstraction.",
                "Respect max_sentences and max_chars.",
                "Prefer the shortest sufficient wording.",
            ],
        }
        result = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Repair a state-locked answer. This is deletion/compression, not new reasoning. "
                        "Do not add knowledge outside the supplied state."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning_effort="low",
            max_tokens=700,
        )
        candidate = (result.text or "").strip()
        if candidate and self._state_lock_problem(candidate, state_lock) is None:
            return candidate
        return answer

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
