from __future__ import annotations

import json
from typing import Any

import requests

from .audit import make_call_record
from .config import Config
from .types import (
    BREADTH_LEVELS,
    COGNITIVE_MODES,
    HORIZONS,
    REASONING_SCOPES,
    RESPONSE_LENGTHS,
    ControlProfile,
)
from .util import answer_confidence, noul_probability, score_expectation, score_level


class JevError(RuntimeError):
    pass


class JevClient:
    def __init__(self, config: Config):
        self.config = config
        self.call_count = 0
        self.audit_calls: list[dict[str, Any]] = []
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.jevllm.mip",
                "X-OpenRouter-Title": "JevNet LLM MVP",
            }
        )

    def decide(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self.config.jev_model, "state": state, "questions": questions}
        self.call_count += 1
        request_log = {"state": state, "questions": questions}
        try:
            response = self.session.post(self.config.jev_url, json=payload, timeout=self.config.request_timeout)
        except requests.RequestException as exc:
            self.audit_calls.append(make_call_record(
                provider="jev",
                model=self.config.jev_model,
                request=request_log,
                response=None,
                usage={"reported": False, "input_tokens": 0, "output_tokens": 0},
                error={"type": type(exc).__name__, "message": str(exc)},
            ))
            raise JevError(f"Jev request failed: {exc}") from exc

        if not response.ok:
            body = response.text[:2000]
            self.audit_calls.append(make_call_record(
                provider="jev",
                model=self.config.jev_model,
                request=request_log,
                response={"http_status": response.status_code, "body": body},
                usage={"reported": False, "input_tokens": 0, "output_tokens": 0},
                error={"type": "HTTPError", "message": f"HTTP {response.status_code}"},
            ))
            raise JevError(f"Jev HTTP {response.status_code}: {body}")

        data = response.json()
        answers = data.get("answers")
        usage_raw = data.get("usage") if isinstance(data, dict) else None
        usage_raw = usage_raw if isinstance(usage_raw, dict) else {}
        input_tokens = int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0)
        output_tokens = int(usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0)
        usage_reported = any(
            key in usage_raw for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens")
        )
        self.audit_calls.append(make_call_record(
            provider="jev",
            model=self.config.jev_model,
            request=request_log,
            response={
                "reasoning": data.get("reasoning") if isinstance(data, dict) else None,
                "analysis": data.get("analysis") if isinstance(data, dict) else None,
                "raw": data,
            },
            usage={
                "reported": usage_reported,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "raw": usage_raw,
            },
        ))
        if not isinstance(answers, dict):
            raise JevError(f"Unexpected Jev response: {data}")
        return data

    def network_budget(self, *, user_text: str, history: list[dict[str, str]], plan: dict[str, Any]) -> dict[str, Any]:
        state = {
            "description": "A user request plus a compact Solar route; choose inference-search resources, not prose style.",
            "records": [{
                "id": "request",
                "record": json.dumps({"user_request": user_text, "recent_conversation": history[-6:], "solar_plan": plan}, ensure_ascii=False),
            }],
        }
        scale = [
            "Minimal: the decision is obvious; extra search would mostly duplicate work.",
            "Small: a little branching can catch local mistakes.",
            "Moderate: several plausible approaches or constraints should be compared.",
            "Large: broad search and repeated filtering should materially improve reliability.",
            "Very large: many interacting constraints/ambiguities justify substantial search.",
            "Maximum practical: unusually difficult or open-ended; use the widest/deepest MVP search.",
        ]
        length = [
            "Micro: one/few compact sentences.",
            "Short: concise but sufficient.",
            "Medium: normal complete answer.",
            "Long: detailed multi-part answer.",
            "Extended: extensive answer is materially needed.",
            "Extended+: broad technical treatment is justified.",
        ]
        qs = {
            "search_width": {"type": "score", "instructions": "For record request: how broad should the initial hypothesis population be?", "criteria": scale},
            "search_layers": {"type": "score", "instructions": "For record request: how many evaluate/prune/recombine layers are useful?", "criteria": scale},
            "mutation_pressure": {"type": "score", "instructions": "For record request: how strongly should surviving hypotheses be mutated/merged before re-evaluation?", "criteria": scale},
            "final_draft_competition": {"type": "score", "instructions": "For record request: how many independently rendered final drafts should compete?", "criteria": scale},
            "response_length": {"type": "score", "instructions": "For record request: choose useful user-facing answer length; difficulty does not imply verbosity.", "criteria": length},
        }
        data = self.decide(state, qs)
        a = data["answers"]
        length_idx = score_level(a.get("response_length", {}), 6)
        length_names = ["micro", "short", "medium", "long", "extended", "extended"]
        return {
            "width": score_level(a.get("search_width", {}), 6),
            "layers": score_level(a.get("search_layers", {}), 6),
            "mutation": score_level(a.get("mutation_pressure", {}), 6),
            "drafts": score_level(a.get("final_draft_competition", {}), 6),
            "response_length_name": length_names[length_idx],
            "raw": data,
        }

    def evaluate_candidate_batch(
        self,
        *,
        user_text: str,
        plan: dict[str, Any],
        candidates: list[str],
        layer_index: int,
        total_layers: int,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        results: list[dict[str, Any]] = []
        batch_size = max(1, min(20, batch_size))

        metric_criteria = {
            "fit": [
                "Misses the request or answers a different problem.", "Weakly related but materially off-target.",
                "Partly addresses the request.", "Mostly fits the user's actual request.",
                "Strong direct fit with only small omissions.", "Excellent fit to intent, constraints and requested output.",
            ],
            "correctness": [
                "Likely wrong or internally invalid.", "Major correctness concerns.", "Several uncertain or weak claims.",
                "Plausibly correct with manageable uncertainty.", "Strong correctness likelihood.", "Very strong correctness likelihood under available information.",
            ],
            "constraints": [
                "Violates core constraints.", "Misses several important constraints.", "Meets some constraints.",
                "Meets most explicit constraints.", "Meets nearly all constraints.", "Very strong instruction/constraint satisfaction.",
            ],
            "coherence": [
                "Contradictory or structurally broken.", "Major logical gaps.", "Some weak links.",
                "Generally coherent.", "Strong internal logic.", "Exceptionally coherent and mutually consistent.",
            ],
            "coverage": [
                "Leaves out the core need.", "Very incomplete.", "Covers a minority of necessary ground.",
                "Covers the main ground.", "Broad useful coverage.", "Covers all material parts without obvious bloat.",
            ],
            "uncertainty": [
                "Low uncertainty; few unsupported assumptions.", "Small uncertainty.", "Moderate-low uncertainty.",
                "Moderate uncertainty or assumption load.", "High uncertainty.", "Very high uncertainty / fragile assumptions.",
            ],
        }

        for offset in range(0, len(candidates), batch_size):
            chunk = candidates[offset: offset + batch_size]
            records = [{
                "id": f"C{offset+i}",
                "record": json.dumps({"candidate": text, "user_request": user_text, "solar_plan": plan}, ensure_ascii=False),
            } for i, text in enumerate(chunk)]
            state = {
                "description": (
                    f"Candidate answer blueprints for one user request. Evaluation layer {layer_index+1}/{total_layers}; "
                    "each record contains the candidate and its request context."
                ),
                "records": records,
            }
            questions: dict[str, Any] = {}
            for i, _text in enumerate(chunk):
                cid = f"C{offset+i}"
                for metric, criteria in metric_criteria.items():
                    questions[f"{cid}__{metric}"] = {
                        "type": "score",
                        "instructions": f"For the record with id {cid}: rate {metric} relative to the user request and route.",
                        "criteria": criteria,
                    }
                questions[f"{cid}__survive"] = {
                    "type": "noul",
                    "instructions": f"For the record with id {cid}: should this candidate survive to the next competitive layer?",
                    "true_when": "It contains enough value/correctness potential to remain in competition.",
                    "false_when": "It is dominated, off-target, too fragile, or not worth further compute.",
                }
            data = self.decide(state, questions)
            answers = data["answers"]
            for i, _text in enumerate(chunk):
                cid = f"C{offset+i}"
                metrics = {m: score_expectation(answers.get(f"{cid}__{m}", {}), 6) for m in metric_criteria}
                survival = noul_probability(answers.get(f"{cid}__survive", {}), 0.5)
                positive = (
                    0.21 * metrics["fit"] + 0.23 * metrics["correctness"] + 0.16 * metrics["constraints"]
                    + 0.14 * metrics["coherence"] + 0.13 * metrics["coverage"] + 0.13 * (1.0 - metrics["uncertainty"])
                )
                activation = max(0.0, min(1.0, 0.84 * positive + 0.16 * survival))
                results.append({
                    "activation": activation,
                    "survival": survival,
                    "uncertainty": metrics["uncertainty"],
                    "metrics": metrics,
                })
        return results

    def search_action(
        self,
        *,
        user_text: str,
        plan: dict[str, Any],
        candidates: list[dict[str, Any]],
        round_index: int,
        max_rounds: int,
        generated_total: int,
        max_generated: int,
        max_live_pool: int,
        max_refill: int,
        allowed_actions: list[str],
    ) -> dict[str, Any]:
        """Let Jev choose the next inference operation and how much compute to spend."""
        if not candidates:
            return {"action": "STOP", "refill_count": 0, "target_span": 1, "focus_id": "POOL", "raw": {}}

        top = candidates[: min(8, len(candidates))]
        records = [{
            "id": str(item.get("id", f"C{i}")),
            "record": json.dumps({
                "candidate": item.get("text", ""),
                "activation": item.get("activation", 0.5),
                "survival": item.get("survival", 0.5),
                "uncertainty": item.get("uncertainty", 0.5),
                "metrics": item.get("metrics", {}),
            }, ensure_ascii=False),
        } for i, item in enumerate(top)]

        state = {
            "description": (
                "Current live hypothesis pool inside an adaptive inference search. "
                "Choose the next computation, not the final prose. The hard budget is an upper bound only."
            ),
            "records": records,
            "context": {
                "user_request": user_text,
                "solar_plan": plan,
                "round": round_index + 1,
                "max_rounds": max_rounds,
                "live_pool": len(candidates),
                "max_live_pool": max_live_pool,
                "generated_total": generated_total,
                "max_generated": max_generated,
                "remaining_generation_budget": max(0, max_generated - generated_total),
                "max_refill_per_action": max_refill,
            },
        }

        action_descriptions = {
            "STOP": "Current pool is sufficient. Stop search and render the best answer now.",
            "REFILL": "Request more alternatives near the currently promising solution region.",
            "DIVERSE_REFILL": "Request genuinely different approaches because the current pool is narrow or correlated.",
            "DEEPEN": "Take promising candidates and elaborate their missing technical/logical structure before judging again.",
            "MUTATE": "Repair or alter promising candidates to remove weak assumptions or constraint failures.",
            "MERGE": "Combine complementary strengths from multiple surviving candidates into new candidates.",
            "CHALLENGE": "Generate adversarial/counter-hypotheses that attack the strongest current route and expose hidden errors.",
            "VERIFY": "Do not generate new proposals yet; spend the next step re-checking the strongest existing candidates more strictly.",
        }
        action_criteria = {a: action_descriptions[a] for a in allowed_actions if a in action_descriptions}
        if not action_criteria:
            action_criteria = {"STOP": action_descriptions["STOP"]}

        refill_criteria = [
            "Tiny refill: 2 new candidates are enough.",
            "Small refill: 4 new candidates.",
            "Moderate refill: 6 new candidates.",
            "Broad refill: 8 new candidates.",
            "Very broad refill: 12 new candidates.",
            "Maximum useful refill now: 16 new candidates.",
        ]
        span_criteria = [
            "Focus on the single strongest candidate.",
            "Work on the top 2 candidates.",
            "Work on the top 3 candidates.",
            "Work on the top 4 candidates.",
            "Work on the top 6 candidates.",
            "Use up to the top 8 candidates as the working set.",
        ]
        focus_criteria = {"POOL": "Operate on the strongest subset as a group rather than one specific candidate."}
        for item in top:
            cid = str(item.get("id", ""))
            focus_criteria[cid] = f"Focus primarily on candidate {cid}."

        questions = {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "Choose the single next search action that maximizes expected answer quality per remaining compute. "
                    "Choose STOP as soon as more search is unlikely to materially improve the answer."
                ),
                "criteria": action_criteria,
            },
            "refill_amount": {
                "type": "score",
                "instructions": "If the chosen action creates new candidates, how many new candidates are useful in this step?",
                "criteria": refill_criteria,
            },
            "target_span": {
                "type": "score",
                "instructions": "How many of the top current candidates should the next operation condition on?",
                "criteria": span_criteria,
            },
            "focus_candidate": {
                "type": "choice",
                "instructions": "Select the most useful focus candidate, or POOL when the operation should work across several survivors.",
                "criteria": focus_criteria,
            },
            "ready_to_stop": {
                "type": "noul",
                "instructions": "Is the current candidate pool already sufficient to finalize without material expected gain from more search?",
                "true_when": "A strong candidate exists and further search is mostly redundant or low-value.",
                "false_when": "Important uncertainty, missing alternatives, conflicts, or verification gaps remain.",
            },
        }
        data = self.decide(state, questions)
        answers = data["answers"]
        action = str(answers.get("next_action", {}).get("choice", "STOP")).upper()
        if action not in action_criteria:
            action = "STOP"
        refill_levels = [2, 4, 6, 8, 12, 16]
        span_levels = [1, 2, 3, 4, 6, 8]
        refill_count = min(max_refill, refill_levels[score_level(answers.get("refill_amount", {}), len(refill_levels))])
        target_span = span_levels[score_level(answers.get("target_span", {}), len(span_levels))]
        focus_id = str(answers.get("focus_candidate", {}).get("choice", "POOL"))
        if focus_id not in focus_criteria:
            focus_id = "POOL"
        return {
            "action": action,
            "refill_count": refill_count,
            "target_span": target_span,
            "focus_id": focus_id,
            "ready_to_stop": noul_probability(answers.get("ready_to_stop", {}), 0.5),
            "action_confidence": answer_confidence(answers.get("next_action", {})),
            "raw": data,
        }

    def verify_candidate_batch(
        self,
        *,
        user_text: str,
        plan: dict[str, Any],
        candidates: list[str],
        round_index: int,
    ) -> list[dict[str, Any]]:
        """A stricter Jev-only verification pass; it does not request new Solar candidates."""
        if not candidates:
            return []
        records = [{
            "id": f"V{i}",
            "record": json.dumps({"candidate": text, "user_request": user_text, "solar_plan": plan}, ensure_ascii=False),
        } for i, text in enumerate(candidates)]
        state = {
            "description": f"Strict verification pass {round_index + 1}. Search for failure modes in the current strongest candidates.",
            "records": records,
        }
        questions: dict[str, Any] = {}
        criteria = {
            "robustness": [
                "Breaks under obvious checking.", "Major unresolved failure modes.", "Several material weaknesses remain.",
                "Mostly robust with manageable caveats.", "Strong under adversarial checking.", "Very robust; no material flaw found in available context.",
            ],
            "support": [
                "Key claims unsupported/invalid.", "Weak support.", "Mixed support.", "Adequately supported.", "Strongly supported.", "Exceptionally well-supported for available context.",
            ],
            "uncertainty": [
                "Very low residual uncertainty.", "Low uncertainty.", "Moderate-low uncertainty.", "Moderate uncertainty.", "High uncertainty.", "Very high uncertainty.",
            ],
        }
        for i, _ in enumerate(candidates):
            cid = f"V{i}"
            for metric, levels in criteria.items():
                questions[f"{cid}__{metric}"] = {
                    "type": "score",
                    "instructions": f"For {cid}, rate {metric} after strict adversarial verification.",
                    "criteria": levels,
                }
            questions[f"{cid}__fatal"] = {
                "type": "noul",
                "instructions": f"Does {cid} contain a fatal or answer-changing flaw?",
                "true_when": "A material error/contradiction/constraint failure would make this route unsafe to select as-is.",
                "false_when": "No fatal issue is found in the available context.",
            }
        data = self.decide(state, questions)
        answers = data["answers"]
        out: list[dict[str, Any]] = []
        for i, _ in enumerate(candidates):
            cid = f"V{i}"
            robustness = score_expectation(answers.get(f"{cid}__robustness", {}), 6)
            support = score_expectation(answers.get(f"{cid}__support", {}), 6)
            uncertainty = score_expectation(answers.get(f"{cid}__uncertainty", {}), 6)
            fatal = noul_probability(answers.get(f"{cid}__fatal", {}), 0.5)
            verification = max(0.0, min(1.0, 0.46 * robustness + 0.38 * support + 0.16 * (1.0 - uncertainty)))
            out.append({
                "verification": verification,
                "fatal_flaw": fatal,
                "uncertainty": uncertainty,
                "metrics": {
                    "verify_robustness": robustness,
                    "verify_support": support,
                    "verify_uncertainty": uncertainty,
                    "verify_fatal": fatal,
                },
            })
        return out

    def _choose(self, *, user_text: str, plan: dict[str, Any], candidates: list[str], label: str, instruction: str) -> tuple[int, dict[str, Any]]:
        if not candidates:
            raise ValueError("cannot choose from zero candidates")
        state = {
            "description": label + " Each record includes the candidate and its request context.",
            "records": [{
                "id": f"C{i}",
                "record": json.dumps({"candidate": c, "user_request": user_text, "solar_plan": plan}, ensure_ascii=False),
            } for i, c in enumerate(candidates)],
        }
        criteria = {
            f"C{i}": f"Choose C{i} when this record is the strongest overall option for the user's request."
            for i in range(len(candidates))
        }
        data = self.decide(state, {
            "winner": {
                "type": "choice",
                "instructions": instruction,
                "criteria": criteria,
            }
        })
        choice = data["answers"].get("winner", {}).get("choice", "C0")
        try:
            idx = int(str(choice).lstrip("C"))
        except ValueError:
            idx = 0
        if not 0 <= idx < len(candidates):
            idx = 0
        return idx, data

    def choose_blueprint(self, *, user_text: str, plan: dict[str, Any], candidates: list[str]) -> tuple[int, dict[str, Any]]:
        return self._choose(
            user_text=user_text, plan=plan, candidates=candidates,
            label="Top candidate answer blueprints after a multi-layer decision network.",
            instruction="Choose the strongest blueprint globally. Prefer correctness, request fit, constraint satisfaction, completeness, and robust logic over clever wording.",
        )

    def choose_final_answer(self, *, user_text: str, plan: dict[str, Any], drafts: list[str]) -> tuple[int, dict[str, Any]]:
        return self._choose(
            user_text=user_text, plan=plan, candidates=drafts,
            label="User-facing final answer drafts generated from the winning blueprint.",
            instruction="Choose the best final answer. Prefer factual/correct content, direct instruction following, clarity, useful completeness, and no unnecessary verbosity.",
        )

    # Legacy controller retained for experiments/comparison.
    def controls(self, *, user_text: str, history: list[dict[str, str]], plan: dict[str, Any], prefix: str, step: int) -> ControlProfile:
        horizon_criteria = [
            "Character scale: exact letters/Hangul/symbols/local spelling dominate the next decision.",
            "Subword/token scale: a tiny morphological or token fragment is the right next prediction unit.",
            "Word scale: choosing the next lexical/code unit is the useful uncertainty boundary.",
            "Phrase scale: choose the next short semantic phrase rather than individual words.",
            "Clause scale: choose a medium semantic clause with local structure resolved together.",
            "Sentence scale: decide the next complete sentence/semantic move as one unit.",
        ]
        mode_criteria = {name: desc for name, desc in zip(COGNITIVE_MODES, [
            "Direct prediction is sufficient.", "Several plausible continuations should be compared.",
            "A small amount of reasoning is needed.", "Multiple constraints need deliberate reasoning.",
            "Substantial ambiguity/correctness risk needs deep reasoning.", "Maximum deliberate reasoning and verification is justified.",
        ])}
        state = {"user_request": user_text, "recent_conversation": history[-6:], "solar_plan": plan, "current_response_prefix": prefix, "guided_step": step}
        questions = {
            "prediction_horizon": {"type": "score", "instructions": "Select next-output granularity.", "criteria": horizon_criteria},
            "cognitive_mode": {"type": "choice", "instructions": "Choose minimum sufficient cognitive depth.", "criteria": mode_criteria},
            "candidate_breadth": {"type": "score", "instructions": "How many candidates?", "criteria": ["1", "2", "3", "4", "6"]},
            "reasoning_scope": {"type": "score", "instructions": "How wide a context field?", "criteria": [d for _, d in REASONING_SCOPES]},
            "response_length": {"type": "score", "instructions": "Useful final answer length.", "criteria": ["micro", "short", "medium", "long", "extended"]},
            "ready_to_finalize": {"type": "noul", "instructions": "Is the scaffold ready for finalization?", "true_when": "The direction is sufficiently fixed for finalization.", "false_when": "Another guided decision would materially improve it."},
        }
        data = self.decide(state, questions); answers = data["answers"]
        hi = score_level(answers.get("prediction_horizon", {}), len(HORIZONS)); horizon = HORIZONS[hi][0]
        ma = answers.get("cognitive_mode", {}); mode = ma.get("choice")
        if mode not in COGNITIVE_MODES:
            probs = ma.get("probabilities") or {}; mode = max(probs.items(), key=lambda kv: float(kv[1] or 0))[0] if probs else "reason_medium"
        bi = score_level(answers.get("candidate_breadth", {}), len(BREADTH_LEVELS)); breadth = BREADTH_LEVELS[bi]
        si = score_level(answers.get("reasoning_scope", {}), len(REASONING_SCOPES)); li = score_level(answers.get("response_length", {}), len(RESPONSE_LENGTHS))
        ready = noul_probability(answers.get("ready_to_finalize", {}), 0.0)
        return ControlProfile(hi, horizon, mode, breadth, si, REASONING_SCOPES[si][0], li, RESPONSE_LENGTHS[li][0], ready, {
            "horizon": answer_confidence(answers.get("prediction_horizon", {})), "mode": answer_confidence(ma),
            "breadth": answer_confidence(answers.get("candidate_breadth", {})), "scope": answer_confidence(answers.get("reasoning_scope", {})),
            "length": answer_confidence(answers.get("response_length", {})),
        }, data)

    def choose_candidate(self, *, user_text: str, plan: dict[str, Any], prefix: str, candidates: list[str], horizon: str, reasoning_scope: str = "whole_answer") -> tuple[int, dict[str, Any]]:
        return self._choose(user_text=user_text, plan=plan, candidates=candidates, label="Literal next fragments", instruction="Choose the best literal next fragment.")

    def doctor(self) -> dict[str, Any]:
        return self.decide(
            {"description": "Connection test", "records": [{"id": "test", "record": "hello"}]},
            {"route": {"type": "choice", "instructions": "Choose ok for this valid harmless connection test.", "criteria": {"ok": "healthy", "not_ok": "not healthy"}}},
        )
