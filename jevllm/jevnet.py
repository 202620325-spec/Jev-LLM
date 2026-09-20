from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .jev import JevClient
from .solar import SolarClient


EventSink = Callable[[str, dict[str, Any]], None]
ExternalVerifier = Callable[[str, dict[str, Any], list[dict[str, str]]], dict[str, Any]]

SEARCH_ACTIONS = (
    "STOP",
    "REFILL",
    "DIVERSE_REFILL",
    "DEEPEN",
    "MUTATE",
    "MERGE",
    "CHALLENGE",
    "VERIFY",
)


@dataclass
class SearchProfile:
    """Hard compute envelope. Jev decides how much of it to actually spend."""

    name: str
    initial_seed: int
    max_rounds: int
    max_live_pool: int
    max_generated: int
    max_refill: int
    final_drafts: int
    response_length: str = "medium"
    reasoning_effort: str = "high"


@dataclass
class CandidateNode:
    id: str
    text: str
    generation: int = 0
    source: str = "solar"
    activation: float = 0.5
    survival: float = 0.5
    uncertainty: float = 0.5
    metrics: dict[str, float] = field(default_factory=dict)
    layer_history: list[dict[str, float]] = field(default_factory=list)
    evaluated: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)


# These are ceilings, not fixed search sizes.
FIXED_PROFILES = {
    "fast": SearchProfile(
        "fast", initial_seed=4, max_rounds=3, max_live_pool=10,
        max_generated=18, max_refill=4, final_drafts=2,
        response_length="short", reasoning_effort="medium",
    ),
    "full": SearchProfile(
        "full", initial_seed=6, max_rounds=6, max_live_pool=20,
        max_generated=42, max_refill=8, final_drafts=3,
        response_length="medium", reasoning_effort="high",
    ),
    "max": SearchProfile(
        "max", initial_seed=8, max_rounds=10, max_live_pool=40,
        max_generated=90, max_refill=16, final_drafts=5,
        response_length="medium", reasoning_effort="high",
    ),
}


class JevDecisionNetwork:
    """Adaptive inference-time decision network.

    Solar proposes compact semantic blueprints. Jev evaluates the live pool and,
    crucially, decides what computation happens next: stop, refill, diversify,
    deepen, mutate, merge, challenge, or verify. Candidate count therefore grows
    or contracts dynamically instead of following a fixed width/layer schedule.

    The hard profile is only an external safety/latency budget. It does not tell
    Jev how many candidates it *must* use.
    """

    def __init__(
        self,
        config: Config,
        jev: JevClient,
        solar: SolarClient,
        event_sink: EventSink | None = None,
        verifier: ExternalVerifier | None = None,
    ):
        self.config = config
        self.jev = jev
        self.solar = solar
        self.emit = event_sink or (lambda _event, _data: None)
        self.verifier = verifier
        self.last_stats: dict[str, Any] = {}
        self._next_node_id = 0

    @staticmethod
    def _profile_from_auto(raw: dict[str, Any], response_length: str) -> SearchProfile:
        # Jev's initial budget estimate only chooses an envelope. The adaptive
        # action loop still decides whether to consume that envelope.
        seeds = [3, 4, 5, 6, 8, 10]
        rounds = [2, 3, 4, 6, 8, 10]
        pools = [8, 10, 14, 20, 28, 40]
        generated = [12, 18, 28, 42, 64, 90]
        refills = [2, 4, 6, 8, 12, 16]
        drafts = [1, 2, 2, 3, 4, 5]

        wi = max(0, min(5, int(raw.get("width", 2))))
        li = max(0, min(5, int(raw.get("layers", 2))))
        mi = max(0, min(5, int(raw.get("mutation", 2))))
        di = max(0, min(5, int(raw.get("drafts", 2))))
        effort = "low" if li <= 1 else "medium" if li <= 3 else "high"
        return SearchProfile(
            name="auto",
            initial_seed=seeds[wi],
            max_rounds=rounds[li],
            max_live_pool=pools[wi],
            max_generated=generated[max(wi, mi)],
            max_refill=refills[mi],
            final_drafts=drafts[di],
            response_length=response_length,
            reasoning_effort=effort,
        )

    def _new_nodes(self, texts: list[str], *, generation: int, source: str) -> list[CandidateNode]:
        out: list[CandidateNode] = []
        for text in texts:
            node = CandidateNode(
                id=f"N{self._next_node_id}",
                text=text,
                generation=generation,
                source=source,
            )
            self._next_node_id += 1
            out.append(node)
        return out

    @staticmethod
    def _dedupe(nodes: list[CandidateNode]) -> list[CandidateNode]:
        seen: set[str] = set()
        out: list[CandidateNode] = []
        for node in nodes:
            key = " ".join(node.text.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(node)
        return out

    @staticmethod
    def _lexical_similarity(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _diversity_adjust(self, nodes: list[CandidateNode]) -> None:
        ranked = sorted(nodes, key=lambda n: n.activation, reverse=True)
        accepted: list[CandidateNode] = []
        for node in ranked:
            max_sim = max((self._lexical_similarity(node.text, other.text) for other in accepted), default=0.0)
            if max_sim > 0.86:
                node.activation *= 0.86
            elif max_sim > 0.72:
                node.activation *= 0.94
            accepted.append(node)

    def _competitive_filter(
        self,
        nodes: list[CandidateNode],
        max_live_pool: int,
        protected_ids: set[str] | None = None,
    ) -> list[CandidateNode]:
        """Competitive inhibition while preserving unresolved contradictory rivals."""
        if not nodes:
            return []
        protected_ids = protected_ids or set()
        self._diversity_adjust(nodes)
        ranked = sorted(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty), reverse=True)
        top = ranked[0].activation
        floor = max(0.36, top - 0.34)
        kept = [
            n for n in ranked
            if n.id in protected_ids or n.activation >= floor or n.survival >= 0.58
        ]
        minimum = min(3, len(ranked))
        if len(kept) < minimum:
            kept_ids = {n.id for n in kept}
            kept.extend(n for n in ranked if n.id not in kept_ids and len(kept) < minimum)

        if len(kept) <= max_live_pool:
            return kept

        protected = [n for n in kept if n.id in protected_ids][:max_live_pool]
        protected_set = {n.id for n in protected}
        rest = [n for n in kept if n.id not in protected_set]
        return protected + rest[: max(0, max_live_pool - len(protected))]

    @staticmethod
    def _apply_evaluations(nodes: list[CandidateNode], evaluations: list[dict[str, Any]], round_index: int) -> None:
        for node, ev in zip(nodes, evaluations):
            layer_score = float(ev.get("activation", 0.5))
            survival = float(ev.get("survival", 0.5))
            uncertainty = float(ev.get("uncertainty", 0.5))
            # Residual state: later evaluations can correct a path without erasing
            # all earlier evidence in one step.
            node.activation = 0.38 * node.activation + 0.62 * layer_score
            node.survival = 0.35 * node.survival + 0.65 * survival
            node.uncertainty = 0.35 * node.uncertainty + 0.65 * uncertainty
            node.metrics.update({k: float(v) for k, v in ev.get("metrics", {}).items()})
            node.evaluated = True
            node.layer_history.append({
                "round": float(round_index),
                "activation": node.activation,
                "survival": node.survival,
                "uncertainty": node.uncertainty,
            })

    @staticmethod
    def _focus_parents(nodes: list[CandidateNode], focus_id: str | None, target_span: int) -> list[CandidateNode]:
        ranked = sorted(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty), reverse=True)
        if focus_id and focus_id != "POOL":
            focused = next((n for n in ranked if n.id == focus_id), None)
            if focused is not None:
                rest = [n for n in ranked if n.id != focus_id]
                return [focused, *rest[: max(0, target_span - 1)]]
        return ranked[:target_span]

    @staticmethod
    def _pool_fingerprint(nodes: list[CandidateNode]) -> tuple[tuple[str, str], ...]:
        ranked = sorted(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty), reverse=True)
        return tuple((n.id, " ".join(n.text.split())) for n in ranked[:8])

    @staticmethod
    def _conflict_key(disagreement: dict[str, Any]) -> tuple[str, str] | None:
        leader = disagreement.get("leader_id")
        rival = disagreement.get("rival_id")
        if not leader or not rival or leader == rival:
            return None
        return tuple(sorted((str(leader), str(rival))))

    @staticmethod
    def _apply_evidence_report(nodes: list[CandidateNode], report: dict[str, Any]) -> None:
        by_id = {node.id: node for node in nodes}
        for item in report.get("results") or []:
            if not isinstance(item, dict):
                continue
            node = by_id.get(str(item.get("candidate_id", "")))
            if node is None:
                continue
            verdict = str(item.get("verdict", "UNCERTAIN")).upper()
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            evidence = {
                "verdict": verdict,
                "confidence": confidence,
                "evidence": str(item.get("evidence", "")),
                "test": str(report.get("test", "")),
                "test_kind": str(report.get("test_kind", "INCONCLUSIVE")),
            }
            node.evidence.append(evidence)

            if verdict == "FAIL":
                node.metrics["evidence_fail"] = max(node.metrics.get("evidence_fail", 0.0), confidence)
                # Concrete falsification must dominate semantic plausibility.
                node.activation *= max(0.08, 1.0 - 0.90 * confidence)
                node.survival *= max(0.05, 1.0 - 0.92 * confidence)
                node.uncertainty = max(node.uncertainty, 0.55 + 0.45 * confidence)
            elif verdict == "PASS":
                node.metrics["evidence_pass"] = max(node.metrics.get("evidence_pass", 0.0), confidence)
                # Passing one test is positive evidence, not a proof of the whole answer.
                node.activation = min(1.0, node.activation + 0.10 * confidence)
                node.survival = min(1.0, node.survival + 0.12 * confidence)
                node.uncertainty = max(0.0, node.uncertainty - 0.12 * confidence)
            else:
                node.metrics["evidence_uncertain"] = max(node.metrics.get("evidence_uncertain", 0.0), confidence)

        winner_id = report.get("winner_id") if report.get("resolved") else None
        winner = by_id.get(str(winner_id)) if winner_id is not None else None
        if winner is not None:
            winner.metrics["evidence_resolved_winner"] = 1.0
            winner.activation = max(winner.activation, 0.82)
            winner.survival = max(winner.survival, 0.84)
            winner.uncertainty = min(winner.uncertainty, 0.18)

    def _run_evidence_test(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        candidates: list[CandidateNode],
        reasoning_effort: str,
    ) -> dict[str, Any]:
        payload = [{"id": n.id, "text": n.text} for n in candidates]
        if self.verifier is not None:
            report = self.verifier(user_text, plan, payload)
            if isinstance(report, dict):
                report = dict(report)
                report.setdefault("source", "external")
                return report

        if hasattr(self.solar, "discriminate_hypotheses"):
            report = self.solar.discriminate_hypotheses(
                user_text=user_text,
                history=history,
                plan=plan,
                candidates=payload,
                reasoning_effort=reasoning_effort,
            )
            report = dict(report)
            report.setdefault("source", "solar_evidence")
            return report

        # Backward-compatible fallback for tests/custom clients. Production uses
        # the evidence verifier above.
        if hasattr(self.jev, "verify_candidate_batch"):
            verified = self.jev.verify_candidate_batch(
                user_text=user_text,
                plan=plan,
                candidates=[n.text for n in candidates],
                round_index=0,
            )
            results = []
            for node, ev in zip(candidates, verified):
                fatal = float(ev.get("fatal_flaw", 0.5))
                results.append({
                    "candidate_id": node.id,
                    "verdict": "FAIL" if fatal >= 0.75 else "UNCERTAIN",
                    "confidence": max(fatal, 1.0 - fatal),
                    "evidence": "Legacy semantic verifier; no deterministic evidence available.",
                })
            return {
                "material_disagreement": False,
                "test": "legacy semantic verification",
                "test_kind": "INCONCLUSIVE",
                "resolved": False,
                "winner_id": None,
                "results": results,
                "source": "legacy_jev",
            }

        return {
            "material_disagreement": False,
            "test": "No verifier available.",
            "test_kind": "INCONCLUSIVE",
            "resolved": False,
            "winner_id": None,
            "results": [],
            "source": "none",
        }

    def run(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        plan: dict[str, Any],
        intensity: str = "auto",
        width_override: int | None = None,
        layers_override: int | None = None,
        drafts_override: int | None = None,
        mutation_override: int | None = None,
        seed_override: int | None = None,
        generated_override: int | None = None,
    ) -> str:
        self._next_node_id = 0
        if intensity == "auto":
            budget = self.jev.network_budget(user_text=user_text, history=history, plan=plan)
            profile = self._profile_from_auto(budget, budget.get("response_length_name", "medium"))
        else:
            profile = SearchProfile(**vars(FIXED_PROFILES.get(intensity, FIXED_PROFILES["full"])))
            budget = {"source": "fixed_cap"}

        if width_override is not None:
            profile.max_live_pool = max(4, min(64, width_override))
        if layers_override is not None:
            profile.max_rounds = max(1, min(16, layers_override))
        if drafts_override is not None:
            profile.final_drafts = max(1, min(8, drafts_override))
        if mutation_override is not None:
            profile.max_refill = max(1, min(24, mutation_override))
        if seed_override is not None:
            profile.initial_seed = max(2, min(24, seed_override))
        if generated_override is not None:
            profile.max_generated = max(profile.initial_seed, min(160, generated_override))

        profile.initial_seed = min(profile.initial_seed, profile.max_live_pool, profile.max_generated)
        profile.max_refill = min(profile.max_refill, profile.max_generated)
        self.emit("network_profile", {"profile": vars(profile), "budget": budget})

        initial = self.solar.expand_reasoning_paths(
            user_text=user_text,
            history=history,
            plan=plan,
            count=profile.initial_seed,
            stage="adaptive seed population",
            reasoning_effort=profile.reasoning_effort,
        )
        nodes = self._dedupe(self._new_nodes(initial, generation=0, source="SEED"))
        if not nodes:
            raise RuntimeError("JevNet received zero initial hypotheses")

        generated_total = len(nodes)
        action_history: list[dict[str, Any]] = []
        evidence_reports: list[dict[str, Any]] = []
        verified_pool_fingerprints: set[tuple[tuple[str, str], ...]] = set()
        resolved_conflicts: dict[tuple[str, str], str | None] = {}
        disagreement_cache: dict[str, Any] = {}
        disagreement_fingerprint: tuple[tuple[str, str], ...] | None = None
        protected_ids: set[str] = set()
        stop_reason = "round_cap"
        evaluated_candidates = 0
        reused_evaluations = 0
        disagreement_checks = 0

        self.emit("expansion", {"generation": 0, "nodes": [vars(n) for n in nodes]})

        for round_index in range(profile.max_rounds):
            # Critical anti-sink change: unchanged hypotheses are NOT semantically
            # rescored every round. Only new candidates consume evaluation compute.
            pending = [n for n in nodes if not n.evaluated]
            if pending:
                evaluations = self.jev.evaluate_candidate_batch(
                    user_text=user_text,
                    plan=plan,
                    candidates=[n.text for n in pending],
                    layer_index=round_index,
                    total_layers=profile.max_rounds,
                    batch_size=self.config.jev_batch_size,
                )
                self._apply_evaluations(pending, evaluations, round_index)
                evaluated_candidates += len(pending)
            reused_evaluations += max(0, len(nodes) - len(pending))

            ranked_for_conflict = sorted(
                nodes,
                key=lambda n: (n.activation, n.survival, -n.uncertainty),
                reverse=True,
            )
            current_fingerprint = self._pool_fingerprint(ranked_for_conflict)

            # Recompute disagreement only when the hypothesis texts actually change.
            if current_fingerprint != disagreement_fingerprint:
                if hasattr(self.jev, "assess_disagreement") and len(ranked_for_conflict) >= 2:
                    disagreement_cache = self.jev.assess_disagreement(
                        user_text=user_text,
                        plan=plan,
                        candidates=[{
                            "id": n.id,
                            "text": n.text,
                            "activation": n.activation,
                            "survival": n.survival,
                            "uncertainty": n.uncertainty,
                            "metrics": n.metrics,
                        } for n in ranked_for_conflict[:8]],
                    )
                    disagreement_checks += 1
                else:
                    disagreement_cache = {
                        "material_disagreement": 0.0,
                        "needs_test": 0.0,
                        "leader_id": ranked_for_conflict[0].id if ranked_for_conflict else None,
                        "rival_id": None,
                    }
                disagreement_fingerprint = current_fingerprint
                self.emit("disagreement", {"round": round_index + 1, "assessment": disagreement_cache})

            conflict_key = self._conflict_key(disagreement_cache)
            material_conflict = (
                conflict_key is not None
                and float(disagreement_cache.get("material_disagreement", 0.0)) >= 0.55
                and float(disagreement_cache.get("needs_test", 0.0)) >= 0.50
                and conflict_key not in resolved_conflicts
            )
            protected_ids = set(conflict_key or ()) if material_conflict else set()

            before = len(nodes)
            nodes = self._competitive_filter(nodes, profile.max_live_pool, protected_ids=protected_ids)
            self.emit("layer", {
                "layer": round_index + 1,
                "before": before,
                "after": len(nodes),
                "pending_evaluated": len(pending),
                "protected_ids": sorted(protected_ids),
                "nodes": [vars(n) for n in nodes],
            })

            remaining_generated = max(0, profile.max_generated - generated_total)
            rounds_left = profile.max_rounds - (round_index + 1)
            current_fingerprint = self._pool_fingerprint(nodes)
            pool_already_verified = current_fingerprint in verified_pool_fingerprints

            allowed_actions: list[str] = ["STOP"]
            if rounds_left > 0:
                allowed_actions.append("VERIFY")
                if remaining_generated > 0:
                    allowed_actions.extend([
                        "REFILL", "DIVERSE_REFILL", "DEEPEN", "MUTATE", "MERGE", "CHALLENGE"
                    ])

            # COLLAPSE gate: unresolved contradictory answers cannot STOP merely
            # because one has a higher semantic score.
            if material_conflict:
                allowed_actions = [a for a in allowed_actions if a != "STOP"]
                if pool_already_verified:
                    # No confidence loops: the same unchanged pool may be evidence-
                    # verified at most once.
                    allowed_actions = [a for a in allowed_actions if a != "VERIFY"]
                elif "VERIFY" not in allowed_actions:
                    allowed_actions.insert(0, "VERIFY")

                if not allowed_actions:
                    # At the hard round cap, spend the last step on evidence rather
                    # than silently collapsing an unresolved contradiction.
                    allowed_actions = ["VERIFY"] if not pool_already_verified else ["STOP"]

            decision = self.jev.search_action(
                user_text=user_text,
                plan=plan,
                candidates=[{
                    "id": n.id,
                    "text": n.text,
                    "activation": n.activation,
                    "survival": n.survival,
                    "uncertainty": n.uncertainty,
                    "metrics": n.metrics,
                    "evidence": n.evidence[-2:],
                } for n in nodes],
                round_index=round_index,
                max_rounds=profile.max_rounds,
                generated_total=generated_total,
                max_generated=profile.max_generated,
                max_live_pool=profile.max_live_pool,
                max_refill=profile.max_refill,
                allowed_actions=allowed_actions,
            )
            requested_action = str(decision.get("action", "STOP")).upper()
            action = requested_action
            if action not in allowed_actions:
                if material_conflict and not pool_already_verified and "VERIFY" in allowed_actions:
                    action = "VERIFY"
                elif material_conflict and "CHALLENGE" in allowed_actions:
                    action = "CHALLENGE"
                else:
                    action = allowed_actions[0] if allowed_actions else "STOP"

            # Global loop breaker even when disagreement detection misses a conflict.
            if action == "VERIFY" and pool_already_verified:
                if "CHALLENGE" in allowed_actions:
                    action = "CHALLENGE"
                elif "DIVERSE_REFILL" in allowed_actions:
                    action = "DIVERSE_REFILL"
                else:
                    action = "STOP"

            decision["jev_action"] = requested_action
            decision["action"] = action
            decision["material_conflict"] = material_conflict
            decision["conflict_key"] = list(conflict_key) if conflict_key else None
            decision["pool_already_verified"] = pool_already_verified
            action_history.append({k: v for k, v in decision.items() if k != "raw"})
            self.emit("search_action", {
                "round": round_index + 1,
                "decision": decision,
                "pool_size": len(nodes),
                "generated_total": generated_total,
                "remaining_generated": remaining_generated,
            })

            if action == "STOP":
                stop_reason = "jev_stop"
                break

            target_span = max(1, int(decision.get("target_span", 3)))
            parents = self._focus_parents(nodes, decision.get("focus_id"), target_span)

            if action == "VERIFY":
                # If a contradiction pair is known, always include BOTH sides.
                verify_nodes: list[CandidateNode] = []
                by_id = {n.id: n for n in nodes}
                if conflict_key:
                    for cid in conflict_key:
                        node = by_id.get(cid)
                        if node is not None and node not in verify_nodes:
                            verify_nodes.append(node)
                for node in parents:
                    if node not in verify_nodes:
                        verify_nodes.append(node)
                    if len(verify_nodes) >= 6:
                        break
                verify_nodes = verify_nodes[:6]

                report = self._run_evidence_test(
                    user_text=user_text,
                    history=history,
                    plan=plan,
                    candidates=verify_nodes,
                    reasoning_effort=profile.reasoning_effort,
                )
                verified_pool_fingerprints.add(current_fingerprint)
                evidence_reports.append({
                    "round": round_index + 1,
                    "candidate_ids": [n.id for n in verify_nodes],
                    "source": report.get("source"),
                    "test": report.get("test"),
                    "test_kind": report.get("test_kind"),
                    "resolved": bool(report.get("resolved")),
                    "winner_id": report.get("winner_id"),
                    "results": report.get("results") or [],
                })
                self._apply_evidence_report(nodes, report)
                if conflict_key and report.get("resolved"):
                    resolved_conflicts[conflict_key] = str(report.get("winner_id") or "") or None
                    protected_ids.clear()

                self.emit("verification", {
                    "round": round_index + 1,
                    "mode": "evidence",
                    "report": evidence_reports[-1],
                    "nodes": [vars(n) for n in verify_nodes],
                })
                # Do not re-run semantic evaluation next round; evidence directly
                # updated the candidate state.
                continue

            requested = max(1, int(decision.get("refill_count", 2)))
            requested = min(requested, profile.max_refill, remaining_generated)
            if requested <= 0:
                stop_reason = "generation_cap"
                break

            additions = self.solar.adaptive_reasoning_operation(
                action=action,
                user_text=user_text,
                history=history,
                plan=plan,
                parents=[n.text for n in parents],
                existing=[n.text for n in nodes],
                count=requested,
                round_index=round_index,
                reasoning_effort=profile.reasoning_effort,
            )
            child_nodes = self._new_nodes(additions, generation=round_index + 1, source=action)
            generated_total += len(child_nodes)
            nodes = self._dedupe(nodes + child_nodes)

            # New hypotheses change the disagreement graph, so re-detect next round.
            disagreement_fingerprint = None
            if len(nodes) > profile.max_live_pool * 2:
                protected = [n for n in nodes if n.id in protected_ids]
                protected_set = {n.id for n in protected}
                rest = sorted(
                    [n for n in nodes if n.id not in protected_set],
                    key=lambda n: (n.activation, n.survival, -n.uncertainty),
                    reverse=True,
                )
                nodes = (protected + rest)[: profile.max_live_pool * 2]

            self.emit("adaptive_expansion", {
                "round": round_index + 1,
                "action": action,
                "requested": requested,
                "children": [vars(n) for n in child_nodes],
                "pool_size": len(nodes),
                "generated_total": generated_total,
            })
        else:
            stop_reason = "round_cap"

        if not nodes:
            raise RuntimeError("JevNet candidate pool became empty")

        nodes = sorted(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty), reverse=True)

        # Evidence has veto power over mere semantic polish. Strongly falsified
        # hypotheses do not enter final selection when any non-falsified survivor exists.
        eligible = [n for n in nodes if n.metrics.get("evidence_fail", 0.0) < 0.65]
        if eligible:
            nodes = eligible

        # When a discriminating test actually resolved a conflict, preserve its
        # winner at the front of the final field instead of letting an old semantic
        # score erase the new evidence.
        evidence_winners = [n for n in nodes if n.metrics.get("evidence_resolved_winner", 0.0) >= 1.0]
        if evidence_winners:
            # Once a discriminating test actually resolves the contradiction,
            # do not let a later style/plausibility vote overturn that evidence.
            finalists = evidence_winners[: min(8, len(evidence_winners))]
        else:
            finalists = nodes[: min(8, len(nodes))]

        if len(finalists) == 1 and evidence_winners:
            chosen_plan_idx = 0
            plan_raw = {"selection_source": "resolved_evidence"}
        else:
            chosen_plan_idx, plan_raw = self.jev.choose_blueprint(
                user_text=user_text,
                plan=plan,
                candidates=[n.text for n in finalists],
            )
        chosen_blueprint = finalists[chosen_plan_idx].text
        self.emit("blueprint_selection", {
            "index": chosen_plan_idx,
            "blueprint": chosen_blueprint,
            "finalists": [vars(n) for n in finalists],
            "raw": plan_raw,
        })

        drafts = self.solar.render_answer_drafts(
            user_text=user_text,
            history=history,
            plan=plan,
            chosen_blueprint=chosen_blueprint,
            supporting_blueprints=[n.text for n in finalists[:5]],
            count=profile.final_drafts,
            response_length=profile.response_length,
            reasoning_effort=profile.reasoning_effort,
        )
        if not drafts:
            raise RuntimeError("Solar produced zero final answer drafts")
        surface_render = dict(getattr(self.solar, "last_render_stats", {}) or {})
        self.emit("surface_render", {"stats": surface_render})
        self.emit("final_drafts", {"drafts": drafts})

        if len(drafts) == 1:
            selected = 0
            final_raw: dict[str, Any] = {}
        else:
            selected, final_raw = self.jev.choose_final_answer(
                user_text=user_text,
                plan=plan,
                drafts=drafts,
            )
        answer = drafts[selected]
        self.emit("final_selection", {"index": selected, "raw": final_raw, "answer": answer})

        self.last_stats = {
            "profile": vars(profile),
            "seed_candidates": len(initial),
            "total_generated_candidates": generated_total,
            "live_pool_at_finish": len(nodes),
            "final_blueprints": len(finalists),
            "final_drafts": len(drafts),
            "surface_render": surface_render,
            "adaptive_actions": [d.get("action") for d in action_history],
            "action_history": action_history,
            "stop_reason": stop_reason,
            "evaluated_candidates": evaluated_candidates,
            "reused_evaluations": reused_evaluations,
            "disagreement_checks": disagreement_checks,
            "evidence_tests": len(evidence_reports),
            "evidence_reports": evidence_reports,
            "solar_calls": self.solar.call_count,
            "jev_calls": self.jev.call_count,
        }
        return answer
