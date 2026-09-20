from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .jev import JevClient
from .sanity import hard_sanity_issues
from .solar import SolarClient


EventSink = Callable[[str, dict[str, Any]], None]

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

    def __init__(self, config: Config, jev: JevClient, solar: SolarClient, event_sink: EventSink | None = None):
        self.config = config
        self.jev = jev
        self.solar = solar
        self.emit = event_sink or (lambda _event, _data: None)
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

    def _competitive_filter(self, nodes: list[CandidateNode], max_live_pool: int) -> list[CandidateNode]:
        """Competitive inhibition without a fixed keep-ratio.

        Weak/dominated nodes disappear, but the pool is allowed to stay broad when
        many candidates are genuinely competitive. This keeps the candidate count
        data-dependent rather than layer-dependent.
        """
        if not nodes:
            return []
        self._diversity_adjust(nodes)
        ranked = sorted(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty), reverse=True)
        top = ranked[0].activation
        floor = max(0.36, top - 0.34)
        kept = [n for n in ranked if n.activation >= floor or n.survival >= 0.58]
        minimum = min(3, len(ranked))
        if len(kept) < minimum:
            kept = ranked[:minimum]
        return kept[:max_live_pool]

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

        # Backward-compatible overrides now modify ceilings, not mandatory width/layers.
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

        # Keep caps coherent.
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
        sanity_vetoes = 0
        sanity_flagged_finalists = 0
        sanity_flagged_drafts = 0
        stop_reason = "round_cap"
        self.emit("expansion", {"generation": 0, "nodes": [vars(n) for n in nodes]})

        for round_index in range(profile.max_rounds):
            evaluations = self.jev.evaluate_candidate_batch(
                user_text=user_text,
                plan=plan,
                candidates=[n.text for n in nodes],
                layer_index=round_index,
                total_layers=profile.max_rounds,
                batch_size=self.config.jev_batch_size,
            )
            self._apply_evaluations(nodes, evaluations, round_index)

            before = len(nodes)
            nodes = self._competitive_filter(nodes, profile.max_live_pool)
            self.emit("layer", {
                "layer": round_index + 1,
                "before": before,
                "after": len(nodes),
                "nodes": [vars(n) for n in nodes],
            })

            remaining_generated = max(0, profile.max_generated - generated_total)
            rounds_left = profile.max_rounds - (round_index + 1)
            allowed_actions = ["STOP"]
            if rounds_left > 0:
                allowed_actions.append("VERIFY")
                if remaining_generated > 0:
                    allowed_actions.extend([
                        "REFILL", "DIVERSE_REFILL", "DEEPEN", "MUTATE", "MERGE", "CHALLENGE"
                    ])

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
                } for n in nodes],
                round_index=round_index,
                max_rounds=profile.max_rounds,
                generated_total=generated_total,
                max_generated=profile.max_generated,
                max_live_pool=profile.max_live_pool,
                max_refill=profile.max_refill,
                allowed_actions=allowed_actions,
            )
            action = str(decision.get("action", "STOP")).upper()
            if action not in allowed_actions:
                # Only the external compute envelope may overrule Jev.
                action = "STOP"

            # Deterministic STOP gate. Jev remains the search controller, but it
            # cannot finalize a leader containing a provable arithmetic
            # contradiction such as exact 75% accuracy over 5 discrete items.
            requested_action = action
            decision["jev_action"] = requested_action
            if action == "STOP" and rounds_left > 0 and nodes:
                leader = max(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty))
                issues = hard_sanity_issues(leader.text)
                if issues:
                    forced_action = (
                        "CHALLENGE" if "CHALLENGE" in allowed_actions
                        else "VERIFY" if "VERIFY" in allowed_actions
                        else None
                    )
                    if forced_action is not None:
                        action = forced_action
                        sanity_vetoes += 1
                        decision["sanity_veto"] = True
                        decision["sanity_issues"] = [
                            {"code": issue.code, "message": issue.message, "evidence": issue.evidence}
                            for issue in issues
                        ]
                        decision["focus_id"] = leader.id
                        decision["target_span"] = max(1, int(decision.get("target_span", 2)))
                        if action == "CHALLENGE":
                            decision["refill_count"] = max(2, int(decision.get("refill_count", 2)))
                        self.emit("sanity_gate", {
                            "round": round_index + 1,
                            "candidate_id": leader.id,
                            "requested_action": requested_action,
                            "forced_action": action,
                            "issues": decision["sanity_issues"],
                        })

            decision["action"] = action
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
                verify_nodes = parents[: min(8, len(parents))]
                verified = self.jev.verify_candidate_batch(
                    user_text=user_text,
                    plan=plan,
                    candidates=[n.text for n in verify_nodes],
                    round_index=round_index,
                )
                for node, ev in zip(verify_nodes, verified):
                    vscore = float(ev.get("verification", 0.5))
                    fatal = float(ev.get("fatal_flaw", 0.5))
                    node.activation = max(0.0, min(1.0, 0.58 * node.activation + 0.42 * vscore - 0.18 * fatal))
                    node.survival = max(0.0, min(1.0, 0.65 * node.survival + 0.35 * (1.0 - fatal)))
                    node.uncertainty = max(0.0, min(1.0, 0.65 * node.uncertainty + 0.35 * float(ev.get("uncertainty", node.uncertainty))))
                    node.metrics.update({k: float(v) for k, v in ev.get("metrics", {}).items()})
                self.emit("verification", {"round": round_index + 1, "nodes": [vars(n) for n in verify_nodes]})
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
            # Live-pool cap is a memory/latency guard. New nodes are allowed to
            # compete at the next evaluation, then inhibition will compress them.
            if len(nodes) > profile.max_live_pool * 2:
                nodes = sorted(nodes, key=lambda n: (n.activation, n.survival, -n.uncertainty), reverse=True)[: profile.max_live_pool * 2]
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

        # Do not hand a provably inconsistent candidate to the final selector when
        # at least one clean alternative survived.
        clean_nodes: list[CandidateNode] = []
        flagged_nodes: list[tuple[CandidateNode, list[Any]]] = []
        for node in nodes:
            issues = hard_sanity_issues(node.text)
            if issues:
                flagged_nodes.append((node, issues))
            else:
                clean_nodes.append(node)
        sanity_flagged_finalists = len(flagged_nodes)
        selection_nodes = clean_nodes if clean_nodes else nodes
        finalists = selection_nodes[: min(8, len(selection_nodes))]

        if flagged_nodes:
            self.emit("sanity_finalists", {
                "flagged": [
                    {
                        "candidate_id": node.id,
                        "issues": [
                            {"code": issue.code, "message": issue.message, "evidence": issue.evidence}
                            for issue in issues
                        ],
                    }
                    for node, issues in flagged_nodes[:8]
                ],
                "clean_available": bool(clean_nodes),
            })

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
        self.emit("final_drafts", {"drafts": drafts})

        clean_draft_pairs: list[tuple[int, str]] = []
        draft_issues: list[tuple[int, list[Any]]] = []
        for i, draft in enumerate(drafts):
            issues = hard_sanity_issues(draft)
            if issues:
                draft_issues.append((i, issues))
            else:
                clean_draft_pairs.append((i, draft))
        sanity_flagged_drafts = len(draft_issues)

        if draft_issues:
            self.emit("sanity_drafts", {
                "flagged": [
                    {
                        "index": i,
                        "issues": [
                            {"code": issue.code, "message": issue.message, "evidence": issue.evidence}
                            for issue in issues
                        ],
                    }
                    for i, issues in draft_issues
                ],
                "clean_available": bool(clean_draft_pairs),
            })

        if clean_draft_pairs:
            clean_drafts = [draft for _, draft in clean_draft_pairs]
            if len(clean_drafts) == 1:
                clean_selected = 0
                final_raw: dict[str, Any] = {"sanity_filtered": sanity_flagged_drafts}
            else:
                clean_selected, final_raw = self.jev.choose_final_answer(
                    user_text=user_text,
                    plan=plan,
                    drafts=clean_drafts,
                )
            selected = clean_draft_pairs[clean_selected][0]
            answer = drafts[selected]
        elif not hard_sanity_issues(chosen_blueprint):
            # Surface rendering introduced a deterministic contradiction into
            # every draft. Prefer the already-selected clean blueprint to knowingly
            # returning impossible arithmetic.
            selected = -1
            final_raw = {"sanity_fallback": "chosen_blueprint"}
            answer = chosen_blueprint
        else:
            # No deterministic clean alternative exists within the compute cap.
            # Preserve previous behavior rather than fabricating a repair.
            if len(drafts) == 1:
                selected = 0
                final_raw = {"sanity_exhausted": True}
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
            "adaptive_actions": [d.get("action") for d in action_history],
            "action_history": action_history,
            "sanity_vetoes": sanity_vetoes,
            "sanity_flagged_finalists": sanity_flagged_finalists,
            "sanity_flagged_drafts": sanity_flagged_drafts,
            "stop_reason": stop_reason,
            "solar_calls": self.solar.call_count,
            "jev_calls": self.jev.call_count,
        }
        return answer
