import unittest

from jevllm.config import Config
from jevllm.controller import AdaptiveController
from jevllm.engine import JevLLM
from jevllm.jevnet import JevDecisionNetwork
from jevllm.sanity import hard_sanity_issues
from jevllm.solar import SolarClient
from jevllm.types import ControlProfile, SolarResult
from jevllm.util import extract_json, recover_candidate_strings, score_expectation, score_level


class CoreTests(unittest.TestCase):
    def test_score_uses_probability_mode_not_expected_score_rounding(self):
        answer = {"score": 2.49, "probabilities": {"1": 0.10, "2": 0.30, "3": 0.60}}
        self.assertEqual(score_level(answer, 6), 3)

    def test_extract_json_from_fence(self):
        self.assertEqual(extract_json('```json\n{"a":1}\n```'), {"a": 1})

    def test_recover_object_candidates(self):
        text = '{"candidates":[{"text":"alpha"},{"content":"beta"}]}'
        self.assertEqual(recover_candidate_strings(text, 4), ["alpha", "beta"])

    def test_recover_labelled_candidates(self):
        text = "C0: alpha\nC1: beta"
        self.assertEqual(recover_candidate_strings(text, 4), ["alpha", "beta"])

    def test_adaptive_predict_collapses_to_one_candidate(self):
        controller = AdaptiveController(run_mode="fast", fast_base_steps=1, full_base_steps=3, max_steps=6)
        p = ControlProfile(
            5, "sentence", "predict", 6,
            reasoning_scope_index=0, reasoning_scope="local",
            response_length_index=0, response_length="micro",
            confidence={"horizon": 0.95, "mode": 0.95, "breadth": 0.95, "scope": 0.95, "length": 0.95},
        )
        a = controller.update(p)
        self.assertEqual(a.cognitive_mode, "predict")
        self.assertEqual(a.breadth, 1)
        self.assertEqual(a.response_length, "micro")

    def test_adaptive_extreme_has_wide_search(self):
        controller = AdaptiveController(run_mode="full", fast_base_steps=1, full_base_steps=3, max_steps=6)
        p = ControlProfile(
            2, "word", "reason_extreme", 1,
            reasoning_scope_index=5, reasoning_scope="multi_angle",
            response_length_index=4, response_length="extended",
            confidence={"horizon": 0.4, "mode": 0.4, "breadth": 0.4, "scope": 0.4, "length": 0.4},
        )
        a = controller.update(p)
        self.assertIn(a.cognitive_mode, {"reason_high", "reason_extreme"})
        self.assertGreaterEqual(a.breadth, 4)
        self.assertGreaterEqual(a.desired_guided_steps, 4)
        self.assertGreater(a.deliberation_budget, 0.6)

    def test_single_candidate_path_requires_no_json(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append(messages)
            return SolarResult(text="direct continuation")

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.propose_continuations(
            user_text="q", history=[], plan={}, prefix="", horizon="sentence",
            horizon_rule="one sentence", cognitive_mode="predict", breadth=1,
            reasoning_effort="low", reasoning_scope="local", response_length="short",
        )
        self.assertEqual(out, ["direct continuation"])
        self.assertEqual(len(calls), 1)
        self.assertIn("ONE literal next response continuation", calls[0][0]["content"])

    def test_malformed_multi_candidate_falls_back_to_single(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        replies = iter([SolarResult(text="this is not a labelled candidate list"), SolarResult(text="fallback")])

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            return next(replies)

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.propose_continuations(
            user_text="q", history=[], plan={}, prefix="", horizon="phrase",
            horizon_rule="short phrase", cognitive_mode="options", breadth=3,
            reasoning_effort="low", reasoning_scope="subproblem", response_length="medium",
        )
        self.assertEqual(out, ["fallback"])

    def test_fragment_join_repairs_lost_space(self):
        self.assertEqual(JevLLM._append_fragment("hello", "world", "word"), "hello world")
        self.assertEqual(JevLLM._append_fragment("hel", "lo", "subword"), "hello")

    def test_engine_offline_integration(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y", default_run_mode="fast")
        engine = JevLLM(config)
        engine.pipeline_mode = "legacy"

        class FakeJev:
            def controls(self, **kwargs):
                return ControlProfile(
                    5, "sentence", "predict", 1,
                    reasoning_scope_index=1, reasoning_scope="sentence",
                    response_length_index=1, response_length="short",
                    ready_to_finalize=0.95,
                    confidence={"horizon": 0.9, "mode": 0.9, "breadth": 0.9, "scope": 0.9, "length": 0.9},
                )

            def choose_candidate(self, **kwargs):
                return 0, {}

        class FakeSolar:
            def __init__(self):
                self.final_kwargs = None

            def plan(self, user_text, history):
                return {"intent": "test", "route": ["answer"], "required_points": []}

            def propose_continuations(self, **kwargs):
                return ["Hello"]

            def finalize(self, **kwargs):
                self.final_kwargs = kwargs
                return "Hello world"

        fake_solar = FakeSolar()
        engine.jev = FakeJev()
        engine.solar = fake_solar
        answer = engine.answer("test")
        self.assertEqual(answer, "Hello world")
        self.assertIsNotNone(fake_solar.final_kwargs)
        self.assertIn(fake_solar.final_kwargs["response_length"], {"micro", "short", "medium"})
        self.assertGreaterEqual(fake_solar.final_kwargs["response_max_tokens"], 512)


    def test_score_expectation_uses_probability_distribution(self):
        answer = {"score": 0, "probabilities": {"0": 0.1, "5": 0.9}}
        self.assertAlmostEqual(score_expectation(answer, 6), 0.9, places=3)

    def test_jevnet_stops_without_forced_refill(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        events = []

        class FakeJev:
            def __init__(self): self.call_count = 0
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                return [{"activation": 0.85 - i * 0.03, "survival": 0.9, "uncertainty": 0.1, "metrics": {"fit": 0.9}} for i, _ in enumerate(candidates)]
            def search_action(self, **kwargs):
                self.call_count += 1
                return {"action": "STOP", "refill_count": 16, "target_span": 3, "focus_id": "POOL", "ready_to_stop": 0.95}
            def choose_blueprint(self, *, candidates, **kwargs): self.call_count += 1; return 0, {}
            def choose_final_answer(self, *, drafts, **kwargs): self.call_count += 1; return 0, {}

        class FakeSolar:
            def __init__(self): self.call_count = 0; self.adaptive_calls = 0
            def expand_reasoning_paths(self, *, count, **kwargs):
                self.call_count += 1; return [f"seed {i}" for i in range(count)]
            def adaptive_reasoning_operation(self, **kwargs):
                self.call_count += 1; self.adaptive_calls += 1; return ["should not happen"]
            def render_answer_drafts(self, *, count, **kwargs):
                self.call_count += 1; return [f"draft {i}" for i in range(count)]

        jev = FakeJev(); solar = FakeSolar()
        net = JevDecisionNetwork(config, jev, solar, lambda e, d: events.append((e, d)))
        answer = net.run(user_text="q", history=[], plan={"route": ["a"]}, intensity="max")
        self.assertEqual(answer, "draft 0")
        self.assertEqual(solar.adaptive_calls, 0)
        self.assertEqual(net.last_stats["total_generated_candidates"], 8)
        self.assertEqual(net.last_stats["adaptive_actions"], ["STOP"])
        self.assertEqual(net.last_stats["stop_reason"], "jev_stop")

    def test_jevnet_refill_then_stop(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        events = []

        class FakeJev:
            def __init__(self): self.call_count = 0; self.actions = iter(["DIVERSE_REFILL", "STOP"])
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                return [{"activation": 0.65 + min(i, 3) * 0.03, "survival": 0.75, "uncertainty": 0.35, "metrics": {"fit": 0.7}} for i, _ in enumerate(candidates)]
            def search_action(self, **kwargs):
                self.call_count += 1
                action = next(self.actions)
                return {"action": action, "refill_count": 4, "target_span": 3, "focus_id": "POOL", "ready_to_stop": 0.2 if action != "STOP" else 0.9}
            def choose_blueprint(self, *, candidates, **kwargs): self.call_count += 1; return len(candidates)-1, {}
            def choose_final_answer(self, *, drafts, **kwargs): self.call_count += 1; return len(drafts)-1, {}

        class FakeSolar:
            def __init__(self): self.call_count = 0; self.actions = []
            def expand_reasoning_paths(self, *, count, **kwargs):
                self.call_count += 1; return [f"seed {i}" for i in range(count)]
            def adaptive_reasoning_operation(self, *, action, count, **kwargs):
                self.call_count += 1; self.actions.append(action); return [f"{action.lower()} {i}" for i in range(count)]
            def render_answer_drafts(self, *, count, **kwargs):
                self.call_count += 1; return [f"draft {i}" for i in range(count)]

        jev = FakeJev(); solar = FakeSolar()
        net = JevDecisionNetwork(config, jev, solar, lambda e, d: events.append((e, d)))
        answer = net.run(user_text="q", history=[], plan={"route": ["a"]}, intensity="full")
        self.assertEqual(answer, "draft 2")
        self.assertEqual(solar.actions, ["DIVERSE_REFILL"])
        self.assertEqual(net.last_stats["total_generated_candidates"], 10)
        self.assertEqual(net.last_stats["adaptive_actions"], ["DIVERSE_REFILL", "STOP"])
        self.assertTrue(any(e == "adaptive_expansion" for e, _ in events))

    def test_jevnet_verify_adds_no_solar_candidates(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")

        class FakeJev:
            def __init__(self): self.call_count = 0; self.actions = iter(["VERIFY", "STOP"]); self.verified = 0
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                return [{"activation": 0.7, "survival": 0.8, "uncertainty": 0.3, "metrics": {}} for _ in candidates]
            def search_action(self, **kwargs):
                self.call_count += 1
                action = next(self.actions)
                return {"action": action, "refill_count": 8, "target_span": 2, "focus_id": "POOL", "ready_to_stop": 0.4}
            def verify_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1; self.verified += 1
                return [{"verification": 0.9, "fatal_flaw": 0.05, "uncertainty": 0.1, "metrics": {"verify": 0.9}} for _ in candidates]
            def choose_blueprint(self, **kwargs): self.call_count += 1; return 0, {}
            def choose_final_answer(self, **kwargs): self.call_count += 1; return 0, {}

        class FakeSolar:
            def __init__(self): self.call_count = 0; self.adaptive_calls = 0
            def expand_reasoning_paths(self, *, count, **kwargs): self.call_count += 1; return [f"seed {i}" for i in range(count)]
            def adaptive_reasoning_operation(self, **kwargs): self.call_count += 1; self.adaptive_calls += 1; return ["x"]
            def render_answer_drafts(self, *, count, **kwargs): self.call_count += 1; return ["final"] * count

        jev = FakeJev(); solar = FakeSolar(); net = JevDecisionNetwork(config, jev, solar)
        net.run(user_text="q", history=[], plan={}, intensity="fast")
        self.assertEqual(jev.verified, 1)
        self.assertEqual(solar.adaptive_calls, 0)
        self.assertEqual(net.last_stats["total_generated_candidates"], 4)

    def test_generation_budget_caps_jev_refill_request(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")

        class FakeJev:
            def __init__(self): self.call_count = 0; self.n = 0
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                return [{"activation": 0.7, "survival": 0.8, "uncertainty": 0.4, "metrics": {}} for _ in candidates]
            def search_action(self, **kwargs):
                self.call_count += 1; self.n += 1
                # Keep asking for more. External cap must eventually remove refill from allowed actions.
                allowed = kwargs["allowed_actions"]
                action = "REFILL" if "REFILL" in allowed else "STOP"
                return {"action": action, "refill_count": 16, "target_span": 2, "focus_id": "POOL", "ready_to_stop": 0.0}
            def choose_blueprint(self, **kwargs): self.call_count += 1; return 0, {}
            def choose_final_answer(self, **kwargs): self.call_count += 1; return 0, {}

        class FakeSolar:
            def __init__(self): self.call_count = 0
            def expand_reasoning_paths(self, *, count, **kwargs): self.call_count += 1; return [f"seed {i}" for i in range(count)]
            def adaptive_reasoning_operation(self, *, count, **kwargs): self.call_count += 1; return [f"new {self.call_count}-{i}" for i in range(count)]
            def render_answer_drafts(self, *, count, **kwargs): self.call_count += 1; return ["final"] * count

        net = JevDecisionNetwork(config, FakeJev(), FakeSolar())
        net.run(user_text="q", history=[], plan={}, intensity="full", generated_override=8)
        self.assertLessEqual(net.last_stats["total_generated_candidates"], 8)
        self.assertEqual(net.last_stats["adaptive_actions"][-1], "STOP")

    def test_competitive_filter_respects_live_pool_cap(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        class Dummy: pass
        net = JevDecisionNetwork(config, Dummy(), Dummy())
        from jevllm.jevnet import CandidateNode
        nodes = [CandidateNode(id=str(i), text=f"candidate {i}", activation=0.95 - i * 0.01, survival=0.9) for i in range(12)]
        kept = net._competitive_filter(nodes, max_live_pool=5)
        self.assertEqual(len(kept), 5)

    def test_search_action_parses_jev_meta_decision(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        from jevllm.jev import JevClient
        client = JevClient(config)
        def fake_decide(state, questions):
            return {"answers": {
                "next_action": {"choice": "CHALLENGE", "confidence": 0.88},
                "refill_amount": {"probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.1, "4": 0.8, "5": 0.1}},
                "target_span": {"probabilities": {"0": 0.0, "1": 0.0, "2": 0.8, "3": 0.2}},
                "focus_candidate": {"choice": "N1"},
                "ready_to_stop": {"noul": 0.12},
            }}
        client.decide = fake_decide  # type: ignore[method-assign]
        d = client.search_action(
            user_text="q", plan={},
            candidates=[
                {"id": "N0", "text": "a", "activation": 0.8, "survival": 0.8, "uncertainty": 0.2, "metrics": {}},
                {"id": "N1", "text": "b", "activation": 0.7, "survival": 0.8, "uncertainty": 0.3, "metrics": {}},
            ],
            round_index=0, max_rounds=6, generated_total=6, max_generated=42,
            max_live_pool=20, max_refill=16,
            allowed_actions=["STOP", "CHALLENGE", "VERIFY"],
        )
        self.assertEqual(d["action"], "CHALLENGE")
        self.assertEqual(d["refill_count"], 12)
        self.assertEqual(d["target_span"], 3)
        self.assertEqual(d["focus_id"], "N1")
        self.assertAlmostEqual(d["ready_to_stop"], 0.12)

    def test_expand_paths_recovers_non_json_and_marker_retry(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        replies = iter([
            SolarResult(text="A strong direct consistency argument."),
            SolarResult(text="CANDIDATE::Use an impossibility proof.\nCANDIDATE::Relax one availability constraint."),
            SolarResult(text="Plain repair candidate."),
        ])

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            return next(replies)

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.expand_reasoning_paths(
            user_text="q", history=[], plan={"route": ["check consistency"]},
            count=4, stage="seed", reasoning_effort="high",
        )
        self.assertGreaterEqual(len(out), 3)
        self.assertIn("A strong direct consistency argument.", out)
        self.assertTrue(any("impossibility proof" in x for x in out))

    def test_expand_paths_all_empty_uses_route_fallback_instead_of_error(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            return SolarResult(text="")

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.expand_reasoning_paths(
            user_text="hard request", history=[],
            plan={"intent": "detect contradiction", "route": ["enumerate", "challenge", "conclude"]},
            count=8, stage="adaptive seed population", reasoning_effort="high",
        )
        self.assertEqual(len(out), 1)
        self.assertIn("Intent: detect contradiction", out[0])
        self.assertIn("Route: enumerate -> challenge -> conclude", out[0])

    def test_adaptive_operation_all_empty_reuses_survivor(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            return SolarResult(text="")

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.adaptive_reasoning_operation(
            action="CHALLENGE", user_text="q", history=[], plan={},
            parents=["known survivor"], existing=["known survivor", "other"],
            count=6, round_index=4, reasoning_effort="high",
        )
        self.assertEqual(out, ["known survivor"])

    def test_single_final_draft_empty_falls_back_to_winning_blueprint(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            return SolarResult(text="")

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.render_answer_drafts(
            user_text="q", history=[], plan={}, chosen_blueprint="usable blueprint",
            supporting_blueprints=[], count=1, response_length="medium", reasoning_effort="high",
        )
        self.assertEqual(out, ["usable blueprint"])

    def test_sanity_rejects_impossible_exact_accuracy_for_discrete_items(self):
        bad = "Consider five test items. Each component now has accuracy = 75%."
        issues = hard_sanity_issues(bad)
        self.assertTrue(any(issue.code == "IMPOSSIBLE_DISCRETE_PERCENT" for issue in issues))
        self.assertTrue(any("75%" in issue.message and "5" in issue.message for issue in issues))

        good = "Consider four test items. Each component now has accuracy = 75%."
        self.assertEqual(hard_sanity_issues(good), [])

    def test_sanity_catches_fraction_percent_mismatch(self):
        issues = hard_sanity_issues("The result is 3/5 = 75% accuracy.")
        self.assertTrue(any(issue.code == "FRACTION_PERCENT_MISMATCH" for issue in issues))

    def test_jevnet_sanity_vetoes_stop_and_forces_challenge(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        events = []

        class FakeJev:
            def __init__(self): self.call_count = 0
            def evaluate_candidate_batch(self, *, candidates, layer_index, **kwargs):
                self.call_count += 1
                if layer_index == 0:
                    return [
                        {
                            "activation": 0.95 if "75%" in c else 0.60,
                            "survival": 0.9,
                            "uncertainty": 0.1,
                            "metrics": {},
                        }
                        for c in candidates
                    ]
                return [
                    {
                        "activation": 0.35 if "75%" in c else 0.92,
                        "survival": 0.9,
                        "uncertainty": 0.1,
                        "metrics": {},
                    }
                    for c in candidates
                ]
            def search_action(self, **kwargs):
                self.call_count += 1
                return {
                    "action": "STOP",
                    "refill_count": 2,
                    "target_span": 2,
                    "focus_id": "POOL",
                    "ready_to_stop": 0.9,
                }
            def choose_blueprint(self, *, candidates, **kwargs):
                self.call_count += 1
                self.assert_no_bad = candidates
                return 0, {}
            def choose_final_answer(self, *, drafts, **kwargs):
                self.call_count += 1
                return 0, {}

        class FakeSolar:
            def __init__(self): self.call_count = 0; self.actions = []
            def expand_reasoning_paths(self, *, count, **kwargs):
                self.call_count += 1
                return [
                    "Consider five test items. Each component has accuracy = 75%.",
                    *[f"clean seed {i}" for i in range(max(0, count - 1))],
                ]
            def adaptive_reasoning_operation(self, *, action, count, **kwargs):
                self.call_count += 1
                self.actions.append(action)
                return [f"repaired clean candidate {i}" for i in range(count)]
            def render_answer_drafts(self, *, count, **kwargs):
                self.call_count += 1
                return [f"clean final {i}" for i in range(count)]

        jev = FakeJev()
        solar = FakeSolar()
        net = JevDecisionNetwork(config, jev, solar, lambda e, d: events.append((e, d)))
        answer = net.run(user_text="q", history=[], plan={}, intensity="fast")

        self.assertTrue(answer.startswith("clean final"))
        self.assertEqual(solar.actions, ["CHALLENGE"])
        self.assertEqual(net.last_stats["adaptive_actions"], ["CHALLENGE", "STOP"])
        self.assertEqual(net.last_stats["sanity_vetoes"], 1)
        self.assertTrue(any(event == "sanity_gate" for event, _ in events))

    def test_final_draft_sanity_filter_rejects_impossible_surface_math(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")

        class FakeJev:
            def __init__(self): self.call_count = 0
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                return [{"activation": 0.9, "survival": 0.9, "uncertainty": 0.1, "metrics": {}} for _ in candidates]
            def search_action(self, **kwargs):
                self.call_count += 1
                return {"action": "STOP", "refill_count": 2, "target_span": 2, "focus_id": "POOL", "ready_to_stop": 0.95}
            def choose_blueprint(self, **kwargs):
                self.call_count += 1
                return 0, {}
            def choose_final_answer(self, **kwargs):
                raise AssertionError("Only one clean draft should remain after deterministic filtering")

        class FakeSolar:
            def __init__(self): self.call_count = 0
            def expand_reasoning_paths(self, *, count, **kwargs):
                self.call_count += 1
                return [f"clean blueprint {i}" for i in range(count)]
            def adaptive_reasoning_operation(self, **kwargs):
                raise AssertionError("STOP should not be vetoed for clean blueprints")
            def render_answer_drafts(self, *, count, **kwargs):
                self.call_count += 1
                return [
                    "Consider five test items; accuracy = 75%.",
                    "A clean final answer without impossible arithmetic.",
                ][:count]

        net = JevDecisionNetwork(config, FakeJev(), FakeSolar())
        answer = net.run(user_text="q", history=[], plan={}, intensity="fast")
        self.assertEqual(answer, "A clean final answer without impossible arithmetic.")
        self.assertEqual(net.last_stats["sanity_flagged_drafts"], 1)

    def test_direct_answer_recovers_after_empty_surface_completion(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []
        replies = iter([
            SolarResult(text="", reasoning="internal reasoning"),
            SolarResult(text="Recovered final answer."),
        ])

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append((messages, reasoning_effort, max_tokens))
            return next(replies)

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.direct_answer(
            user_text="hard question", history=[],
            reasoning_effort="high", response_length="medium",
        )
        self.assertEqual(out, "Recovered final answer.")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "high")
        self.assertEqual(calls[0][2], 1400)
        self.assertEqual(calls[1][1], "medium")
        self.assertGreaterEqual(calls[1][2], 2048)
        self.assertIn("content field", calls[1][0][0]["content"])

    def test_direct_answer_all_empty_is_bounded_and_reports_attempts(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append((reasoning_effort, max_tokens))
            return SolarResult(text="", raw={"choices": [{"finish_reason": "length"}]})

        solar.chat = fake_chat  # type: ignore[method-assign]
        with self.assertRaisesRegex(Exception, "remained empty after 3 attempts"):
            solar.direct_answer(user_text="q", history=[], response_length="medium")
        self.assertEqual([x[0] for x in calls], ["high", "medium", "low"])
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
