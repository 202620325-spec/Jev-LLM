import unittest

from jevllm.config import Config
from jevllm.controller import AdaptiveController
from jevllm.engine import JevLLM
from jevllm.jevnet import JevDecisionNetwork
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

if __name__ == "__main__":
    unittest.main()
