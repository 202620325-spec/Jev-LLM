import json
import tempfile
import unittest
from pathlib import Path

from jevllm.audit import ConversationJSONLogger, summarize_usage
from jevllm.config import Config
from jevllm.controller import AdaptiveController
from jevllm.engine import JevLLM
from jevllm.jevnet import JevDecisionNetwork
from jevllm.solar import SolarClient
from jevllm.types import ControlProfile, SolarResult
from jevllm.util import extract_json, recover_candidate_strings, score_expectation, score_level


class CoreTests(unittest.TestCase):
    def test_usage_summary_counts_only_provider_reported_tokens(self):
        calls = [
            {
                "provider": "solar",
                "usage": {"reported": True, "input_tokens": 120, "output_tokens": 30},
            },
            {
                "provider": "jev",
                "usage": {"reported": False, "input_tokens": 0, "output_tokens": 0},
            },
            {
                "provider": "solar",
                "usage": {"reported": True, "input_tokens": 80, "output_tokens": 20},
            },
        ]
        usage = summarize_usage(calls)
        self.assertEqual(usage["input_tokens"], 200)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 250)
        self.assertEqual(usage["reported_calls"], 2)
        self.assertEqual(usage["unreported_calls"], 1)
        self.assertFalse(usage["all_calls_reported"])
        self.assertEqual(usage["by_provider"]["solar"]["total_tokens"], 250)
        self.assertEqual(usage["by_provider"]["jev"]["unreported_calls"], 1)

    def test_conversation_json_logger_preserves_reasoning_and_events_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = ConversationJSONLogger(tmp)
            reasoning = 'line 1\\nline 2\\n{"raw": true}'
            calls = [
                {
                    "ts_ns": 2,
                    "timestamp": "2026-01-01T00:00:02+00:00",
                    "provider": "solar",
                    "model": "solar-pro3",
                    "request": {"messages": [{"role": "user", "content": "q"}]},
                    "response": {
                        "content": "answer",
                        "reasoning": reasoning,
                        "raw": {"choices": [{"message": {"reasoning": reasoning}}},
                    },
                    "usage": {
                        "reported": True,
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "raw": {"prompt_tokens": 11, "completion_tokens": 7},
                    },
                    "error": None,
                },
                {
                    "ts_ns": 1,
                    "timestamp": "2026-01-01T00:00:01+00:00",
                    "provider": "jev",
                    "model": "typesafe/jev-1.13",
                    "request": {"state": {"x": 1}, "questions": {"next": {"type": "choice"}}},
                    "response": {"reasoning": None, "raw": {"answers": {"next": {"choice": "STOP"}}}},
                    "usage": {"reported": False, "input_tokens": 0, "output_tokens": 0, "raw": {}},
                    "error": None,
                },
            ]
            path = logger.append_turn(
                question="q",
                answer="answer",
                pipeline="net",
                calls=calls,
                events=[{"ts_ns": 3, "event": "search_action", "data": {"action": "STOP"}}],
                stats={"pipeline": "net"},
            )
            self.assertIsNotNone(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            turn = data["turns"][0]
            self.assertEqual(turn["question"], "q")
            self.assertEqual(turn["answer"], "answer")
            self.assertEqual(turn["token_usage"]["input_tokens"], 11)
            self.assertEqual(turn["token_usage"]["output_tokens"], 7)
            self.assertEqual(turn["token_usage"]["unreported_calls"], 1)
            self.assertEqual(turn["thought_log"]["api_calls"][0]["provider"], "jev")
            self.assertEqual(turn["thought_log"]["api_calls"][1]["response"]["reasoning"], reasoning)
            self.assertEqual(turn["thought_log"]["jevnet_events"][0]["event"], "search_action")

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
        config = Config(
            openrouter_api_key="x",
            upstage_api_key="y",
            default_run_mode="fast",
            conversation_log_enabled=False,
        )
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

    def test_final_markdown_answer_is_not_split_into_heading_fragments(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []
        markdown = (
            "**Answer.** The minimum is 5.\n\n"
            "**Construction.**\n"
            "- Classifier A is correct on examples 1,2,3.\n"
            "- Classifier B is correct on examples 1,4,5.\n\n"
            "**Minimality.** For n < 5, the required integer counts cannot coexist."
        )

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append((messages, reasoning_effort, max_tokens))
            return SolarResult(
                text=markdown,
                raw={"choices": [{"finish_reason": "stop"}]},
            )

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.render_answer_drafts(
            user_text="Find the minimum; give a construction and prove minimality.",
            history=[],
            plan={"answer_shape": "construction plus proof", "required_points": ["construction", "minimality"]},
            chosen_blueprint="n=5 with a construction and lower-bound proof",
            supporting_blueprints=[],
            count=5,
            response_length="medium",
            reasoning_effort="high",
        )
        self.assertEqual(out, [markdown])
        self.assertEqual(len(calls), 1)
        self.assertEqual(solar.last_render_stats["protocol"], "raw_prose")
        self.assertFalse(solar.last_render_stats["repaired"])

    def test_truncated_final_fragment_is_repaired_instead_of_selected(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []
        repaired = (
            "The minimum is n=5. Construction: choose three classifiers whose "
            "correctness sets overlap so each has 3/5 accuracy while majority is "
            "correct on only 2/5 examples. Minimality follows by checking n<5."
        )
        replies = iter([
            SolarResult(
                text="*Construction.**",
                raw={"choices": [{"finish_reason": "length"}]},
            ),
            SolarResult(
                text=repaired,
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ])

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append((messages, reasoning_effort, max_tokens))
            return next(replies)

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.render_answer_drafts(
            user_text="Find the minimum; give a construction and prove minimality.",
            history=[],
            plan={"answer_shape": "construction plus proof", "required_points": ["construction", "minimality"]},
            chosen_blueprint="n=5; construct and prove lower bound",
            supporting_blueprints=[],
            count=5,
            response_length="medium",
            reasoning_effort="high",
        )
        self.assertEqual(out, [repaired])
        self.assertEqual(len(calls), 2)
        self.assertTrue(solar.last_render_stats["repaired"])
        self.assertEqual(solar.last_render_stats["finish_reason"], "length")

    def test_valid_structured_final_drafts_keep_multi_draft_selection_without_repair(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []
        raw = (
            '{"candidates":['
            '"The minimum is 5. Here is a complete construction assigning correctness patterns to all examples, followed by a lower-bound argument proving that every smaller n is impossible.",'
            '"n=5. This alternative gives the full classifier table, checks each individual accuracy and the majority-vote accuracy, then proves minimality by exhausting the smaller integer cases."'
            ']}'
        )

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append((messages, reasoning_effort, max_tokens))
            return SolarResult(text=raw, raw={"choices": [{"finish_reason": "stop"}]})

        solar.chat = fake_chat  # type: ignore[method-assign]
        out = solar.render_answer_drafts(
            user_text="Find the minimum; give a construction and prove minimality.",
            history=[],
            plan={"answer_shape": "construction plus proof", "required_points": ["construction", "minimality"]},
            chosen_blueprint="n=5",
            supporting_blueprints=[],
            count=2,
            response_length="medium",
            reasoning_effort="high",
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(solar.last_render_stats["protocol"], "structured")
        self.assertFalse(solar.last_render_stats["repaired"])

    def test_plan_retries_reasoning_only_empty_content(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        calls = []
        replies = iter([
            SolarResult(
                text="",
                reasoning="spent budget reasoning",
                raw={"choices": [{"finish_reason": "length"}]},
            ),
            SolarResult(
                text='{"intent":"solve","route":["preserve alternatives","test claims","answer"],'
                     '"required_points":["proof"],"verification_needs":["check construction"],'
                     '"answer_shape":"short proof","risk_or_uncertainty":[]}',
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ])

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            calls.append((reasoning_effort, max_tokens))
            return next(replies)

        solar.chat = fake_chat  # type: ignore[method-assign]
        plan = solar.plan("hard question", [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(plan["route"][1], "test claims")
        self.assertEqual(plan["verification_needs"], ["check construction"])

    def test_jevnet_does_not_rescore_unchanged_pool_after_verify(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        events = []

        class FakeJev:
            def __init__(self):
                self.call_count = 0
                self.eval_calls = 0
                self.action_calls = 0
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                self.eval_calls += 1
                return [
                    {"activation": 0.8 - i * 0.05, "survival": 0.8, "uncertainty": 0.3, "metrics": {}}
                    for i, _ in enumerate(candidates)
                ]
            def assess_disagreement(self, **kwargs):
                self.call_count += 1
                return {
                    "material_disagreement": 0.9,
                    "needs_test": 0.9,
                    "leader_id": "N0",
                    "rival_id": "N1",
                    "raw": {},
                }
            def search_action(self, *, allowed_actions, **kwargs):
                self.call_count += 1
                self.action_calls += 1
                # Try to VERIFY forever. Network must block the second identical VERIFY.
                return {
                    "action": "VERIFY",
                    "refill_count": 2,
                    "target_span": 2,
                    "focus_id": "N0",
                    "ready_to_stop": 0.2,
                }
            def choose_blueprint(self, *, candidates, **kwargs):
                self.call_count += 1
                return 0, {}
            def choose_final_answer(self, *, drafts, **kwargs):
                self.call_count += 1
                return 0, {}

        class FakeSolar:
            def __init__(self):
                self.call_count = 0
                self.verify_calls = 0
                self.challenge_calls = 0
                self.revive_calls = 0
                self.last_render_stats = {}
            def expand_reasoning_paths(self, *, count, **kwargs):
                self.call_count += 1
                return ["claim A", "claim not-A", *[f"seed {i}" for i in range(max(0, count - 2))]]
            def discriminate_hypotheses(self, **kwargs):
                self.call_count += 1
                self.verify_calls += 1
                return {
                    "material_disagreement": True,
                    "test": "inconclusive check",
                    "test_kind": "INCONCLUSIVE",
                    "resolved": False,
                    "winner_id": None,
                    "results": [
                        {"candidate_id": "N0", "verdict": "UNCERTAIN", "confidence": 0.5, "evidence": ""},
                        {"candidate_id": "N1", "verdict": "UNCERTAIN", "confidence": 0.5, "evidence": ""},
                    ],
                }
            def adaptive_reasoning_operation(self, *, action, count, **kwargs):
                self.call_count += 1
                if action == "CHALLENGE":
                    self.challenge_calls += 1
                if action == "REVIVE":
                    self.revive_calls += 1
                return [f"{action} evidence candidate {i}" for i in range(count)]
            def render_answer_drafts(self, *, count, **kwargs):
                self.call_count += 1
                return ["final"] * count

        jev = FakeJev()
        solar = FakeSolar()
        net = JevDecisionNetwork(config, jev, solar, lambda e, d: events.append((e, d)))
        net.run(
            user_text="q",
            history=[],
            plan={},
            intensity="fast",
            layers_override=3,
            generated_override=8,
        )
        self.assertGreaterEqual(solar.verify_calls, 1)
        self.assertGreaterEqual(solar.revive_calls, 1)
        # Initial candidates are evaluated once; no all-pool semantic rescore after VERIFY.
        self.assertEqual(jev.eval_calls, 2)  # initial pool + newly revived candidates
        self.assertEqual(net.last_stats["adaptive_actions"][:2], ["VERIFY", "REVIVE"])
        self.assertGreater(net.last_stats["reused_evaluations"], 0)

    def test_evidence_falsification_can_remove_semantic_leader(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        class FakeJev:
            def __init__(self): self.call_count = 0; self.actions = iter(["VERIFY", "STOP"])
            def evaluate_candidate_batch(self, *, candidates, **kwargs):
                self.call_count += 1
                # False hypothesis is semantic leader before evidence.
                return [
                    {"activation": 0.95 if "false" in c else 0.65, "survival": 0.9, "uncertainty": 0.2, "metrics": {}}
                    for c in candidates
                ]
            def assess_disagreement(self, **kwargs):
                self.call_count += 1
                return {
                    "material_disagreement": 0.95,
                    "needs_test": 0.95,
                    "leader_id": "N0",
                    "rival_id": "N1",
                    "raw": {},
                }
            def search_action(self, **kwargs):
                self.call_count += 1
                return {
                    "action": next(self.actions),
                    "refill_count": 2,
                    "target_span": 2,
                    "focus_id": "POOL",
                    "ready_to_stop": 0.2,
                }
            def choose_blueprint(self, *, candidates, **kwargs):
                self.call_count += 1
                return 0, {}
            def choose_final_answer(self, *, drafts, **kwargs):
                self.call_count += 1
                return 0, {}

        class FakeSolar:
            def __init__(self): self.call_count = 0; self.last_render_stats = {}
            def expand_reasoning_paths(self, *, count, **kwargs):
                self.call_count += 1
                return ["false but polished conclusion", "correct rival conclusion"] + [
                    f"neutral {i}" for i in range(max(0, count - 2))
                ]
            def discriminate_hypotheses(self, **kwargs):
                self.call_count += 1
                return {
                    "material_disagreement": True,
                    "test": "replay the claimed construction",
                    "test_kind": "DETERMINISTIC",
                    "resolved": True,
                    "winner_id": "N1",
                    "results": [
                        {
                            "candidate_id": "N0",
                            "verdict": "FAIL",
                            "confidence": 0.99,
                            "evidence": "construction does not produce the claimed state",
                        },
                        {
                            "candidate_id": "N1",
                            "verdict": "PASS",
                            "confidence": 0.95,
                            "evidence": "invariant holds under every allowed move",
                        },
                    ],
                }
            def adaptive_reasoning_operation(self, **kwargs):
                raise AssertionError("No expansion needed after decisive evidence")
            def render_answer_drafts(self, *, chosen_blueprint, count, **kwargs):
                self.call_count += 1
                return [chosen_blueprint] * count

        net = JevDecisionNetwork(config, FakeJev(), FakeSolar())
        answer = net.run(
            user_text="q",
            history=[],
            plan={},
            intensity="fast",
            layers_override=2,
        )
        self.assertIn("correct rival", answer)
        self.assertNotIn("false but polished", answer)
        self.assertEqual(net.last_stats["evidence_tests"], 1)

    def test_diversity_pressure_is_idempotent_across_rounds(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        class Dummy: pass
        net = JevDecisionNetwork(config, Dummy(), Dummy())
        from jevllm.jevnet import CandidateNode
        nodes = [
            CandidateNode(id="N0", text="same hypothesis alpha", activation=0.8),
            CandidateNode(id="N1", text="same hypothesis alpha variant", activation=0.7),
        ]
        before = [n.activation for n in nodes]
        net._diversity_adjust(nodes)
        first_factors = [n.diversity_factor for n in nodes]
        net._diversity_adjust(nodes)
        second_factors = [n.diversity_factor for n in nodes]
        self.assertEqual([n.activation for n in nodes], before)
        self.assertEqual(first_factors, second_factors)

    def test_discriminator_cannot_resolve_without_pass_fail_separation(self):
        config = Config(openrouter_api_key="x", upstage_api_key="y")
        solar = SolarClient(config)
        replies = iter([
            SolarResult(text='{"material_disagreement":true,"test":"check","test_kind":"DERIVATION",'
                             '"resolved":true,"winner_id":"N0","results":['
                             '{"candidate_id":"N0","verdict":"PASS","confidence":0.95,"evidence":"ok"},'
                             '{"candidate_id":"N1","verdict":"UNCERTAIN","confidence":0.9,"evidence":"not checked"}]}'),
            SolarResult(text='{"material_disagreement":true,"test":"check","test_kind":"DERIVATION",'
                             '"resolved":true,"winner_id":"N0","results":['
                             '{"candidate_id":"N0","verdict":"PASS","confidence":0.95,"evidence":"holds"},'
                             '{"candidate_id":"N1","verdict":"FAIL","confidence":0.92,"evidence":"counterexample"}]}'),
        ])

        def fake_chat(messages, *, reasoning_effort=None, max_tokens=None):
            return next(replies)

        solar.chat = fake_chat  # type: ignore[method-assign]
        candidates = [
            {"id": "N0", "text": "claim A"},
            {"id": "N1", "text": "claim not-A"},
        ]
        first = solar.discriminate_hypotheses(
            user_text="q", history=[], plan={}, candidates=candidates
        )
        second = solar.discriminate_hypotheses(
            user_text="q", history=[], plan={}, candidates=candidates
        )
        self.assertFalse(first["resolved"])
        self.assertIsNone(first["winner_id"])
        self.assertTrue(second["resolved"])
        self.assertEqual(second["winner_id"], "N0")

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
