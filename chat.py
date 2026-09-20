from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from jevllm.config import Config
from jevllm.engine import JevLLM
from jevllm.jev import JevError
from jevllm.solar import SolarError


BANNER = r"""
======================================================================
 JEVNET -> LLM v1.3 | Adaptive Jev search over Solar proposal space
======================================================================
 Jev decides: STOP / REFILL / DIVERSE / DEEPEN / MUTATE / MERGE /
              CHALLENGE / VERIFY
 Default: :net + :auto
 Type :help for commands.
""".strip()


def _short(text: str, n: int = 105) -> str:
    return text[:n] + ("..." if len(text) > n else "")


def make_event_sink(state: dict[str, bool]):
    def sink(event: str, data: dict):
        if event == "network_profile":
            p = data["profile"]
            print(
                f"\n[JevNet cap] profile={p['name']} seed={p['initial_seed']} "
                f"rounds<={p['max_rounds']} live_pool<={p['max_live_pool']} "
                f"generated<={p['max_generated']} refill<={p['max_refill']} "
                f"drafts={p['final_drafts']}"
            )
            return

        if event == "layer" and state["trace"]:
            nodes = data.get("nodes", [])
            top = nodes[0] if nodes else None
            if top:
                print(
                    f"[Evaluate R{data['layer']}] {data['before']} -> {data['after']} "
                    f"| top={top.get('activation', 0.0):.3f} unc={top.get('uncertainty', 0.0):.3f} "
                    f"| {_short(top.get('text', ''))}"
                )
            else:
                print(f"[Evaluate R{data['layer']}] {data['before']} -> {data['after']}")
            return

        if event == "search_action" and state["trace"]:
            d = data["decision"]
            print(
                f"[Jev action R{data['round']}] {d.get('action')} "
                f"refill={d.get('refill_count')} span={d.get('target_span')} "
                f"focus={d.get('focus_id')} ready={d.get('ready_to_stop', 0.0):.2f} "
                f"| pool={data['pool_size']} generated={data['generated_total']} "
                f"remaining={data['remaining_generated']}"
            )
            return

        if event == "adaptive_expansion" and state["trace"]:
            print(
                f"[{data['action']}] +{len(data.get('children', []))} "
                f"(requested {data['requested']}) -> pool={data['pool_size']} "
                f"generated={data['generated_total']}"
            )
            return

        if event == "verification" and state["trace"]:
            print(f"[VERIFY] re-checked {len(data.get('nodes', []))} candidates")
            return

        if event == "sanity_gate" and state["trace"]:
            issues = data.get("issues", [])
            first = issues[0].get("message", "deterministic contradiction") if issues else "deterministic contradiction"
            print(
                f"[Sanity R{data['round']}] veto {data['requested_action']} -> {data['forced_action']} "
                f"| {data['candidate_id']}: {first}"
            )
            return

        if event == "sanity_drafts" and state["trace"]:
            print(
                f"[Sanity final] rejected {len(data.get('flagged', []))} draft(s) "
                f"| clean_available={data.get('clean_available')}"
            )
            return

        if not state["debug"]:
            return

        if event == "plan":
            print("[Solar route]", json.dumps(data["plan"], ensure_ascii=False, indent=2))
        elif event == "expansion":
            print(f"[Seed] {len(data.get('nodes', []))} candidates")
            for n in data.get("nodes", []):
                print(f"  {n['id']}: {n['text']}")
        elif event == "layer":
            print(f"[Evaluation R{data['layer']}] {data['before']} -> {data['after']}")
            for n in data.get("nodes", []):
                print(
                    f"  {n['id']} act={n['activation']:.3f} survive={n['survival']:.3f} "
                    f"unc={n['uncertainty']:.3f} src={n['source']} :: {n['text']}"
                )
        elif event == "search_action":
            print("[Jev meta]", json.dumps(data["decision"], ensure_ascii=False, indent=2))
        elif event == "adaptive_expansion":
            print(f"[{data['action']}] generated {len(data.get('children', []))}")
            for n in data.get("children", []):
                print(f"  + {n['id']} [{n['source']}]: {n['text']}")
        elif event == "verification":
            print("[VERIFY result]")
            for n in data.get("nodes", []):
                print(
                    f"  {n['id']} act={n['activation']:.3f} survive={n['survival']:.3f} "
                    f"unc={n['uncertainty']:.3f} :: {n['text']}"
                )
        elif event == "blueprint_selection":
            print(f"[Jev blueprint] C{data['index']} -> {data['blueprint']}")
        elif event == "final_drafts":
            for i, d in enumerate(data["drafts"]):
                print(f"[Draft {i}] {d}")
        elif event == "final_selection":
            print(f"[Jev final] draft={data['index']}")
        elif event == "sanity_gate":
            print("[Deterministic sanity veto]", json.dumps(data, ensure_ascii=False, indent=2))
        elif event == "sanity_finalists":
            print("[Sanity finalist filter]", json.dumps(data, ensure_ascii=False, indent=2))
        elif event == "sanity_drafts":
            print("[Sanity draft filter]", json.dumps(data, ensure_ascii=False, indent=2))
        elif event == "control":
            a = data["adaptive"]
            print(
                f"[Legacy Jev] horizon={a['horizon']} mode={a['cognitive_mode']} "
                f"choices={a['breadth']} scope={a['reasoning_scope']}"
            )
        elif event == "candidates":
            for i, c in enumerate(data["candidates"]):
                print(f"  C{i}: {c!r}")
        elif event == "selection":
            print(f"[Legacy select] C{data['index']}")
        elif event == "prefix":
            print(f"[Legacy scaffold] {data['prefix']!r}")
    return sink


def ensure_env_template() -> None:
    env = Path(".env")
    sample = Path(".env.example")
    if not env.exists() and sample.exists():
        try:
            shutil.copyfile(sample, env)
            print("[setup] .env.example -> .env. Put API keys in .env and run again.\n")
        except OSError:
            pass


def print_help() -> None:
    print(r"""
PIPELINE
  :net             adaptive JevNet inference (default)
  :solar           Solar Pro 3 alone; baseline
  :legacy          old fragment-by-fragment Jev controller

COMPUTE CAPS — Jev may stop far below these
  :auto            Jev chooses a compute envelope, then controls search dynamically
  :fast            small cap; seed 4, <=3 meta rounds, <=18 generated candidates
  :full            medium cap; seed 6, <=6 rounds, <=42 generated candidates
  :max             large cap; seed 8, <=10 rounds, <=90 generated candidates

CAP OVERRIDES
  :seed N          initial seed population, N=2..24
  :pool N          maximum live candidate pool, N=4..64
  :rounds N        maximum Jev meta-decision rounds, N=1..16
  :generated N     maximum total candidates generated in a turn, N=2..160
  :refill N        maximum candidates Jev may request in one refill, N=1..24
  :drafts N        final answer drafts competing under Jev, N=1..8
  :clear           clear overrides and return to :auto

BACKWARD-COMPATIBLE ALIASES
  :width N         same as :pool N
  :layers N        same as :rounds N
  :mutate N        same as :refill N

OBSERVE / TEST
  :trace           compact adaptive search trace
  :debug           all candidates, activations, and Jev meta decisions
  :stats           last-turn search/call stats
  :profile         current pipeline + caps
  :compare PROMPT  Solar baseline vs adaptive JevNet on the same prompt
  :doctor          live-test Jev + Solar APIs
  :reset           clear conversation history
  :help            show this list
  :quit            exit

JEV META ACTIONS
  STOP             enough evidence; finalize now
  REFILL           more candidates near promising regions
  DIVERSE_REFILL   search genuinely different regions
  DEEPEN           add missing structure to strong candidates
  MUTATE           repair weak assumptions / constraints
  MERGE            combine complementary survivors
  CHALLENGE        adversarial alternatives / counterexamples
  VERIFY           Jev-only stricter re-check; no new Solar candidates
""".strip())


def _set_int(engine: JevLLM, cmd: str, arg: str) -> bool:
    try:
        n = int(arg)
    except ValueError:
        print(f"usage: {cmd} <integer>")
        return True

    if cmd == ":seed":
        engine.seed_override = max(2, min(24, n))
        print(f"seed_override={engine.seed_override}")
    elif cmd in {":pool", ":width"}:
        engine.width_override = max(4, min(64, n))
        print(f"max_live_pool={engine.width_override}")
    elif cmd in {":rounds", ":layers"}:
        engine.layers_override = max(1, min(16, n))
        print(f"max_rounds={engine.layers_override}")
    elif cmd == ":generated":
        engine.generated_override = max(2, min(160, n))
        print(f"max_generated={engine.generated_override}")
    elif cmd in {":refill", ":mutate"}:
        engine.mutation_override = max(1, min(24, n))
        print(f"max_refill_per_action={engine.mutation_override}")
    elif cmd == ":drafts":
        engine.drafts_override = max(1, min(8, n))
        print(f"final_drafts={engine.drafts_override}")
    else:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive JevNet-controlled Solar chat MVP")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ensure_env_template()
    config = Config.load()
    missing = config.validate()
    if missing:
        print("Missing API keys: " + ", ".join(missing))
        print("Edit .env and set OPENROUTER_API_KEY and UPSTAGE_API_KEY.")
        return 2

    state = {"debug": bool(config.debug or args.debug), "trace": False}
    engine = JevLLM(config, event_sink=make_event_sink(state))

    if args.doctor:
        try:
            ok, detail = engine.doctor()
            print(("PASS" if ok else "FAIL") + " - " + detail)
            return 0 if ok else 1
        except (JevError, SolarError) as exc:
            print("FAIL -", exc)
            return 1

    print(BANNER)
    print(f"Jev={config.jev_model} | Solar={config.solar_model}\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user:
            continue

        parts = user.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {":quit", ":q", ":exit"}:
            print("bye")
            break
        if cmd == ":help":
            print_help(); continue
        if cmd == ":net":
            engine.pipeline_mode = "net"; print("pipeline=net"); continue
        if cmd == ":solar":
            engine.pipeline_mode = "solar"; print("pipeline=solar (baseline)"); continue
        if cmd == ":legacy":
            engine.pipeline_mode = "legacy"; print("pipeline=legacy"); continue
        if cmd in {":auto", ":fast", ":full", ":max"}:
            engine.intensity = cmd[1:]; print(f"intensity={engine.intensity}"); continue
        if cmd == ":clear":
            engine.clear_overrides(); print("intensity=auto; cap overrides cleared"); continue
        if cmd in {":seed", ":pool", ":rounds", ":generated", ":refill", ":drafts", ":width", ":layers", ":mutate"}:
            _set_int(engine, cmd, arg); continue
        if cmd == ":trace":
            state["trace"] = not state["trace"]; print(f"trace={state['trace']}"); continue
        if cmd == ":debug":
            state["debug"] = not state["debug"]; print(f"debug={state['debug']}"); continue
        if cmd == ":stats":
            print(json.dumps(engine.last_stats or {"note": "no completed turn yet"}, ensure_ascii=False, indent=2)); continue
        if cmd == ":profile":
            print(json.dumps({
                "pipeline": engine.pipeline_mode,
                "intensity": engine.intensity,
                "seed_override": engine.seed_override,
                "max_live_pool_override": engine.width_override,
                "max_rounds_override": engine.layers_override,
                "max_generated_override": engine.generated_override,
                "max_refill_override": engine.mutation_override,
                "drafts_override": engine.drafts_override,
            }, ensure_ascii=False, indent=2)); continue
        if cmd == ":compare":
            if not arg:
                print("usage: :compare <prompt>"); continue
            original_pipeline = engine.pipeline_mode
            original_history = list(engine.history)
            try:
                base_start = engine.solar.call_count
                baseline_error = None
                try:
                    baseline = engine.solar.direct_answer(
                        user_text=arg,
                        history=original_history,
                        reasoning_effort="high",
                        response_length="medium",
                    )
                except SolarError as exc:
                    baseline_error = str(exc)
                    baseline = f"[Solar baseline unavailable: {baseline_error}]"
                base_calls = engine.solar.call_count - base_start

                # A baseline transport/surface failure must not cancel the JevNet
                # half of the controlled comparison.
                engine.history = list(original_history)
                engine.pipeline_mode = "net"
                net_answer = engine.answer(arg)
                print(f"\n--- SOLAR BASELINE ({base_calls} Solar call) ---\n{baseline}")
                if baseline_error:
                    print(f"[baseline recovery exhausted] {baseline_error}")
                print(f"\n--- JEVNET ADAPTIVE ---\n{net_answer}")
                print("\n[JevNet stats]", json.dumps(engine.last_stats, ensure_ascii=False))
            except (JevError, SolarError, ValueError, RuntimeError) as exc:
                engine.history = original_history
                print(f"\n[error] {exc}\n")
            finally:
                engine.pipeline_mode = original_pipeline
            continue
        if cmd == ":reset":
            engine.reset(); print("history cleared"); continue
        if cmd == ":doctor":
            try:
                ok, detail = engine.doctor(); print(("PASS" if ok else "FAIL") + " - " + detail)
            except (JevError, SolarError) as exc:
                print("FAIL -", exc)
            continue

        try:
            answer = engine.answer(user)
            print(f"\nllm> {answer}\n")
        except (JevError, SolarError, ValueError, RuntimeError) as exc:
            print(f"\n[error] {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
