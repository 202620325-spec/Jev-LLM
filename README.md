# JevNet -> LLM v1.3.1

CMD chat MVP combining:

- **Solar Pro 3**: generates semantic answer/solution candidates.
- **Jev Decisions**: evaluates the live candidate pool and decides what inference operation happens next.

The identity of v1.3 is **adaptive search**, not a fixed 8/16/24/30-candidate pipeline.

## Core loop

```text
question
  -> Solar route
  -> small seed candidate pool
  -> Jev multi-axis evaluation
  -> Jev chooses NEXT ACTION
       STOP
       REFILL
       DIVERSE_REFILL
       DEEPEN
       MUTATE
       MERGE
       CHALLENGE
       VERIFY
  -> perform only that operation
  -> evaluate again
  -> repeat until Jev chooses STOP or an external compute cap is reached
  -> Jev chooses winning blueprint
  -> Solar renders competing final drafts
  -> Jev chooses final answer
```

### Important

`fast`, `full`, and `max` are **hard compute ceilings**, not mandatory search sizes.

Example: `:max` allows a much larger search, but if Jev chooses `STOP` after the first evaluation, only the seed population is used.

The external controller can only stop Jev from exceeding configured latency/cost limits. Within that envelope Jev chooses whether to expand, diversify, deepen, mutate, merge, challenge, verify, or stop.

## Adaptive actions

| Action | Effect |
|---|---|
| `STOP` | Current pool is sufficient; finalize. |
| `REFILL` | Add nearby strong alternatives. |
| `DIVERSE_REFILL` | Explore substantially different approaches. |
| `DEEPEN` | Add missing technical/logical structure to strong candidates. |
| `MUTATE` | Repair weak assumptions or constraint failures. |
| `MERGE` | Combine complementary survivors into new candidates. |
| `CHALLENGE` | Generate adversarial/counter-hypotheses to attack the leader. |
| `VERIFY` | No new Solar candidates; Jev re-checks strong candidates under stricter verification criteria. |

Jev also chooses:

- requested refill size
- how many top candidates the operation should condition on
- a specific candidate vs the whole pool as the focus
- when search is sufficiently complete

## Run

Windows:

```bat
setup.bat
```

Edit `.env`:

```env
OPENROUTER_API_KEY=...
UPSTAGE_API_KEY=...
```

Then:

```bat
start.bat
```

or:

```bat
python chat.py
```

## Commands

### Pipeline

```text
:net             adaptive JevNet (default)
:solar           Solar-only baseline
:legacy          older fragment controller
```

### Compute envelopes

```text
:auto            Jev chooses the envelope, then adaptive search decides actual spend
:fast            seed 4, <=3 meta rounds, <=18 generated candidates
:full            seed 6, <=6 meta rounds, <=42 generated candidates
:max             seed 8, <=10 meta rounds, <=90 generated candidates
```

Again, those candidate counts are maxima except for the small seed. They are not fixed targets.

### Cap overrides

```text
:seed N          initial seed, 2..24
:pool N          max live candidate pool, 4..64
:rounds N        max Jev meta-decision rounds, 1..16
:generated N     max total generated candidates, 2..160
:refill N        max candidates in a single Jev-requested refill, 1..24
:drafts N        final answer drafts, 1..8
:clear           reset overrides and return to :auto
```

Backward-compatible aliases:

```text
:width N         alias of :pool
:layers N        alias of :rounds
:mutate N        alias of :refill
```

### Inspect / benchmark

```text
:trace           show compact adaptive-search trace
:debug           show candidates, activations, and full Jev meta decisions
:stats           last turn stats and action history
:profile         current caps/settings
:compare PROMPT  same prompt: Solar baseline vs adaptive JevNet
:doctor          live Jev + Solar connectivity test
:reset           clear conversation history
:help
:quit
```

A trace can look like:

```text
[JevNet cap] profile=full seed=6 rounds<=6 live_pool<=20 generated<=42 refill<=8 drafts=3
[Evaluate R1] 6 -> 5 | top=0.82 unc=0.24
[Jev action R1] DIVERSE_REFILL refill=6 span=3 focus=POOL ready=0.18
[DIVERSE_REFILL] +6 -> pool=11 generated=12
[Evaluate R2] 11 -> 7 | top=0.89 unc=0.16
[Jev action R2] CHALLENGE refill=4 span=2 focus=N7 ready=0.31
[CHALLENGE] +4 -> pool=11 generated=16
[Evaluate R3] 11 -> 6 | top=0.93 unc=0.09
[Jev action R3] VERIFY refill=2 span=2 focus=N7 ready=0.67
[VERIFY] re-checked 2 candidates
[Evaluate R4] 6 -> 4 | top=0.95 unc=0.06
[Jev action R4] STOP ... ready=0.92
```

## Testing

```bat
python -m compileall -q .
python -m unittest discover -s tests -v
```

v1.3.1 offline status at patch time:

- Python compile: PASS
- unit/regression tests: **21/21 PASS**

Tests cover, among other things:

- malformed/non-JSON Solar candidate output is recovered without aborting JevNet.
- repeated successful-but-empty Solar candidate completions degrade to route/survivor fallbacks instead of raising `no recoverable candidates`.
- final-draft empty content falls back to the winning blueprint rather than crashing.
- Jev can stop immediately even under `:max`.
- Jev can request a refill and then stop.
- `VERIFY` adds no Solar candidates.
- a repeated Jev refill request cannot exceed `max_generated`.
- meta-action parsing selects action/refill/focus/span correctly.
- prior Solar malformed-candidate recovery remains covered.

## What this is / is not

This does **not** train or modify Jev's neural-network weights. It is an inference-time decision/search network: Solar proposes semantic alternatives and Jev repeatedly applies structured selection pressure and meta-control over the search process.

Whether this is actually more accurate than Solar alone must be measured with `:compare` or a proper benchmark. More inference compute by itself is not evidence of higher intelligence.

## License

Released under the [MIT License](LICENSE).
