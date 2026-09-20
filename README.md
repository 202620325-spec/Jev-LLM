# JevNet -> LLM v1.5.0

CMD chat MVP combining:

- **Solar Pro 3**: generates semantic answer/solution candidates.
- **Jev Decisions**: evaluates the live candidate pool and decides what inference operation happens next.

The core identity is **adaptive search with evidence-gated collapse and locked surface generation**, not a fixed candidate-count pipeline.

## Core loop

```text
question
  -> query router
       simple definition -> small grounded search
       reasoning/proof   -> evidence/claim validation enabled
       normal            -> adaptive search
  -> Solar semantic proposals
  -> Jev evaluates NEW hypotheses
  -> preserve material disagreement
  -> Jev chooses NEXT ACTION
       STOP / REFILL / DIVERSE_REFILL / DEEPEN / MUTATE
       MERGE / CHALLENGE / REVIVE / VERIFY
  -> VERIFY acquires discriminating evidence, not confidence-only rescoring
  -> collapse only when justified
  -> Jev chooses semantic blueprint
  -> proof-like answers: atomic claim audit + repair/re-audit
  -> simple definitions: STATE LOCK
  -> Solar verbalizes the locked/verified state
  -> optional state-conformance or surface claim audit
  -> output
```

### Important

`fast`, `full`, and `max` are **hard compute ceilings**, not mandatory search sizes.

Example: `:max` allows a much larger search, but if Jev chooses `STOP` after the first evaluation, only the seed population is used.

The external controller limits latency/cost. Within that envelope Jev chooses whether to expand, diversify, deepen, mutate, merge, challenge, revive, verify, or stop. Simple-definition routing may intentionally choose a much smaller deterministic envelope before the adaptive loop.

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
| `REVIVE` | Hide the current pool and generate clean-room hypotheses from the original request. |
| `VERIFY` | Acquire one discriminating evidence report; never same-pool confidence-only rescoring. |

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

## v1.5: STATE LOCK + claim-level proof audit

v1.5 separates two failure classes that should not be solved by the same mechanism.

### Simple-definition path

High-confidence definition queries such as `What is X?` or Korean `X가 뭐냐` are routed through a small deterministic envelope:

```text
query router
  -> <=3 grounded definition seeds
  -> lightweight Jev evaluation
     - semantic fit
     - correctness
     - scope discipline
     - directness
  -> one selected blueprint
  -> STATE LOCK
  -> Solar verbalizer (one draft only)
  -> Jev state-conformance gate
```

The state lock contains required/active/optional/suppressed concepts, register, abstraction and hard surface budgets. The final Solar call does not receive losing blueprints and is explicitly forbidden from opening new factual branches.

Low-confidence final argmax is no longer accepted blindly: multi-draft decisions below 0.50 confidence fall back toward the selected blueprint and concise wording.

### Proof / derivation path

Proof-like requests now receive a claim-level audit even when competing candidates agree on the same conclusion.

```text
selected conclusion/proof
  -> extract atomic support claims
  -> recompute/check counts, parity, invariants, ranks,
     state-space/BFS claims and constructions
  -> conclusion can PASS while proof still FAILS
  -> remove/repair FAIL and material UNCERTAIN claims
  -> re-audit
  -> render one answer
  -> audit the user-facing surface again
```

This prevents a correct conclusion from laundering false support such as a wrong cell count or unsupported GF(2) rank.

### Cost control

- simple definitions skip the Solar route-planner API call;
- simple definitions cap seed=3, rounds=2, generated=6, drafts=1;
- simple-definition candidate evaluation uses four compact metrics instead of the full seven-metric battery;
- shared request/plan context is sent once per Jev batch rather than duplicated inside every candidate record;
- normal evidence-gated v1.4 behavior remains for difficult reasoning tasks.

## v1.4: disagreement before collapse

v1.4 changes VERIFY from repeated Jev plausibility scoring into evidence acquisition.

The adaptive loop now follows this rule:

```text
Generate hypotheses
  -> evaluate new hypotheses once
  -> detect answer-changing disagreement
  -> PRESERVE leader + strongest incompatible rival
  -> VERIFY by a discriminating evidence test
       - replay a construction
       - substitute/check equations
       - derive an invariant
       - try a counterexample
       - use an external deterministic verifier when one is attached
  -> FAIL evidence can eliminate a high-scoring hypothesis
  -> unresolved same-pool VERIFY cannot repeat
  -> REVIVE: hide the current pool and generate clean-room hypotheses
  -> CHALLENGE / DIVERSE when targeted adversarial search is still useful
  -> COLLAPSE only after disagreement is resolved or no material conflict remains
```

Important implementation changes:

- unchanged candidates are not fully re-scored by Jev every round;
- one unchanged pool can be VERIFY'd only once;
- unresolved verification triggers clean-room REVIVE so a dominant framing cannot keep feeding itself;
- Jev detects disagreement but does not certify truth;
- Solar performs the default evidence-producing verification;
- `JevDecisionNetwork(..., verifier=...)` accepts an external deterministic verifier hook, intended for environments such as JevCoder where compile/test/typecheck/runtime checks are available;
- a hypothesis with strong concrete FAIL evidence is excluded from final selection when a non-falsified alternative survives;
- a conflict explicitly resolved by evidence cannot later be overturned by a pure semantic/style vote;
- the Solar route planner retries when provider-side reasoning consumes the whole output budget and visible JSON is empty.

## Automatic JSON conversation audit

Every completed user turn is automatically appended to one local session file:

```text
logs/conversations/session_<UTC timestamp>_<id>.json
```

Each turn contains:

- `question` and final `answer`
- pipeline (`solar`, `net`, `legacy`, or `compare`)
- provider-reported input/output token totals, split by Solar/Jev
- every Solar request/response, including the provider-returned `message.reasoning` field as received
- every Jev Decisions request and raw decision response
- JevNet events such as plan, candidate evaluation, adaptive action, verification, blueprint selection, surface rendering, and final selection
- last-turn search statistics

Token totals are never guessed. If a provider response contains no usage object, that call is recorded as `reported: false` and counted under `unreported_calls`.

The files are local runtime data and are ignored by git because they may contain prompts, answers, and model reasoning.

Configuration:

```env
CONVERSATION_LOG_ENABLED=1
CONVERSATION_LOG_DIR=logs/conversations
```

`:reset` clears chat history and starts a new log-session filename. `:profile` shows the current target log path. `:compare` stores the Solar baseline and JevNet answer together as one comparison turn.

## Testing

```bat
python -m compileall -q .
python -m unittest discover -s tests -v
```

v1.5.0 regression suite currently defines **41 tests**.

Run locally:

```bat
python -m compileall -q .
python -m unittest discover -s tests -v
```

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
