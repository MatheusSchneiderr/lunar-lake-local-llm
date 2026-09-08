# 4. Thinking Mode and Preservation

`Qwen3.6-35B-A3B` — the GPU tier's model (see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md)) — is a hybrid reasoning model: it defaults to **thinking-mode-on**, emitting a `<think>...</think>` block before its actual answer on every request, unless something tells it not to. This chapter covers two things investigated on this exact stack: whether that default is actually worth its cost for this project's usage pattern, and whether preserving prior turns' reasoning traces across a multi-turn conversation (rather than discarding them once a turn is "closed") measurably helps. Both were benchmarked, not guessed at, and both benchmarks changed the shipped config.

If you only read one section of this chapter, read the callout box below first — the three mechanisms it disambiguates are easy to confuse even having just lived through building this, and getting them backwards will cost you a debugging session.

## The three mechanisms — read this before touching any of the flags below

There are three superficially similar levers in play here. Two of them work. One is a confirmed no-op. They are not interchangeable, and none of them is a synonym for another:

> **(a) The per-request JSON field `"reasoning": "off"` — a confirmed no-op.**
> Sending `{"reasoning": "off"}` at the top level of a `/v1/chat/completions` request body does **nothing** on this stack. It was tested directly against the running `gpu-server-hard` and produced no observable change in behavior — thinking still ran exactly as it would have without the field. Don't reach for this field; it is not how you control reasoning per-request here.
>
> **(b) The per-request `chat_template_kwargs` fields — these are the ones that actually work.**
> `{"chat_template_kwargs": {"enable_thinking": false}}` and `{"chat_template_kwargs": {"preserve_thinking": true}}` are real, functioning per-request overrides. They work because the model's Jinja chat template itself reads these two variable names directly — not because llama-server has any special-cased handling for them. This was confirmed by extracting and reading the actual patched template (see [configs/gpu-tier/chat_template.patched.jinja](../configs/gpu-tier/chat_template.patched.jinja)):
>   - Line 149: `{%- if enable_thinking is defined and enable_thinking is false %}` — when `enable_thinking` is explicitly `false`, the template emits a pre-closed `<think>\n\n</think>\n\n` immediately after the generation prompt, which is what actually suppresses reasoning (the model never gets a chance to open a real thinking block).
>   - Line 100: `{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}` — for a **historical** assistant message being re-rendered into the prompt, this decides whether that older turn's `<think>...</think>` content gets included again or stripped down to just its final answer. `loop.index0 > ns.last_query_index` is the normal case (only the most recent turn's reasoning is ever kept by default); `preserve_thinking is true` overrides that and keeps every historical turn's reasoning too.
>
>   Because these are read straight out of the Jinja context, they work anywhere the request reaches this chat template — no server restart needed, and they were confirmed to work per-request against a live, already-running instance.
>
> **(c) The llama-server CLI startup flags `--reasoning [on|off|auto]` and `--reasoning-preserve` — server-wide defaults for the same underlying behavior.**
> These are set once in `ExecStart` (see [configs/gpu-tier/default.nix](../configs/gpu-tier/default.nix)) and become what every request gets *when it doesn't specify otherwise*. `gpu-server-hard` runs with `--reasoning off` as its permanent default (rationale below). Critically, **a per-request `chat_template_kwargs` field always overrides the server-wide flag for that one request** — confirmed directly: with `--reasoning off` live, a plain request produces zero reasoning content, but the exact same request with `{"chat_template_kwargs": {"enable_thinking": true}}` added fully re-enables reasoning for that call, with no other change. `--reasoning-preserve` was evaluated as a server-wide default for the (b)-equivalent `preserve_thinking` behavior but was never enabled — see the preservation verdict below.
>
> **The mental model that keeps these straight**: (a) doesn't exist as far as this stack is concerned — forget it. (b) is what you set per-request, from a client. (c) is what the server does when a client doesn't set (b). Same underlying behavior, two different altitudes of control, plus one field that looks plausible but is dead weight.

## Investigation 1: does thinking mode actually help, and what does it cost?

### Why this got investigated

The question arose from an ordinary nvim session, not a planned benchmarking pass: "does showing reasoning cost tokens?", followed immediately by "how much generation speed do we lose?" Rather than answer from intuition, this became a real paired benchmark against the live `gpu-server-hard` endpoint.

### Method

Five prompt types, each with a single, mechanically verifiable correct answer (so grading is objective, not a judgment call):

1. An arithmetic word problem
2. A recursive Fibonacci trace (asking for the exact call sequence/values, not just the final number)
3. A 3-box weight-ordering logic puzzle
4. A Python list-comprehension formatting task
5. A day-of-week calculation

Each prompt type was run 2–3 times with fixed seeds (not a single stochastic sample — see the seeding discipline established in the KV-cache-quantization work, [docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md)), under two conditions:

- **Thinking on** (the model's own default — no override sent)
- **Thinking off** — `chat_template_kwargs: {"enable_thinking": false}` sent explicitly

Both conditions: 1536-token cap, non-streaming, direct HTTP against `/v1/chat/completions`, speculative decoding (`draft-mtp`) active throughout since that's the production config regardless of thinking mode.

### Results — one-line verdict, full table lives in 07

Thinking **on**: 7/11 correct. Thinking **off**: 11/11 correct. Every thinking-on failure was the same shape: the model ran out the entire 1536-token budget still inside an unclosed `<think>` block, never reaching an answer — 2 of the 5 prompt types (the Fibonacci trace and the list-comprehension task) failed this way on every seed. Generation speed was near-identical either way (~39 tok/s on vs ~41 tok/s off) — thinking mode's cost is wall-clock time from extra tokens generated before an answer even starts, not a per-token throughput penalty. Draft acceptance (speculative decoding) was modestly better with thinking off (~87% vs ~83%). Full numbers: [docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md).

This is the same failure shape already logged as "degenerate stalls" / "false tool-refusal" elsewhere in this project's troubleshooting history ([docs/08-troubleshooting-and-incidents.md](08-troubleshooting-and-incidents.md)) — a plain single-turn, no-tools prompt reproducing it here is stronger evidence that thinking mode is a real contributor to that failure class in general, though it is **not** proof it's the specific cause of the tool-calling variant of that bug — a separate synthetic tool-calling test with a small token budget did not reproduce a stall under either condition. The real multi-turn/tool-heavy bug likely needs context this synthetic test didn't replicate.

### Verdict and what actually shipped

**`--reasoning off` is the `gpu-server-hard` startup default** (`configs/gpu-tier/default.nix`'s `ExecStart`, see the flag's own inline comment there). The reasoning: this benchmark showed no upside for typical short requests and a real, repeatable convergence failure on two prompt shapes, so "off by default, opt in per-request when you actually want visible deliberation" is the safer posture — especially important for any client (see below) that has no way to request thinking mode per-message at all and would otherwise silently inherit whatever the server defaults to.

The per-request override was wired into both clients so the default doesn't mean thinking is unavailable, just off unless asked for:

- **nvim / codecompanion**: `enable_thinking` is a schema field on the `gpu_hard` adapter (`configs/nvim/nvim-config.nix`), `mapping = "body.chat_template_kwargs"` — codecompanion's `Client.merge_body` merges this straight into the top-level request JSON, the same mechanism the built-in `gemini.lua` adapter uses for its `thinkingLevel` setting. It shows up as an editable boolean in the chat buffer's settings block (requires `display.chat.show_settings = true`, also set in that file — codecompanion hides the settings block by default with no per-chat keymap to reveal it otherwise). Defaulted to `false`, per the benchmark. See [docs/05-nvim-integration.md](05-nvim-integration.md) for the rest of that adapter.
- **OpenCode**: has **zero** client-side control over thinking mode by design — confirmed by proxying `local-gpu`'s traffic through a `socat` TCP relay and inspecting the real outgoing request JSON. `@ai-sdk/openai-compatible` never sends `chat_template_kwargs`, `reasoning_effort`, or any reasoning-control field, for the main model or for `small_model`'s background calls. The `--thinking` CLI flag only controls whether OpenCode *displays* reasoning blocks it receives, not whether the model generates them; `--variant` only applies to catalog models with predefined provider variants (this project's custom `local-gpu` model has none, so it's a no-op); a model's `"reasoning": true` config field is a capability flag, not a switch. The fix that actually works: OpenCode's per-model `options` object is a raw passthrough merged directly into the outgoing request body with no whitelist — confirmed by probing several arbitrary keys at once and watching them all appear verbatim in the proxied request. `configs/opencode/opencode.json` defines a second, explicitly-selectable model, `qwen3.6-35b-a3b-gpu-thinking`, identical to the default except `"options": {"chat_template_kwargs": {"enable_thinking": true}}`. Selecting it (`opencode run ... -m local-gpu/qwen3.6-35b-a3b-gpu-thinking --thinking`, the `--thinking` flag needed too, for display) produces real, visible deliberation for planning/design work where it genuinely helps, while the default model stays fast and non-thinking for everyday tasks. See [docs/06-opencode-integration.md](06-opencode-integration.md).

**A practical rule of thumb that came out of this** (documented directly in the nvim adapter's comments): if you'd be satisfied with the model's first reasonable answer, leave thinking off. For a tool-calling chain specifically, leave it off too by default — the tool-calling case was never itself part of this benchmark (only plain single-turn prompts were tested), and tool-chains are exactly where the degenerate-stall/false-refusal failure mode already shows up. Instead, work in two steps: gather info and draft a result with thinking off, then flip it on for one focused follow-up turn if the result looks shallow, asking the model to double check what it already produced.

## Investigation 2: does preserving thinking across turns help multi-turn debugging?

### Why this got investigated

`preserve_thinking`'s stated purpose — per its behavior in the template — is exactly the kind of task this project's coding-assistant use case cares about most: iteratively debugging code across many messages without losing the internal reasoning trace built up earlier in the conversation. So rather than a synthetic on/off toggle check, it was tested for that actual purpose, per explicit direction.

### Method

A scripted, deterministic 6-turn synthetic conversation: a classic Python race-condition bug hunt (a shared unsynchronized counter, log evidence to work through, a widening-the-race-window red herring, a false alternative-cause red herring, and a final multi-core-vs-single-core confirmation question). A system prompt forced terse (≤20-word) visible answers on every turn, so nearly all of the state needed to reach the correct final answer had to live in each turn's hidden reasoning, not in the visible conversation history either condition could see — this is what makes the test actually probe whether preserved reasoning does anything, rather than just measuring whether the model can re-derive things from short answers alone.

2 seeds × 2 conditions: `preserve_thinking: true` vs `false`, both with `enable_thinking: true` (preservation is meaningless with thinking off — there'd be nothing to preserve). Each condition maintained its own growing message history including `reasoning_content` on historical assistant messages, so both conditions had *identical information available* to them — the only variable being tested was whether the template was told to re-render that reasoning back into the prompt for a later turn.

### Results — one-line verdict, full table lives in 07

**No measured quality benefit, a large confirmed context cost, and a hint of instability:**

- **Quality**: all 4 conversations — both conditions, both seeds — independently reached the *same* correct root cause, the *same* correctly-ordered evidence chain, and correctly rejected both red herrings, visible directly in each one's final-turn reasoning trace. Discarding reasoning each turn and reconstructing from the short visible answers worked exactly as well as preserving it, on this test.
- **Context cost**: summing tokens carried forward into turn 6's prompt, preserve-mode's two seeds carried **5,721 and 2,776 tokens**; discard-mode's two seeds carried **~125 tokens** — a **20–40x context multiplier** from just five turns of a deliberately short test. At this project's 32768-token context ceiling (which is not straightforwardly raisable — see the GPU driver instability documented in [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) / [docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md)), this growth rate would plausibly consume the entire context window before a real 10+ turn debugging session even finished.
- **Suggestive instability**: one of the two preserve-mode seeds got stuck hitting the token cap on 4 consecutive turns once reasoning had accumulated (the other conditions only hit the cap intermittently). Not conclusive on 2 seeds alone, but consistent with a compounding spiral — more carried-forward reasoning producing more reasoning generated per subsequent turn, not more efficient answers.
- **Separate, orthogonal finding, noted so it isn't confused with the above**: 3 of the 4 conversations (including both discard-mode seeds) never produced a visible final answer at all on the last "give full detail" turn — they ran out of a 1400-token cap mid-reasoning. This is a `max_tokens` budget issue under thinking mode's general verbosity, and it happened regardless of `preserve_thinking` condition — it's evidence about thinking-mode verbosity in general, not about preservation specifically.

Full per-seed numbers: [docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md).

### Verdict

**Rejected.** `--reasoning-preserve` is not enabled on `gpu-server-hard`, and `preserve_thinking: true` is not set anywhere in this project's client configs. The reasoning: no measured benefit for the exact multi-turn coherence task it's meant to help with, a confirmed 20–40x context-cost multiplier even on a short test, and a suggestive (if not conclusively proven) instability signal — none of that is worth trading away context headroom on a model already capped at 32768 tokens. This is logged as resolved-negative, not left open: if you're tempted to flip `--reasoning-preserve` on for your own longer conversations, know that this is the actual data behind why this project didn't.

## Summary table

| Mechanism | Level | Confirmed working? | Default on this stack |
|---|---|---|---|
| `"reasoning": "off"` (bare request field) | per-request | **No — confirmed no-op** | n/a |
| `chat_template_kwargs: {"enable_thinking": bool}` | per-request | Yes | not sent → falls back to server default (`off`) |
| `chat_template_kwargs: {"preserve_thinking": bool}` | per-request | Yes | not sent by any client in this project |
| `--reasoning [on\|off\|auto]` | server-wide default | Yes | `off` (`configs/gpu-tier/default.nix`) |
| `--reasoning-preserve` | server-wide default | Yes (untested here beyond the rejection benchmark) | not set (preservation rejected) |

## Where this is wired in this repo

- Server-wide default: `--reasoning off` in `ExecStart`, [configs/gpu-tier/default.nix](../configs/gpu-tier/default.nix).
- Template mechanics for both `enable_thinking` and `preserve_thinking`: [configs/gpu-tier/chat_template.patched.jinja](../configs/gpu-tier/chat_template.patched.jinja) (lines ~100 and ~149).
- nvim per-chat toggle: the `enable_thinking` schema field and `display.chat.show_settings`, [configs/nvim/nvim-config.nix](../configs/nvim/nvim-config.nix) — see [docs/05-nvim-integration.md](05-nvim-integration.md).
- OpenCode explicit thinking-mode model: `qwen3.6-35b-a3b-gpu-thinking` in [configs/opencode/opencode.json](../configs/opencode/opencode.json) — see [docs/06-opencode-integration.md](06-opencode-integration.md).

## What's untested

- Thinking mode's effect specifically inside a long tool-calling chain was not benchmarked here — only plain single-turn, no-tools prompts were. Treat the tool-chain guidance above as a reasonable inference from the degenerate-stall pattern observed elsewhere, not as data from this benchmark.
- `preserve_thinking`'s cost growth was only measured across a deliberately short 6-turn synthetic conversation; the 20–40x multiplier is what this test produced, not a universally-derived constant — a different conversation shape could scale differently, though there's no reason from the mechanism to expect it to scale *better*.
- Neither investigation was repeated on the `Qwen3-30B-A3B` fallback model or on any other model — these results are specific to `Qwen3.6-35B-A3B` at the quantization and inference stack pinned in [configs/shared/versions.md](../configs/shared/versions.md).
