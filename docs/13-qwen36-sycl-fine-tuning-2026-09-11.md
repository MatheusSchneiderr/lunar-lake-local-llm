# Fine-Tuning Qwen3.6-35B-A3B on SYCL to Its Peak

[Chapter 12](12-sycl-reversal-and-qwen36-migration-2026-09-10.md) ended
with Qwen3.6-35B-A3B IQ1_M promoted to production on a from-scratch
llama.cpp-SYCL build. This chapter is the fine-tuning pass that followed:
a batch-size sweep, a real production incident that forced a full
root-cause investigation into the model's own runaway-reasoning failure
mode, a Flash Attention/KV-cache investigation that came within one flag
of near-OOMing the machine, and three more optimization attempts that were
each tested honestly and rejected. It ends with every open question from
chapter 12's promotion resolved and a config that's been fully
interrogated rather than assumed.

---

## 0. A concurrency bug found first, because it distorts everything downstream

Before any tuning could be trusted, a real source of noise had to be
found: the first `-b`/`-ub` sweep run showed two concurrent generation
slots on the server, dragging decode speed down to ~5-10 tok/s (vs. the
established ~17-18 tok/s single-stream baseline) during the overlap.
Captured via a logging proxy: `opencode` automatically fires a
session-title-generation call (system prompt: "You are a title
generator...") on every new session, *concurrently* with the real task's
first message, via its `small_model` config. Our `opencode.json` had
`small_model` pointing at the same GPU backend as the real model — two
calls competing for one GPU on every fresh session.

**Fix**: point `small_model` at a deliberately unreachable port
(`no-titlegen/x` → `127.0.0.1:8999`) so the title-gen call fails fast and
silently instead of contending for GPU time. This is a real,
built-in `opencode` behavior worth knowing about for anyone wiring up a
similar local backend — it doesn't show up until `small_model` and the
real model point at the same place.

## 1. `-b`/`-ub` batch sizing: a fresh, independent sweep

`-ub` is the actual GPU compute chunk size (physical batch); `-b` only
needs to satisfy `b ≥ ub` for logical buffering. The two are not meant to
be set equal, and this sweep tested only genuinely asymmetric candidates
against the real `csharp_notesapi` task (build → 7-instruction filter/
sort/paginate task → verify), each preceded by a `systemctl stop`/`start`
to eliminate cross-run residue, run under thinking-off to keep results
uncontaminated by reasoning-length variance (see section 3 for why that
variance matters):

| Candidate | Wall | Decode tok/s | Prefill tok/s |
|---|---|---|---|
| default (`-b 2048 -ub 512`) | 142s | 18.21 | 70.3 |
| `-b 2048 -ub 1024` | 123s | 18.41 | 59.4 |
| **`-b 4096 -ub 2048`** | **101s** | 17.81 | **117.6** |

**Winner: `-b 4096 -ub 2048`** — ~29% faster wall time than default,
~1.7x prefill throughput, decode ~2% lower (noise-level, irrelevant given
the wall-time gain). Applied to production.

## 2. codecompanion parity: one toggle, one preset

`opencode` already supported per-request sampling overrides via its
native `options` field on each model entry. `codecompanion.nvim` needed
the same capability — a single `thinking: true/false` per-chat setting
that applies a *full* preset (temperature, top_p, top_k, min_p,
presence_penalty, and the `enable_thinking` template flag) at once,
rather than five fields to hand-edit every time.

A real bug was caught before deployment by reading `codecompanion`'s
adapter source directly (`adapters/http/init.lua`'s
`map_schema_to_params()`): a schema field's `mapping` string is the
**parent container path**, and the schema key name becomes the **leaf**.
The first attempt used `mapping = "meta.thinking"` with schema key
`thinking`, which would have produced `self.meta.thinking.thinking` —
wrong, and silently so (no error, just a value nothing ever reads).
Fixed to `mapping = "meta"`, which correctly lands the boolean at
`self.meta.thinking`. A custom `form_parameters(self, params, messages)`
handler then reads that throwaway field, deletes it, and injects the full
preset — confirmed via source inspection that `map_schema_to_params()`
runs *before* `form_parameters` is called, so the schema-mapped value is
already present when the handler reads it.

## 3. The runaway-thinking investigation

### The incident

During the `-b`/`-ub` sweep's `-b 2048 -ub 1024` candidate, a real
`opencode` task generated 9800+ tokens with **zero visible output** —
consistent with the model being stuck entirely inside its `<think>`
block and never reaching an answer. This directly matched a failure mode
found earlier in an isolated synthetic probe (tiny `max_tokens`, empty
content, `finish_reason=length`) that had been dismissed at the time as
"not a realistic production risk" because three earlier full comparison
runs never showed it. **The incident proved that dismissal wrong**: a
large `max_tokens` budget doesn't prevent runaway thinking, it just
delays hitting the wall. The failure is real, rare, and stochastic — not
systematic, but not hypothetical either.

### Two wrong assumptions found while investigating

Querying the live `/slots` endpoint during the incident revealed the
server's actual effective sampling temperature was **1.0**, not the
assumed 0.6 or llama-server's own binary default of 0.80 — this comes
from the model's own embedded GGUF `generation_config.json` metadata,
which both the command line and `opencode`'s request left unset,
letting llama-server fall through to whatever the model itself
recommends. This also corrected an "opencode always sends temperature
0.0" assumption carried over from an old, unrelated OpenVINO wrapper
script — never re-verified against the real SYCL production path. A
logging reverse-proxy directly captured a real `opencode run --auto`
request and found **no `temperature` field at all**, and `max_tokens:
32000`.

### Root cause research

llama.cpp has no adaptive/scaling reasoning-budget mechanism — only a
fixed `-1`/`0`/N. The actual root cause, confirmed via Qwen's own GitHub
issues and the Qwen3.6-35B-A3B HF discussion page, is *paraphrastic*
(semantically-reworded, not literal) self-repetition suppressing the
`</think>` token's own probability — a documented, model-family-wide
Qwen3/3.5/3.6 issue, not something specific to this deployment. This is
why `repeat_penalty`/DRY sampling don't catch it: they only detect
literal, not semantic, repetition. Aggressive quantization is separately
flagged in the literature (arXiv:2606.25519, "Token Inflation") as a
plausible contributor — low-bit quantization measurably degrades a
model's ability to emit rare tokens like `</think>` at the right moment,
and the paper states directly that aggressive quantization "can lead to
models entering reasoning loops." IQ1_M, our chosen quant, sits at
exactly that end of the spectrum.

### The presence_penalty A/B, done properly

The user's explicit demand at this point: presets with **proven,
validated** results, not further ad-hoc parameter mixing. Research
turned up no single validated combined preset anywhere (checked Cline,
Roo Code, Continue.dev, aider, Kilo Code, vLLM, SGLang) — but did surface
Qwen's own troubleshooting note recommending `presence_penalty=1.5`
specifically for "significant endless repetitions." A direct empirical
comparison against `presence_penalty=1.0` (a research-suggested
midpoint) settled it on real data instead:

| Configuration | Wall times |
|---|---|
| Thinking-on, presence_penalty=1.0 | 126s, 172s |
| Thinking-on, presence_penalty=1.5 | 246s, 219s, 217s |
| Thinking-off, presence_penalty=1.0 | 106s |
| Thinking-off, presence_penalty=1.5 | 57s (fastest of everything tested) |

A real wrinkle surfaced mid-comparison: every thinking-on run at *any*
`presence_penalty > 0` hit at least one self-corrected first-try build
error (a different bug each time), while the one
`presence_penalty=0` baseline run built clean on the first try — but took
**454s**. This first looked like "presence_penalty makes the model
careless." Rerunning to test that hypothesis directly (same config,
repeated) produced 4 failures out of 4 attempts at `presence_penalty >
0`, against 1 clean run at 0 — but the reframing that actually explains
the data is the user's own: *they all failed, but reasoned faster.* Net
wall time still favored `presence_penalty > 0` even with an extra
self-correction round folded in, because the correction was cheap
relative to the time saved by not reasoning as long in the first place.
Total run-to-run variance at a *fixed* setting was comparable to the gap
between 1.0 and 1.5 (noisy data, genuinely), but the direction was
consistent every time tested: 1.0 beat 1.5 for thinking-on.

**Final decision: `presence_penalty=1.0` for thinking-on,
`presence_penalty=1.5` for thinking-off** (matching Qwen's own documented
non-thinking preset — no reason to deviate there; overall fastest
configuration found in this entire investigation). Reflected in both
`opencode.json`'s two model entries and `codecompanion`'s `thinking`
toggle.

## 4. Flash Attention and KV-cache quantization

### FA on/auto with quantized KV: slower, not faster

The obvious next lever, KV-cache quantization, hard-requires Flash
Attention for a quantized V-cache (`-fa off` with `-ctk q4_0 -ctv q4_0`
errors immediately: "quantized V cache requires flash_attn to be
enabled" — confirmed directly, this is not a soft warning). Testing
`-ctk q4_0 -ctv q4_0` with the full production flag set for a fair
comparison gave **363s** and **396s** (FA-on and FA-auto respectively) —
both markedly slower than the 126-246s range already established with no
KV quantization at all. **Rejected**: real memory savings, but a real
speed cost with no offsetting benefit for this model/hardware/context
combination.

### FA off: not a viable alternative, at any context size that matters

Testing `-fa off` (unquantized f16/f16 KV) at production's real `-c
131072` **crashed with a SYCL memcpy error** during the model-load
warmup decode — not an OOM (28GB RAM was still free at the moment of the
crash). It ran cleanly at a much smaller `-c 8192`. Halving the gap to
`-c 98304` (96K) to see where the wall actually sits: the server started
and ran, but RAM usage climbed to **29GB/30GB used, only 1.0GB available
system-wide** — a genuinely dangerous near-OOM state, caught live and
killed proactively before it could crash the whole session.

This confirms the mechanism, not just the symptom: standard (non-FA)
attention's intermediate score buffer scales **O(n²)** with context
length, where Flash Attention's is O(n). That quadratic blowup is what
makes `-fa off` fundamentally incompatible with this 30GB machine at any
context length in the actual target range (120-150K+ tokens) — not a
narrow edge case, not a tunable memory budget, just structurally
unworkable here. **Flash Attention (already silently active via the
default `auto` setting) is mandatory for this deployment, not optional.**

## 5. Three checks that resolved to "no change needed" or "doesn't apply"

Not every investigation ends in a new flag. Three didn't, and each is a
real result in its own right.

### mmap double-allocation: already fixed, always had been

`--load-mode` (the modern replacement for the deprecated `--mmap`/
`--no-mmap`/`--mlock` flags) defaults to `auto`, defined as "mmap, unless
a device does not support it." Verified directly on the live production
process rather than trusted from the flag description alone: RSS sat at
~1.1GB against the model's ~10GB file size, with the model's bytes
living in `buff/cache` (~8.9GB, matching the GGUF's size) instead —
textbook mmap behavior, no double allocation. Since this was never
broken, there was no meaningful "before" state to benchmark against —
every number in this chapter and the last already reflects the
"with mmap" condition. **Lesson**: confirm a flag's actual effect on the
live process before spending a test run "verifying" something that was
never in question — a redundant comparison test was initially started
here and correctly cancelled once this was pointed out.

### `--cache-reuse`: architecturally incompatible, not a config oversight

`--cache-reuse` defaults to `N=0` (off) and was never enabled in
production — only the separate, always-on `--cache-prompt` (basic
longest-common-prefix caching) was active. Research turned up a direct
hit: a llama.cpp maintainer, in a discussion about this exact model on
this exact use case (`ggml-org/llama.cpp#22354`), confirms
**Qwen3.6-35B-A3B uses recurrent-state attention layers** (a
Qwen3-Next-style hybrid mixing standard attention with Gated-DeltaNet-
style recurrent/linear-attention layers). `--cache-reuse`'s KV-shifting
mechanism fundamentally can't apply to recurrent state — it has to be
recomputed sequentially, not sliced — and another user in that same
discussion confirmed no benefit at either `256` or `2048` on this exact
model.

Worth stating precisely, since it's an easy conflation: this is an
**attention-mechanism** limitation, not a MoE one. MoE routing lives
entirely in the feed-forward/expert layers and has nothing to do with
the KV cache; a MoE model built on plain standard attention throughout
would have `--cache-reuse` work completely normally, and a dense model
built on the same recurrent-attention design would hit the identical
wall. Qwen3.6-35B-A3B happens to be both MoE and hybrid-attention — it's
the second property that rules this out.

### `GGML_SYCL_F16`: a documented dense-model win that didn't transfer

This build flag switches the SYCL backend's dequantize-before-matmul
intermediate type (`dfloat`/`dfloat2` in `ggml-sycl/common.hpp`) from
fp32 to fp16, letting Xe2's XMX matrix units accelerate the compute-bound
prefill step. Real community benchmarks on Arc B580/B570 hardware show
this working — one configuration went from 388 to 1355 tok/s prefill
(3.5x) — but every clean number found was on **dense** models
(Llama-2-7B, Llama-3.1-8B, Qwen2.5-7B); no verified isolated benchmark
existed for a MoE model before this test, and one B580 configuration
actually *regressed* slightly (2063→1954 tok/s), so it isn't even a
universal dense-model win.

Tested directly: full rebuild (426s, one cmake flag flip forces a
complete recompile — Nix builds are hermetic, no partial cache reuse
across a derivation hash change), then a clean sequential A/B (production
stopped, test binary alone, run, stop, restore production — running two
full model instances side-by-side first drove the machine to **100%
swap**, the same danger class as the FA-off near-OOM above, and was
abandoned immediately in favor of testing one binary at a time):

| Preset | Baseline (fp32 dequant) | `GGML_SYCL_F16` | Delta |
|---|---|---|---|
| Thinking-on | 172s | 264s | **+53% slower** |
| Thinking-off | 57s | 97s | **+70% slower** |

Both builds succeeded first-try both times — not noise from
error-recovery. **Rejected outright**, not a close call. Reverted
immediately; the rebuild-back was a pure Nix store cache hit (10s, no
recompile). Best available explanation: the dense-model prefill win
doesn't transfer to MoE because the per-token active-expert compute is
already smaller to begin with, so the dequant-step overhead this flag
adds isn't offset by the same prefill speedup a dense model sees.

**Lesson (process hygiene, not results):** testing this safely on a
30GB shared-memory machine meant never running the new and old binaries
concurrently — the first attempt to do exactly that (side-by-side on
different ports, to avoid a production outage during testing) was the
one that hit 100% swap. The safer pattern that actually worked: stop
production, test the alternative alone, restore production, repeat for
the next test. A related trap resurfaced here too: launching the new
binary via a bare `nohup` without the systemd service's `LD_LIBRARY_PATH`
and `OCL_ICD_VENDORS` environment variables produced the same silent
"no usable GPU found → falls back to CPU" failure noted in chapter 12's
`ONEAPI_DEVICE_SELECTOR` trap — always replicate a systemd service's full
`Environment=` block when reproducing its behavior manually.

## 6. Final production configuration

```
llama-server -m Qwen3.6-35B-A3B-UD-IQ1_M.gguf -ngl 99 -c 131072 \
  -b 4096 -ub 2048 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
  --port 8901 --chat-template-file <qwen3.6-35b-a3b-chat-template.jinja>
```

`--temp`/`--top-p`/`--top-k`/`--min-p` are the server-wide safety-net
default (Qwen's "precise/coding" thinking preset) for any client that
doesn't send its own sampling values — currently only `codecompanion`
needs this fallback, since `opencode` always sends explicit per-request
overrides. `-fa` and `-ctk`/`-ctv` are deliberately absent (default
`auto`/`f16`/`f16` — the only combination that worked at this context
size). `--cache-reuse` and `GGML_SYCL_F16` are deliberately absent, not
overlooked — both were tested and rejected above.

Per-client sampling presets (full detail in `opencode.json` and
`codecompanion`'s `thinking` toggle — see
[configs/opencode/](../configs/opencode/) and
[configs/nvim/](../configs/nvim/)):

| | Thinking-on | Thinking-off |
|---|---|---|
| temperature | 0.6 | 0.7 |
| top_p | 0.95 | 0.8 |
| top_k | 20 | 20 |
| min_p | 0 | 0 |
| presence_penalty | 1.0 | 1.5 |

## 7. Q&A: clarifying questions asked during this investigation

**Why not just always run thinking-off, since it was fastest in every
comparison?** It was — 57s was the single fastest result in this whole
chapter. This wasn't changed to a global default because both modes stay
available and useful for different task shapes; the tuning here makes
whichever one is chosen per-request as fast and stable as each can be,
rather than picking a winner and deleting the loser.

**Is Flash Attention's correctness on this exact backend fully verified
now?** No — see chapter 12, section 2. FA is confirmed *mandatory* here
(FA-off is unworkable at any real context size), but its correctness
under this specific hardware/kernel combination is inherited as an open
caveat from the earlier chapter, not independently re-verified by
anything in this chapter.

**Does any of this need revisiting if the model or quant changes?** Yes,
specifically the runaway-thinking mitigation (presence_penalty values)
and the FA/KV-cache findings — both are stated in the literature and
confirmed here to be at least partly quantization- and
architecture-dependent. The batch-size (`-b`/`-ub`) and mmap findings are
more likely to transfer to a different model on the same hardware.

## Status at time of writing

Every item on the post-promotion fine-tuning checklist has been tested
and resolved. Qwen3.6-35B-A3B IQ1_M on llama.cpp SYCL, with the
configuration above, is deployed to production and is the confirmed
daily driver.
