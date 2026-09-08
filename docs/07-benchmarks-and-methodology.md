# Benchmarks and Methodology

This chapter is the data backbone for the rest of the guide. Every tuning
decision referenced elsewhere — the NPU model choice in
`configs/npu-tier/npu-server/default.nix`, the GPU backend and flag set in
`configs/gpu-tier/default.nix` — was arrived at empirically, on the exact
hardware and software stack pinned in `configs/shared/versions.md`, not
picked from a vendor spec sheet. This page is where the actual numbers live;
other chapters link back here instead of restating them.

One story here (KV cache quantization) is deliberately told as a mistake
followed by a correction, because the mistake is a generic trap anyone
benchmarking local LLM inference can fall into, not just something specific
to this project. If you read nothing else on this page, read that section.

None of the thinking-mode/`preserve_thinking` benchmarks live here — they're
a large, self-contained topic with their own methodology and belong entirely
in [docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md).

---

## 1. NPU model bake-off: Qwen2.5-Coder-7B-Instruct vs. DeepSeek-R1-Distill-Qwen-7B

Before settling on the NPU tier's model (see
[docs/02-npu-tier-setup.md](02-npu-tier-setup.md)), both 7B-class candidates
were run through a fixed set of **15 debugging prompts** — a mix of
"classic gotcha" bugs (off-by-one errors, mutable default arguments, closure
capture in loops) and algorithmic/tracing bugs (manually tracing recursive
or stateful code to find where behavior diverges from intent) — the kind of
task this NPU tier exists for. Each response was scored for correctness
(did it actually find/fix the bug) and timed for throughput.

| | Qwen2.5-Coder-7B-Instruct | DeepSeek-R1-Distill-Qwen-7B |
|---|---|---|
| Correct on the 15-prompt debugging set | **9 / 10** | 4 / 10 |
| Result | Won decisively on both speed and correctness | Clearly weaker on this hardware/task combination |

(The score is reported as "x / 10" in the source log despite 15 prompts run
— treat it as the pass rate on the scored subset; the decisive gap, not the
denominator, is the point.)

**Verdict: Qwen2.5-Coder-7B-Instruct wins outright.** This is the model
actually wired into `npu-server-coder` — see the comment block directly
above `systemd.user.services.npu-server-coder` in
`configs/npu-tier/npu-server/default.nix`, which records this exact result
next to the service definition it justifies. No further NPU dense-model
candidates were found worth re-running this bake-off against — see
docs/02 for why the search for a newer NPU-confirmed coding model came up
empty.

---

## 2. GPU backend bake-off: llama.cpp+Vulkan vs. OpenVINO GPU-MoE

This is the pivot that determined the entire GPU tier's runtime (see
[docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) for the full narrative and
wiring).

Before this test even ran, **fully-resident OpenVINO GPU-MoE inference was
already ruled out on throughput alone**: the closest available reference
(a community benchmark on essentially this same hardware generation - Core
Ultra 7 258V, Arc 140V, 32GB unified RAM, *more* than this machine has)
showed the model fully resident collapsing to **1.4-1.7 tok/s** from
paging - unusable, and never run through a correctness suite at all as a
result. `OFFLOAD_RATIO` (OpenVINO's GPU-plugin MoE expert-streaming
feature) was the one variant that looked potentially viable (research
estimated 10-21 tok/s), so it's the one actually taken into a matched
correctness comparison against `llama-cpp-vulkan`, both serving the same
model, **`Qwen3.6-35B-A3B`**, under: **10 prompts, 4000-token budget,
thinking mode on** (required, since this model reasons by default),
temperature 0.6 / top_p 0.95 / top_k 20 sampling for both.

| | OpenVINO int4 (GPU, `VLMPipeline`, `OFFLOAD_RATIO=20`) | llama.cpp Q4_K_M (Vulkan, `-ngl 99`, full residency) |
|---|---|---|
| Converged to a complete, correct answer | **0 / 10** | **9 / 10** (1 cut off one line short of the end) |
| Never closed `<think>`, ran out the 4000-token budget | 6 / 10 (6,727-14,877 characters of unclosed rambling) | 0 / 10 |
| Crashed outright (`CL_OUT_OF_RESOURCES`) | 4 / 10 - worsened across repeated large generations within one process, eventually killing the whole process on exit | 0 / 10 |

Greedy decoding and the model's own recommended sampling parameters were
both tried first on the OpenVINO side as a way to rule out "bad sampling"
as the cause of the non-convergence - neither helped; every OpenVINO run
still spiraled into repetitive self-second-guessing rather than answering.

**Root-cause split confirmed by a side experiment**: a single-prompt spot
check of `llama-cpp-vulkan` against the exact same `Qwen3.6-35B-A3B` Q4_K_M
GGUF (~27 tok/s, clean and complete) proved the model itself is fine — it's
OpenVINO's GPU-plugin MoE inference path (both the always-resident and the
`OFFLOAD_RATIO`-streamed expert-offload variant) that's broken on this
stack, not the model. OpenVINO GenAI's native GGUF-direct-load feature was
also checked as a possible way to get a real Q4_K_M quant onto the OpenVINO
runtime instead — ruled out because it only supports SmolLM/Qwen2.5
topologies as of this OpenVINO version; loading a `qwen3moe`-architecture
GGUF crashes immediately (`IndexError: unordered_map::at`) on both CPU and
GPU backends.

**Verdict: drop OpenVINO as the GPU-tier delivery mechanism entirely.**
`llama-cpp-vulkan` + a genuine Q4_K_M GGUF, run fully resident
(`-ngl 99`), is the confirmed working path — this is exactly what
`configs/gpu-tier/default.nix`'s `mkGpuService` runs today, and the
rationale comment directly above it in that file cites this same 9/10-vs-crash
result.

---

## 3. `-ub` (physical batch size) sweep

**Symptom that motivated this test**: prompt-processing (prefill)
throughput on the Vulkan iGPU backend degrades badly as context length
grows, and KV-cache reuse between turns is inconsistent in real
codecompanion sessions (observed longest-common-prefix similarity as low as
0.362 vs. 0.974 on adjacent turns) — meaning a long tool-heavy conversation
can trigger a near-full ~20K-token reprocess that looks like a hang (3+
minutes) even though the server is genuinely still working.

`-ub` (llama.cpp's physical/micro batch size for prompt processing, default
512) was swept against a **real ~30K-token prompt** pulled from an actual
long conversation, not a synthetic filler prompt, with a fixed **300-second
budget per value**:

| `-ub` | Completed within 300s? | Full-prompt average throughput |
|---|---|---|
| 512 | No | — |
| 1024 | No | — |
| 2048 | No | — |
| **4096** | **Yes** | **102.29 tok/s** (full-prompt average) |
| 8192 | No | — (no faster than 4096 even where it did progress) |

**Verdict: `-ub 4096` is the measured ceiling on this iGPU.** 512/1024/2048
all needed longer than the budget to finish the same prompt; 8192 bought
nothing over 4096 and also didn't finish in the same window, meaning bigger
batches stop helping past this point on this hardware — this is a
measured plateau, not a value picked by pattern ("powers of two feel safe").
This does not eliminate the prefill slowdown on long contexts, it just
meaningfully reduces it. `-ub 4096` is what's live in
`configs/gpu-tier/default.nix`'s `ExecStart` line today.

---

## 4. Speculative decoding (`--spec-type draft-mtp`)

This GGUF ships its own built-in Multi-Token-Prediction ("nextn") head —
the `blk.N.nextn.*` tensors that llama.cpp logs as "unused" on every model
load without this flag turned out to be exactly that. `--spec-type
draft-mtp` activates self-speculation directly against the same model file:
no separate draft model to source, convert, or keep in sync.

Verified across **4 varied requests** (3 different debugging prompts + one
tool-calling request), `--spec-draft-n-max 3`:

| Metric | Result |
|---|---|
| Generation throughput | **35–39 tok/s**, vs. **~26.6–27 tok/s** baseline (no speculation) — **+30–45%** |
| Draft acceptance rate | **68–84%** |
| Mean accepted draft length | ~3–3.5 tokens |
| Correctness regressions | None — tool-call formatting stayed clean, and the tool-calling request had the *best* acceptance rate of the batch |
| Memory cost | ~2GB extra for the draft context (comfortably absorbed) |

**Verdict: enabled permanently.** `--spec-type draft-mtp
--spec-draft-n-max 3` is part of the production `ExecStart` in
`configs/gpu-tier/default.nix`.

---

## 5. KV cache quantization: how NOT to benchmark LLM inference, and how to fix it

This is the flagship methodology story of this whole guide, because the
mistake made here is common and easy to repeat, and the fix generalizes
well beyond this one flag.

### The setup

KV cache quantization was tested to claw back context-window headroom,
given the 32768-token ceiling established in the previous section (raising
context further was tried and rejected — see the sidebar below). Three
configurations were compared, all with `-fa on` (Flash Attention is
required to run a quantized V-cache at all):

- **Baseline**: `f16` K / `f16` V (no quantization)
- **Symmetric**: `q8_0` K / `q8_0` V
- **Asymmetric**: `q8_0` K / `q4_0` V

**Correctness** was the same for all three: an 18K-token needle-in-haystack
test (a planted code buried in filler context) was correctly retrieved at
every quantization level — no accuracy loss from KV quantization at any
level tested here.

### The mistake

The first comparison pass ran **one, unseeded, stochastic generation per
configuration** and compared the resulting numbers directly. The result
looked dramatic and clean: `q8_0/q4_0` appeared to decisively beat
`q8_0/q8_0` on every axis measured — speed and draft acceptance both.

This was wrong, and it was caught before it shipped into the config for a
good reason: it didn't match prior expectations. Public benchmarks of
asymmetric 4-bit V-cache quantization generally show it as *slightly worse*
than symmetric 8-bit, not better — a result that reverses the expected
direction and claims a clean sweep on every metric is exactly the shape of
result that should trigger suspicion, not confidence, in any benchmark
involving sampled generation.

**Why it was wrong**: draft acceptance rate and generation-phase timing
both depend on the actual tokens sampled during generation, which are
stochastic unless a seed is fixed. A single unseeded run per configuration
doesn't isolate the effect of the KV cache format — it's one draw from a
noisy distribution per configuration, and the "decisive win" was sampling
noise dressed up as a finding.

### The fix

The retest fixed **3 seeds per configuration**, removing sampling noise as
a confound. The corrected picture was materially different from the first
pass:

| Configuration | Draft acceptance rate |
|---|---|
| Baseline (`f16`/`f16`) | ~77.3% |
| Symmetric (`q8_0`/`q8_0`) | ~78.6–78.7% |
| Asymmetric (`q8_0`/`q4_0`) | ~78.6–78.7% |

`q8_0/q8_0` and `q8_0/q4_0` are **statistically tied** with each other on
both speed and draft acceptance, both sitting a modest ~1.3 points above
baseline — nothing like the first pass's "decisive win" for the asymmetric
config. (Prefill-speed differences between configurations, by contrast,
*are* real and expected regardless of seeding — prefill is a deterministic
forward pass, not sampled, so a smaller KV cache genuinely reduces memory-
bandwidth pressure there; that part of the first pass wasn't wrong, only
the sampled-generation comparison was.)

### The actual decision, on corrected data

Since `q8_0/q4_0` ties `q8_0/q8_0` on quality and speed, the tie-breaker is
memory: `q8_0/q4_0` saves **~2GB** (26GB → 24GB used at 32768-token
context) for no measured cost once seeding is controlled for. That's the
real, defensible reason `-ctk q8_0 -ctv q4_0 -fa on` is what's live in
`configs/gpu-tier/default.nix` today — not the inflated first-pass numbers.

A natural follow-up question — does the freed 2GB fix the `-c 49152`
GPU-driver crash from the context-window experiments? — was retested
directly: the same ~43K-token prompt that had crashed at 76% progress
without KV quantization crashed again with quantization on, identical
`vk::DeviceLostError`, but noticeably later (~90%+ progress this time).
Quantization bought margin by reducing memory-bandwidth pressure, but did
not fix the underlying issue — it's much more likely tied to `-ub 4096`'s
large batch dispatches sustained over a long duration tripping the GPU
driver's hang-detection watchdog, independent of KV cache size. **32768
remains the production context ceiling** (see
`configs/gpu-tier/default.nix`'s `-c 32768`); going bigger would need a
different lever entirely (e.g. a smaller `-ub` specifically at larger `-c`,
since that's most directly tied to per-dispatch GPU job duration), not
pursued further in this project.

> **Sidebar — why not just raise `-c` instead of quantizing the KV cache?**
> `-c 49152` and `-c 65536` were both tried first, with MTP speculative
> decoding enabled the same as production. Memory scaled gradually rather
> than as a cliff (32768 → ~5.2–5.6GB available, 49152 → ~4.8GB, 65536 →
> ~4.3GB), and 65536 additionally needed raising `ANV_SYS_MEM_LIMIT` (the
> Mesa/ANV env var controlling what fraction of system RAM the Vulkan
> driver exposes as its device-local heap, default 75%) above default,
> since `vulkaninfo` showed the driver's own ~23GiB heap ceiling — not
> available system RAM — was the actual limit at that size. None of that
> mattered in the end: a real ~43K-token prefill at `-c 49152` triggered a
> genuine `vk::DeviceLostError` GPU driver crash (confirmed at the kernel
> level via `journalctl -k`: `xe ...: Tile0: GT0: Timedout job ... in
> llama-server`), the driver's hang-detection watchdog force-resetting the
> device under sustained heavy load. That's a stability wall, not a memory
> one, which is why KV cache quantization was investigated afterward as a
> narrower, lower-risk way to buy back headroom instead.

### Methodology principles

> - **Fix seeds for anything involving sampled/stochastic generation.**
>   Draft acceptance rate, generation-phase timing under speculative
>   decoding, and any accuracy metric on non-greedy sampling are all draws
>   from a distribution, not fixed numbers — a single unseeded run is not a
>   valid comparison between configurations, no matter how clean the result
>   looks.
> - **Run multiple trials.** Three seeds per configuration was enough here
>   to flip a "decisive win" into "statistically tied" — the KV-quant story
>   above is the direct demonstration of why one run isn't enough.
> - **Prefer real prompts over synthetic ones where possible.** The `-ub`
>   sweep and the GPU-driver-crash retest both used prompts pulled from
>   actual long conversations, not generated filler — the ~20K-token
>   near-full-reprocess symptom and the ~43K-token crash were both real,
>   observed failure conditions, not hypothesized ones.
> - **Isolate one variable at a time.** Each experiment on this page
>   changed exactly one axis against an otherwise-fixed configuration:
>   backend (section 2), batch size (section 3), speculative decoding on/off
>   (section 4), KV cache format (section 5) — never several at once, which
>   is what makes attributing a measured change to a specific cause
>   possible at all.
> - **Be suspicious of a result that's "too clean."** The invalid first
>   KV-quant pass showed a decisive sweep on every axis, in the opposite
>   direction from prior public benchmarks. That combination — too clean,
>   and contrary to prior evidence — is a strong signal to check your
>   methodology before trusting the number, not a signal you found
>   something surprising.

---

See also: [docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md)
for the separate thinking-mode-on-vs-off and `preserve_thinking` benchmarks
(convergence rate, context cost, and the decision to default reasoning off
server-wide) — deliberately not duplicated on this page.
