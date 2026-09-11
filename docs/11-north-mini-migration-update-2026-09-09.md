# The Best Model as of September 2026

Today's session started with a wall we kept running into — `gpu-server-hard`'s context ceiling — and ended with a smaller model in production carrying 2.67x the context window at real speed gains over the model it replaced. In between: nine ruled-out candidates, a chat-template bug found and fixed by hand-parsing a GGUF's raw bytes, a genuine Intel-Arc architectural dead end, a scientific-method detour into CPU thermal throttling that overturned one of our own earlier conclusions, and a final config change made on principle rather than a benchmark number. This is the full account.

> **Update (2026-09-10):** North-Mini-Code-1.0, the model this chapter
> lands on, failed in real production use the very next day, despite
> passing every benchmark below. That triggered a wider engine+model
> search that replaced both the model and the Vulkan backend entirely —
> see [docs/12-sycl-reversal-and-qwen36-migration-2026-09-10.md](12-sycl-reversal-and-qwen36-migration-2026-09-10.md).
> This chapter's methodology and reasoning are left intact below as a
> real, valuable account of the process, even though its production
> conclusion no longer holds.

---

## 1. Why we went looking for a smaller model

`gpu-server-hard` had been running `Qwen3.6-35B-A3B` (Q4_K_M) at `-c 24576` — itself already a step down from an earlier 32768 ceiling, sacrificed to buy back Flash Attention performance on this hardware (see doc 10). In real day-to-day use through codecompanion and OpenCode, that ceiling wasn't a comfortable margin, it was a wall we kept running into: conversations compacted far more often than felt reasonable, and at least one real request had already hard-failed outright — 24,588 tokens against a 24,576 limit, missing by twelve tokens.

The instinct at that point could have been "tune Qwen3.6 further." We explicitly rejected that. Every lever already pulled to get Qwen3.6 running acceptably (`-fa off`, unquantized KV cache, a patched chat template) was itself a concession bought at the cost of context or throughput. Squeezing more out of a 35B-total model on this iGPU meant continuing to trade against the exact ceiling we were trying to raise. What we actually wanted was a smaller model with real headroom to spare — not a differently-tuned version of the same size problem.

So the search was deliberately broad rather than a narrow shortlist: survey as many small-to-mid MoE coding candidates as we could find, re-check whether any upstream engine (SYCL, `ggml-openvino`, vLLM) had picked up fixes since they were last ruled out, and empirically test rather than trust benchmark claims — this hardware has a documented history of real numbers not matching vendor-reported ones.

---

## 2. The candidate search: what we tried, and why each one fell out

### Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-Instruct

The most obvious "smaller Qwen" candidates, and the first ruled out. Two independent problems, not one:

- A real capability regression: SWE-bench Verified ~51.6% against Qwen3.6's 73.4%.
- A real, documented Vulkan/Intel-Arc bug (the same `issue #19887`-class progressive-prefill-degradation pattern seen elsewhere on this GPU family) — prefill throughput visibly degrades as the conversation grows, independent of any flag tuning. We confirmed this held under both a vanilla, untouched config and a properly tuned one, so it wasn't a matter of finding the right flags.

### GLM-4.7-Flash

This one took real digging, in two separate rounds.

**Round one — a hard crash.** Loading it and sending a moderately long request (~2,368 tokens) reliably crashed the server with `VK_ERROR_DEVICE_LOST`. Root-caused via research to a documented Intel Arc/Xe2 Vulkan cooperative-matrix (coopmat) shader bug that collides specifically with MoE expert matmuls — `llama.cpp` issue #20554 describes the exact same GPU (Arc 140V) hitting this in the wild. The fix was a one-line environment variable:

```
GGML_VK_DISABLE_COOPMAT=1
```

That genuinely fixed the crash — the same request that died before now completed cleanly.

**Round two — a deeper, unfixable problem.** Pushing to a larger (~9K token) prompt surfaced something worse: prefill throughput collapsed progressively (206 → 75 tok/s across successive chunks) and generation speed fell to roughly 5 tok/s. This is not a driver bug — it's architectural, and it's not Intel-specific either: `llama.cpp` issue #19081 documents the identical degradation on AMD hardware, and the maintainers closed it "not planned." The root cause is GLM's unusually large attention head dimensions (512/576, versus Qwen-class models' 128/128) landing outside the head-dim values `llama.cpp`'s flash-attention kernels are actually tuned for — a CUDA-side crash trace elsewhere names the exact failing template, `flash_attn_ext_tile_case<576,512>`, confirming this isn't backend-specific.

`-fa off` avoided the crash, but at that point the model was running roughly 4x too slow to be usable. Ruled out for production regardless of flag combination. (We kept the GGUF on disk for a while in case a future Vulkan update changed the picture — that call was later reversed once North-Mini was confirmed in production, and the file was deleted along with the other also-rans.)

### Devstral Small 2 24B

This one we actually fixed, then ruled out anyway for an unrelated reason.

The bug: Devstral's own unsloth-embedded Jinja chat template only had a branch for a system message at `messages[0]`. Nowhere else in the main conversation-rendering loop did it handle a `system`-role message at all — so a *second* system message (exactly the shape codecompanion sends via its `<rules>`/`@{agent}` attachment mechanism) fell through to the template's catch-all `else` branch and raised:

```
Only user, assistant and tool roles are supported, got system.
```

Confirming this required extracting the model's *actual* embedded template directly from the GGUF — the template published on Devstral's HF page had already diverged from what was baked into the unsloth quant and didn't reproduce the crash. We wrote a small pure-Python GGUF metadata parser (no `gguf` package dependency needed) to pull the real `tokenizer.chat_template` string out of the file, found the missing branch, and patched it to render a second system message as an ordinary turn instead of raising. The fix worked — verified via a direct multi-system-message request that previously 500'd and now returns cleanly.

Ruled out anyway: consistent ~6.7-6.9 tok/s generation even at short context, well under the ~18.7-19.1 tok/s baseline. This isn't a bug, it's arithmetic — Devstral is a *dense* 24B model (all 24B parameters active on every token), while Qwen3.6 despite its larger 35B total only activates ~3B parameters per token. No amount of tuning closes that gap.

### Faster side-researches

A handful of other candidates got a research pass rather than full empirical testing:

| Candidate | Verdict | Why |
|---|---|---|
| gpt-oss-20b | Deprioritized, not ruled out | Real MoE (20.9B total / 3.6B active), Apache 2.0, native `llama.cpp` support. Genuinely viable but untested once North-Mini looked strong. |
| Llama-3.1-MoE-4x8B | Ruled out | A real but weak mergekit "frankenMoE" — its own published benchmarks (~17.5% average) are *worse* than the plain dense Llama-3.1-8B-Instruct it's built from. ~24 downloads/month. |
| Sarvam-30B | Ruled out | Same size class as the Qwen3.6 model already in use (~21.7GB), no real SWE-bench numbers published, fundamentally an Indic-language-first reasoning model with coding as a secondary concern. |
| DeepSeek-Coder-V3-Lite | Doesn't exist | DeepSeek's lineup jumps straight from Coder-V2-Lite (already dated, already ruled out) to the full 671B DeepSeek-V3. Nothing sits between them. |
| Qwopus-MoE-35B-A3B | Ruled out | A real model, but an unvetted hobbyist QLoRA fine-tune of Qwen3.5-35B-A3B using Claude-Opus-distilled reasoning data. ~186 downloads/month, zero published benchmarks. |

### The standout: Cohere North-Mini-Code-1.0

This is where the real candidate emerged. North-Mini-Code-1.0 reports a real, vendor-sourced SWE-bench Verified score of 80.2% pass@10 — beating Qwen3.6's own 73.4% — with clean native `cohere2moe` architecture support merged into `llama.cpp` (PR #24260, June 2026), and no known Vulkan bugs turned up in research.

It wasn't a clean, immediate win, though — getting it to actually perform reliably in practice took the rest of this investigation, including one genuine near-miss where it looked ready to be ruled out before we found the real fix. That's the subject of the sections that follow.

---

## 3. Taming a thinking model: reasoning budgets and sampling landmines

### The `--reasoning-budget` discovery

North-Mini-Code-1.0 thinks by default, and there's no clean way to fully suppress it that doesn't cost something. Reading the model's actual Jinja chat template turned up the real disable mechanism: passing `chat_template_kwargs: {"reasoning_effort": "none"}` sets an internal `reasoning` variable to `false`, which makes the template emit an immediately-closed `<|START_THINKING|><|END_THINKING|>` block instead of letting the model deliberate.

That full disable, though, cost real correctness. On a 5-prompt quick accuracy suite:

| Config | Result |
|---|---|
| Full disable (`reasoning_effort: "none"`) | List-comprehension question flatly **wrong** (`[8,6,4,2]` instead of `[64,16,4]` — forgot to square the numbers); 2 of 5 answers truncated at an 800-token cap |
| Default, unbounded thinking | Correct, but responses took 400+ seconds |

The fix was llama-server's own `--reasoning-budget N` flag — a hard token cap on thinking, independent of whatever the model's template does. Tested at 300–400, this let thinking happen but bounded it: the same list-comprehension question that came out wrong under full disable came out perfect and complete once thinking was capped rather than switched off, and responses stopped running away in wall-clock time.

### Sampling landmines

None of this was obvious going in, and each failure mode was found the hard way:

- **`temperature=0` (greedy)**: on a long (~17K token) real code-review prompt, the model fell into a genuine degenerate loop — one exact sentence repeated verbatim 9+ times, burning the entire token budget with nothing useful produced.
- The only documented sampling guidance for this model anywhere (the Unsloth GGUF model card) is `temperature=1.0, top_p=0.95` — not `0`, which is likely exactly why greedy decoding broke down.
- **`temperature=1.0` (the "recommended" value)**, tested at full production scale, caused a *different* and worse failure: a hard error, `"The model produced output that does not match the expected peg-native format"` — the model's own structured-output PEG-grammar parser broke under the extra sampling randomness, after 475 seconds of generation.
- **`temperature=0.2, top_p=0.95`** avoided both failure modes cleanly across repeated tests — no loops, no parser breaks, correct output — and became the validated final choice.

We also tried `repeat_penalty` as an alternative anti-repetition mechanism. At `1.15` it killed the loop but introduced a new bug: literal special tokens (`<|END_THINKING|><|START_TEXT|>`) leaking into the visible response, followed by the whole response duplicating near-verbatim. A lighter `repeat_penalty:1.05` avoided both problems too — but `temperature=0.2` alone, with no penalty at all, was ultimately preferred, since penalties carry a real (if subtle) output-quality cost by discouraging legitimate token reuse.

### The `-ub`/`-b` batch-size sweep

A full production-scale (~17.3K token prompt) round-trip benchmark swept llama.cpp's `-ub`/`-b` (ubatch/batch size):

| Config | Prefill | Generation | Wall-clock |
|---|---|---|---|
| Default (512) | ~75–76 tok/s | ~17.0–17.3 tok/s | ~256–261s |
| `-ub 2048 -b 2048` | ~109 tok/s (+45%) | ~16.5 tok/s (flat) | ~189s (−28%) |
| `-ub 4096 -b 4096` | ~114 tok/s (marginal further gain) | ~15.4 tok/s (declining) | ~183s (barely moved) |

`-ub 4096` showed diminishing, even reversing returns — a small further prefill gain traded against a real, continuing decline in generation speed, for almost no additional wall-clock benefit. `-ub 2048 -b 2048` was chosen as the sweet spot: most of the prefill win, without giving up generation throughput — which mattered explicitly here, since "more tokens per second" was one of two success criteria set alongside total wall-clock time.

---

## 4. Speculative decoding: a hard tokenizer constraint, then a real win from prompt-lookup

### Classic draft-model spec-decode: ruled out, but not on the reason we first assumed

Our first pass concluded that speculative decoding "doesn't help small-active-param MoE models" — based on a single anecdotal source: a HuggingFace discussion benchmarking a *different* model (Qwen3.6-35B-A3B) on an RTX 3090/CUDA, n=1 setup, 5 prompts, one draft model, reporting a 10-39% regression attributed to "expert-saturation during verification." We didn't accept that from one thin source, and re-investigated properly.

The re-investigation found llama.cpp's `--model-draft`/`-md` path has no architecture allowlist — it works generically. But there's a real, hard blocker for North-Mini-Code-1.0 specifically: Cohere's tokenizer family is distinct from the Llama/Qwen lineage, and **no small (<1B) Cohere-tokenizer-compatible draft model exists anywhere**. The only smaller same-family option, Command-R7B, is itself ~7B dense — far too large to serve as an efficient draft (a draft model should be roughly an order of magnitude smaller than the target's *active* params, not its total). So classic draft-model spec-decode is ruled out on a genuine, verified structural constraint — no compatible draft model exists — independent of whatever the MoE-verification economics would have been. The original single-source pessimistic claim was thin, but the conclusion happened to still hold, just for a different and better-verified reason.

### Prompt-lookup/n-gram decoding: the real alternative

The same research pass surfaced llama.cpp's `--spec-type` n-gram/prompt-lookup variants — `ngram-simple`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`, `ngram-cache` — which need no separate draft model at all: pure pattern-matching against the model's own token stream, architecture-agnostic. This looked well-suited to code-review/editing workloads, where large spans of input get echoed back in the output.

First test: `ngram-simple` on a prompt engineered to force high repetition (asking the model to repeat a paragraph back verbatim). Confirmed real engagement:

| Metric | Value |
|---|---|
| `draft_n` | 234 |
| `draft_n_accepted` | 96 |
| Acceptance rate | 41% |
| Mean accepted run length | 20.2 tokens |

That proved the mechanism genuinely works. But a plain "write a fresh code review" prompt showed **zero** engagement — no `draft_n` field appeared in the response at all. The mechanism only helps when there's real repetition/echoing between input and output, not on fresh-prose generation.

On a more realistic task — "apply an edit, reproduce the full file with additions," representative of actual coding-assistant usage rather than prose essays — `ngram-simple` showed a real, large win:

| Metric | Value |
|---|---|
| `draft_n` | 528 |
| `draft_n_accepted` | 386 |
| Acceptance rate | 73% |
| Generation speed | 35.2 tok/s (vs. ~16-17 tok/s non-spec-decode baseline) |

Output was verified clean and correct — the original code was faithfully reproduced with real docstring additions, no corruption.

### The size-m sweep for ngram-map-k4v (numbers that didn't survive a proper re-test — see section 5)

A 4-way comparison of all `--spec-type` ngram variants on a full ~17.3K-token production-scale edit task found `ngram-map-k4v` an apparent clear winner at default settings:

| Variant | Generation speed |
|---|---|
| `ngram-simple` | 19.58 tok/s |
| `ngram-mod` | 16.41 tok/s (no improvement over non-spec baseline) |
| `ngram-map-k` | 20.04 tok/s |
| `ngram-map-k4v` | 23.24 tok/s (`draft_n`=3089, accepted=2164, 70% acceptance) |

Sweeping `ngram-map-k4v`'s `--spec-ngram-map-k4v-size-m` (max draft length) found real further gains — until it didn't:

| `size-m` | Generation speed | `draft_n` / accepted | Acceptance | Mean accepted length |
|---|---|---|---|---|
| 48 (default) | 23.24 tok/s | 3089 / 2164 | 70% | — |
| 128 | **24.33 tok/s (best)** | 4006 / 2752 | 68.9% | 87.00 |
| 192 | 10.57 tok/s (collapse) | 8239 / 952 | 11.6% | 22.64 |
| 256 | 20.66 tok/s | — | — | — |

`size-m=192`'s collapse was a real, non-monotonic instability, not smooth diminishing returns — draft attempts ballooned 2x while acceptance cratered. `size-m=128` was re-confirmed via an exact repeat test (24.35 tok/s, nearly identical stats to the first run) and treated as the validated winner — at the time.

**This entire ranking was later found to be significantly contaminated by CPU thermal throttling and had to be redone under controlled conditions** (see the next section for why this ranking didn't survive a proper re-test).

---

## 5. The Ghost in the Machine: A CPU Thermal-Throttling Investigation

### The trigger: a clean reboot that wasn't clean

By this point in the day, `-c 65536 -ub 2048 -b 2048 --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-m 128 -ctk f16 -ctv q8_0` had scored 24.35 tok/s on our real ~17K-token edit-style benchmark — twice, independently reproduced. To lock that in with confidence, we rebooted the machine and ran the exact same config from a cold start, expecting either the same number or something even better.

Instead, three consecutive fresh-boot attempts all landed badly degraded:

| Run | Result | Notes |
|---|---|---|
| 1 | Correctness failure | Hallucinated a fake JSON function-schema instead of performing the requested edit; `finish_reason: stop` after only 512 tokens. Never reproduced again — flagged as sampling-variance noise (`temperature=0.2` is non-zero) rather than the thermal story below. |
| 2 (retry) | 15.28 tok/s | Correct content this time, but `draft_n_accepted=0` — zero drafts accepted. |
| 3 (same server, third request — testing for a "warm-up" effect) | 15.53 tok/s | `draft_n` field absent entirely. No improvement across three consecutive attempts. |

All three sat in a tight 13.7–15.5 tok/s band, nowhere near the pre-reboot best of 24.35 tok/s. A *fresh* boot performing *worse* than a session that had already been running for hours was the opposite of what we expected, and it demanded an explanation rather than a shrug.

### Ruling things out, one hypothesis at a time

**Disk/swap?** The most direct question to ask was whether the model's memory was quietly getting paged out. Checked directly:

```
/proc/<llama-server-pid>/status
VmSwap:        0 kB
```

Zero. The process itself had nothing swapped. (Its `VmRSS`/`VmHWM` did read strangely low — around 166MB for a ~19GB model — but that's a known quirk of GPU/Vulkan workloads: compute-buffer memory gets accounted through the DRM/GPU driver subsystem, not ordinary process RSS. Not evidence of anything wrong, just an odd number to notice and move past.)

**GPU clock throttling?** Sampled live, during an actual test run, straight from the Intel Xe kernel driver's sysfs interface:

```
/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0/cur_freq
/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0/act_freq
/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0/throttle/reasons
```

The GPU sat pinned at its maximum, 1950 MHz, for the entire test, and `throttle/reasons` read `none` throughout. Cleanly ruled out.

**CPU thermal throttling?** This is where it stopped being a dead end:

```
/sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count
/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count
```

The *package* counter already read 6 — on a boot that was only 20–25 minutes old. This is a laptop, not a rack server; it doesn't take long. Sampling live per-core frequencies from `/proc/cpuinfo` during a test showed genuinely wild swings: some cores hitting 4196 MHz (near max boost) one moment, then crashing to 400–900 MHz moments later. That's a textbook thermal/power-budget throttling signature, not noise.

Why does this matter specifically for North-Mini's speculative decoding? Because the n-gram draft/lookup step — hashing recent token history to propose a continuation — runs on the **CPU**. Only the verification step runs on the GPU. A CPU that's being throttled degrades exactly the mechanism (`draft_n`, acceptance rate) we'd watched fall apart, while GPU-bound raw compute stayed nominally fine the whole time. The pieces fit.

### The confirming experiment

Hypotheses are cheap; we wanted a smoking gun. We let the machine idle for a real ~3 minutes with nothing running, and confirmed the state actually changed:

- Package temperature: 57°C → 37°C
- GPU: fully idle (`act_freq: 0`)

Then re-ran the *exact same* benchmark immediately after. Performance snapped back:

| | Post-reboot (hot) | Post-cooldown (same config, same prompt) |
|---|---|---|
| Generation speed | 15.28–15.53 tok/s | **23.05 tok/s** |
| `draft_n` / accepted | 0 or absent | 3122 / 2690 (86.2% acceptance) |

Same binary, same flags, same prompt — the only thing that had changed was thermal state, and the result flipped from "badly degraded" to matching (arguably exceeding) the best number we'd ever measured. That's about as clean a confirmation as this kind of investigation gets.

### The consequence: an earlier conclusion didn't survive

This forced an uncomfortable but necessary re-examination. Every `--spec-type` and `size-m` comparison in Section 4 had been run back-to-back with zero cooldown between configs — meaning "`ngram-map-k4v` is clearly the best variant, and `size-m=128` is clearly the sweet spot" could have been measuring thermal state as much as real algorithmic difference.

We redid it properly: all 4 `--spec-type` ngram variants, on the same fast 3-prompt suite (a real ~150-line code review, a bin-packing scheduling puzzle with a verifiable optimal answer of makespan=14, and a subtly-buggy merge-sort bug-find), this time checking `x86_pkg_temp` via sysfs and waiting for it to drop below ~50°C before swapping configs.

| Spec-type | Code review | Scheduling | Bug-find |
|---|---|---|---|
| ngram-simple | 21.97 | 26.09 | 24.51 |
| ngram-map-k | 22.37 | 27.11 | 26.57 |
| ngram-mod | 22.45 | 28.00 | 27.70 |
| ngram-map-k4v | 22.37 | 27.30 | 25.73 |

Under fair thermal conditions, all four land within about a tok/s or two of each other — `ngram-mod`, previously dismissed as the weakest, wasn't weak at all under contaminated back-to-back testing; it was just unlucky in the order it happened to run. There is no clear winner here. All correctness checks (the makespan=14 answer, the correct merge-sort fix) passed cleanly across every configuration. The honest correction: our earlier "`ngram-map-k4v` wins by a wide margin" narrative was, at least in significant part, a thermal artifact — not a real property of the algorithm.

---

## 6. Choosing on principle: ngram-mod, KV cache, and the context window we actually came for

### Picking `ngram-mod` for reasons that have nothing to do with speed

Once the thermal confound was controlled for (Section 5), all four `--spec-type` variants scored within a percent or two of each other. That raised an obvious question: if speed doesn't distinguish them, is there any real reason to prefer one — not a benchmark reason, an *understanding* reason?

Digging into `llama.cpp`'s own `docs/speculative.md`, PR #19164, and the actual source (`common/ngram-mod.{h,cpp}`, `common/speculative.cpp`) turned up a real structural difference the benchmarks alone never would have surfaced:

- **`ngram-map-k4v`** is a hash *map* keyed by n-gram (the "k" is the key length), where each key stores up to 4 candidate continuation values (the "v"). The footprint of this structure **grows** with the number of distinct n-grams seen — across a long context or a long generation, that map keeps getting bigger.
- **`ngram-mod`** ("mod" = modulo hashing) instead hashes the last N tokens straight into a fixed-size bucket table. Confirmed directly from source: `common_ngram_mod` is a flat array hardcoded to `4*1024*1024` entries (~16MB), not derived from any of its own tuning parameters. Each slot holds exactly one `int32` token id, last-write-wins, no collision chaining. Memory is constant, full stop, regardless of context length.

PR #19164's own discussion recommends `ngram-mod` explicitly for "code iteration/editing" and repetitive workloads — which is exactly what this server exists for — and calls it out as the most memory-efficient of the ngram variants.

Given the speed tie already established, and `ngram-mod` being both the documented right fit for this workload *and* meaningfully cheaper on memory — a goal that's been the point of this entire search — the final config switched from `ngram-map-k4v` to `ngram-mod` on these grounds, not a speed win.

#### A quick parameter sweep, mostly to rule things out

Two follow-up experiments, both against the same 3-prompt fast test suite (a scheduling puzzle with a verifiable optimal makespan of 14, and a subtly-buggy merge-sort bug-finding task, both graded for correctness on every run):

| Parameter | Default | Tested | Result |
|---|---|---|---|
| `--spec-ngram-mod-n-max` | 64 | 128 | No measurable effect (within 1-2% of default) |
| `--spec-ngram-mod-n-min` | 48 | 16 | No measurable effect (within 1-2% of default) |

`n-max=128` was the real analog to `ngram-map-k4v`'s tuned `size-m` — and unlike that experiment, this one carried none of the memory-growth risk, since `ngram-mod`'s per-bucket cost is constant no matter what `n-max` is set to. It just didn't move the needle either way. Settled on plain defaults for both — no extra tuning flags needed.

One honest caveat worth recording: `llama.cpp`'s own docs suggest MoE models may generically benefit from *longer* drafts, but a separate community benchmark found `ngram-mod` causing a real 3-5% regression on a different MoE model (Qwen3.6-35B-A3B), attributed to "expert-saturation during verification." No guidance exists anywhere specifically for North-Mini-Code-1.0 — this project's own empirical testing was the only real source of truth for this exact model.

### KV cache quantization: a real four-way fight

Tested on the same fast 3-prompt suite (code review ~2.4K tokens / scheduling / bug-finding), correctness-checked on every run:

| Config | Code review | Scheduling | Bug-finding |
|---|---|---|---|
| `f16` / `f16` (baseline) | 19.94 tok/s | 27.70 tok/s | 27.40 tok/s |
| `q8_0` / `q8_0` (symmetric) | 19.99 tok/s | 26.93 tok/s | 25.73 tok/s |
| `q8_0` / `q4_0` (aggressive asymmetric) | **15.40 tok/s** | — | — |
| `f16` / `q8_0` (light asymmetric) | **22.33 tok/s** | 27.30 tok/s | 25.73 tok/s |

Symmetric `q8_0`/`q8_0` came in essentially free — within normal noise of the unquantized baseline, a real memory win with no measured cost. The aggressive `q8_0`/`q4_0` combination was a genuine regression on the code-review prompt specifically: `draft_n` ballooned to 2988 attempts while acceptance dropped, meaning the more aggressive V-cache quantization's numerical imprecision was interacting badly with the n-gram speculative decoder's own draft-verification step — more candidate matches found, but verified less reliably against the model's own, correspondingly noisier output.

The actual winner was the lightest asymmetric option — `f16` for K, `q8_0` for V — beating every other config outright (22.33 tok/s, best of all four) with `draft_n=130`/`accepted=11`: far fewer wasted draft attempts than any other config, meaning it found real matches efficiently rather than thrashing. This tracks: K precision matters more for attention/similarity scoring, and by extension for the speculative decoder's own verification precision, so keeping K full-precision while only quantizing V sidestepped the `q8_0`/`q4_0` regression entirely while still banking real V-cache memory savings. Final choice: `-ctk f16 -ctv q8_0`.

### Context window: the reason we started this whole search

The entire point of this migration was context headroom, so this is the number that mattered most. Tested progressively larger `-c` values against the final config, checking real system RAM via `free -h` before and after each:

| `-c` value | RAM used | RAM available | Notes |
|---|---|---|---|
| 24576 (old Qwen3.6 ceiling) | ~23GB | ~7-7.5GB | Baseline |
| **65536 (2.67x)** | ~24GB | ~6-6.6GB | Barely any extra cost; 22.28 tok/s, matching the 24576 baseline exactly |
| 98304 (1.5x more) | ~29GB | ~1.2GB | Steep, disproportionate jump |

`65536` cost roughly 1GB more RAM for a 2.67x larger context window — a genuinely cheap trade. Pushing further to `98304` broke that pattern badly: only ~1.2GB left system-wide, against ~30GB total. That's not a technical wall so much as a real, explicit one — this machine is a daily-driver laptop that also runs Firefox, Chrome, and Podman containers day to day, not a dedicated benchmark box, and leaving almost nothing for normal desktop use wasn't worth the extra headroom. Final choice: `-c 65536` — a real 2.67x increase over the old production ceiling, with comfortable RAM (6+GB) still free for everything else this machine does.

---

## 7. Locking it in

### The final config

```
llama-server -m /home/schneider/models/gguf/North-Mini-Code-1.0-UD-Q4_K_M.gguf \
  -ngl 99 -c 65536 -ub 2048 -b 2048 --parallel 1 \
  --reasoning-budget 400 --spec-type ngram-mod -ctk f16 -ctv q8_0 \
  --temp 0.2 --port 8901
```

| Flag | Why |
|---|---|
| `-ngl 99` | Full GPU residency. |
| `-c 65536` | 2.67x the old Qwen3.6 ceiling (24576), at negligible extra RAM cost (Section 6). |
| `-ub 2048 -b 2048` | The batch-size sweep winner (Section 3) — most of the prefill gain without sacrificing generation throughput. |
| `--reasoning-budget 400` | Capped, not disabled, thinking (Section 3) — bounds latency while keeping the correctness benefit on multi-step tasks. |
| `--spec-type ngram-mod` | The final pivot from `ngram-map-k4v` on real architectural and use-case grounds, not raw speed (Section 6). |
| `-ctk f16 -ctv q8_0` | The KV-quantization winner — real memory savings, no measured cost (Section 6). |
| `--temp 0.2` | The validated safe sampling default (Section 3), set server-side so any client that doesn't override temperature gets this instead of llama-server's own untested built-in default of 0.80. |

### Final validation

Full production-scale (~17-18K token) round-trip testing with the complete config above:

| Check | Result |
|---|---|
| Generation speed | ~22-28 tok/s depending on task type — beats the old Qwen3.6 baseline's ~18.7-19.1 tok/s |
| Tool-calling | Clean, correctly formatted |
| Multi-system-message chat template | Handled natively — no patch needed, unlike Qwen3.6 |
| Content quality (short, ~2.4K tokens) | No leaked special tokens, no repetition |
| Content quality (long, ~9-17K tokens) | No leaked special tokens, no repetition/corruption |

The multi-system-message result is worth calling out on its own: Qwen3.6 required us to hand-patch its embedded chat template to survive codecompanion's `<rules>`/`@{agent}` attachment shape. North-Mini's template handles the same shape correctly out of the box — one less moving part in production.

### The production cutover

Two files changed in `nixos-dotfiles`:

**`config/gpu-server/default.nix`** — `modelPath` repointed to North-Mini's GGUF, `description` updated, `chatTemplateFile` removed entirely (a real simplification — nothing to patch this time), `ExecStart` replaced with the config above, and extensive inline comments added documenting the full rationale, matching this file's existing convention of WHY-focused comments on every tuning decision.

**`config/nvf/nvim-config.nix`** — model display name updated (`qwen3.6-35b-a3b-gpu` → `north-mini-code-1.0-gpu`), and `context_window` metadata bumped from 24576 to 65536. That second change is functionally important, not cosmetic: it's what drives codecompanion's token-count-percentage display and when it decides to start compacting a conversation.

This cutover also surfaced a real latent bug worth fixing while we were in there: the existing `enable_thinking` schema field — a per-chat toggle mapped to `chat_template_kwargs.enable_thinking` — was written for Qwen3.6/GLM's chat templates, which check that exact field. North-Mini's template checks `reasoning_effort=="none"` instead and never reads `enable_thinking` at all. Left in place, the toggle would have silently done nothing for the new model. We removed it; thinking is now left on by default and bounded server-side via `--reasoning-budget 400`, matching this project's own finding that capped thinking measurably helps this model's correctness.

Both files were validated with `nix-instantiate --parse` and `nix flake check --no-build` before applying, then brought live with `sudo nixos-rebuild switch --flake .#laptop-schneider` — confirmed via `systemctl --user status gpu-server-hard` showing the new description.

### Cleanup

With North-Mini confirmed live, every now-unused model GGUF came off disk:

| Model | Size freed |
|---|---|
| Qwen3.6-35B-A3B | 22GB |
| Devstral Small 2 24B | 14GB |
| GLM-4.7-Flash | 18GB |
| **Total** | **~54GB** (77GB → 128GB available) |

GLM-4.7-Flash had initially been kept around "in case a future Vulkan update fixes it" — that call was explicitly reversed once North-Mini proved itself in production.

## Q&A: clarifying questions asked during this investigation

**Why does the n-gram speculative decoding lookup run on the CPU instead of the GPU?**
Because it's fundamentally a hash-table lookup — has this exact sequence of recent tokens been seen before, and what followed it? That's sequential, branch-heavy, pointer-chasing work, which is a poor fit for GPU-style massively-parallel arithmetic. The actual verification step — a real forward pass through the model — is where GPUs earn their keep, and that correctly stays on GPU. Cheap symbolic lookup on CPU, expensive numeric verification on GPU: that's the standard, correct split, not something left on the wrong device.

**What does `ngram-map-k4v` actually mean?**
"k" is the n-gram key length used to look up matches in a hash map. "v" is that each key stores up to 4 candidate continuation values (with occurrence counts) instead of just the single most-recent one — useful when a given prefix has historically led to several different continuations and you want the statistically dominant one, not just whatever came last.

**Is the ~22-28 tok/s number reliable in real usage, or was some of it thermal luck?**
Honestly: both things are true. The peak numbers (24+ tok/s) were measured under good thermal conditions and reproduced multiple times under fresh/cooled conditions — that's genuinely achievable, not a fluke. But real, continuous back-to-back usage on this specific laptop can and did measurably degrade performance into the 15-17 tok/s range once the CPU heats up under sustained load. Actual production traffic — spaced-out human requests with natural typing/thinking gaps — is expected to behave much more like the cool case than the synthetic back-to-back benchmarking that triggered the degraded numbers. That's a real, open thing worth watching in production, not something to declare fully solved.

**Why couldn't we just use a classic draft model for speculative decoding instead of n-gram lookup?**
Cohere's tokenizer family is distinct from Llama/Qwen, and there's no small (<1B) Cohere-tokenizer-compatible draft model available anywhere. The only smaller same-family option, Command-R7B, is itself ~7B dense — too large to serve as an efficient draft. That's a hard structural blocker, independent of whether the underlying verification economics would have worked out.

**Why did Devstral's chat-template bug need extracting the GGUF's actual embedded template, instead of just reading the model's published template on HuggingFace?**
Because they'd already diverged. The template published on Devstral's HF page didn't reproduce the crash we were seeing — it had moved on since the GGUF was quantized. The only reliable source of truth for what a given GGUF will actually do at inference time is the template baked into that specific file, which is why we wrote a small pure-Python GGUF metadata parser to pull it out directly rather than trusting the model card.
