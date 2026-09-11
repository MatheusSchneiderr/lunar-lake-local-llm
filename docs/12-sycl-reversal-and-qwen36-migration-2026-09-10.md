# The SYCL Reversal: North-Mini Fails, Qwen3.6-35B-A3B Takes Over

Chapter 11 ended with `Cohere North-Mini-Code-1.0` winning a nine-candidate
search and going to production with 2.67x the prior context ceiling. It
lasted one day. This chapter starts with that model failing in real use
despite having passed every benchmark thrown at it, and ends with a
complete reversal on two fronts at once: the model changes again, and so
does the backend — to SYCL, the exact backend [chapter
10](10-fine-tuning-update-2026-09-08.md) had, two days earlier, "conclusively
ruled out." Both reversals are argued on their merits below, including an
honest reckoning with why the earlier SYCL verdict doesn't automatically
apply to what we actually shipped.

Everything below happened on the same pinned stack as chapter 11 unless
noted otherwise; new pins introduced in this chapter (the SYCL toolchain,
the new model/quant) are appended to
[configs/shared/versions.md](../configs/shared/versions.md).

---

## 1. The trigger: a model that passed every test and still failed

North-Mini-Code-1.0 cleared chapter 11's full validation pass — the
nine-candidate bake-off, the CPU-thermal-throttling-controlled benchmarks,
the `ngram-mod` speculative-decoding tuning — and went to production. It
then failed on a real task the very next day. The specific failure mode
wasn't captured in detail at the time (worth flagging as a process gap:
when something like this happens, grab the transcript before moving on),
but the signal itself was unambiguous and is worth stating plainly: a
model can pass a rigorous, honest benchmark suite and still not be
trustworthy in daily use. Benchmarks are a filter, not a guarantee.

This reopened the search — but this time with a wider net (five engines,
not one) and, per explicit instruction, a rule that no engine gets
pre-eliminated on the strength of an old verdict without being re-tried
against the *current* candidate models.

## 2. Reconciling chapter 10's SYCL verdict, honestly

Chapter 10 didn't hand-wave SYCL away — it cited four specific, dated
GitHub issues and called the backend "actively worse than Vulkan, not just
unproven." Revisiting SYCL now means addressing every one of those
citations on its own terms, not quietly stepping around them. Here's where
each one actually lands against the config we ended up shipping:

- **Issue #23203** (speculative-decoding memory/performance regression on
  SYCL): moot. This project's own speculative-decoding research for
  Qwen3.6-35B-A3B (see [section 4](#4-quant-ladder-and-model-selection-with-qwen36-back-on-the-table)
  below) found MTP and DFlash both a net *loss* on this model on this
  hardware — the shipped config uses no speculative decoding of any kind,
  so this regression has nothing to attach to.
- **Issues #19276 and #27046** (quantized-KV-cache segfaults, and a
  Lunar-Lake-specific SIGSEGV that reproduces specifically with
  `-fa on -ctk q8_0 -ctv q8_0`): moot. [Chapter 13](13-qwen36-sycl-fine-tuning-2026-09-11.md)
  covers testing KV-cache quantization on this exact SYCL build and
  rejecting it (slower, no benefit) — the shipped config runs
  unquantized f16/f16 KV cache. #27046's trigger condition is a
  quantized KV pair we never use.
- **Issue #28193** (Flash Attention corruption, reproduced via
  `test-backend-ops -o FLASH_ATTN_EXT` on the permuted Q8_0/Q8_0 KV case):
  this is the one that doesn't fully dissolve, and it would be dishonest
  to claim otherwise. The shipped config runs FA via the default `auto`
  setting, which chapter 13 confirms is effectively mandatory at our
  context size (`-fa off` crashes and near-OOMs — see chapter 13, section
  5). That is FA, on Xe2, on the same backend the bug report covers. The
  saving distinction is that the reported corruption is specifically on a
  **Q8_0/Q8_0 KV** case, and our KV cache is **unquantized f16/f16** — a
  different tensor-dtype path through the same kernel family. We have
  **not** run llama.cpp's own `test-backend-ops` FLASH_ATTN_EXT suite
  against our exact build to directly confirm or refute this. What we do
  have is indirect: dozens of real agentic coding tasks across this
  entire search and the fine-tuning pass that follows, every one producing
  structurally correct, building C# code with no observed output
  corruption. That's reassuring, not proof. Anyone deploying this same
  stack should treat FA-on-SYCL-with-f16-KV as *probably fine, not
  independently verified* — a real open item, not a resolved one.
- **"Not packaged in nixpkgs, would need a from-scratch overlay"**:
  correct, and it was exactly as much work as that sentence implies (see
  [section 6](#6-building-llamacpp-sycl-from-scratch-the-packaging-pain-was-real)
  below). We did it anyway because the payoff — a working MoE backend at
  4x the context ceiling North-Mini/Vulkan gave us — was worth the build
  pain.

**Reconciled verdict:** chapter 10 was reconfirmed correct *for the flag
combination it evaluated* — Flash Attention paired with speculative
decoding and/or quantized KV cache, on SYCL, on this hardware, is a real
minefield with real open GitHub issues. The config that actually shipped
sidesteps two of those three ingredients entirely by not needing them.
What's left (FA alone, unquantized KV) is a narrower, less-tested surface
that has not misbehaved in extensive real use, but has also not been
independently verified against the specific upstream bug report that
prompted the original caution. This is not "chapter 10 was wrong" — it's
"the question changed once the answer to a different question (which
model/config to run) changed with it."

## 3. The search: five engines, four MoE candidates, no pre-eliminated survivors

The full plan (five engines — OpenVINO GenAI GPU, llama.cpp SYCL, vLLM-XPU,
MLC-LLM, IPEX-LLM — against four MoE candidates — LFM2.5-8B-A1B,
GPT-OSS-20B, Gemma-4-26B-A4B, Qwen1.5-MoE-A2.7B) ran with the same safety
protocol as every prior round: `systemctl --user stop` both production
services before touching anything, `free -h` before and after every heavy
load, never two heavy engines running concurrently.

### OpenVINO GenAI, GPU device: working, after a real packaging fix

The GPU-plugin MoE kernel (`moe_3gemm_swiglu_opt`) was silently disabled in
nixpkgs' `openvino` build (`-DENABLE_ONEDNN_FOR_GPU=false`) — every MoE
model crashed on `generate()` with "depends on onednn" →
`CL_OUT_OF_RESOURCES`. This is a build-flag choice, not a hardware limit.
Fix: `overlays/openvino-onednn-gpu.nix` flips the flag; `openvino-genai`
cascades the rebuild automatically via `callPackage`. Full rebuild: ~44
minutes.

**Tested and passing**: LFM2.5-8B-A1B, GPT-OSS-20B (16K-token escalating
context, no corruption). Gemma-4-26B-A4B-it also works but is exported as
a VLM (needs `VLMPipeline`, not `LLMPipeline` — a cryptic "Port for tensor
name input_ids was not found" error is what you get if you use the wrong
one). Qwen1.5-MoE-A2.7B was skipped on this engine — no pre-quantized
OpenVINO IR exists for it anywhere on Hugging Face, and exporting from its
GPTQ-Int4 checkpoint hit real dependency conflicts (`auto-gptq` fails to
build on Python 3.14; `gptqmodel` drags in `torchvision` and bumps
`transformers` past what `optimum-intel` in the same venv tolerates).

### llama.cpp SYCL: working, after a genuinely hard packaging fight

No `llama-cpp-sycl` attribute exists in nixpkgs — only `-vulkan`/`-cuda`/
`-rocm`. Building it needed Intel's proprietary DPC++ compiler (`icx`/
`icpx` from `intel-oneapi-toolkit`); the open `adaptivecpp`/
`generic-sycl-components` packages do not satisfy llama.cpp's SYCL cmake
detection. Full detail on the fight itself is in
[section 6](#6-building-llamacpp-sycl-from-scratch-the-packaging-pain-was-real).

**Tested and passing, all four candidates**: LFM2.5-8B-A1B (~600-640 tok/s
prefill, ~55 tok/s gen — genuinely GPU-bound, confirmed via CPU usage
staying at 40-48% during generation rather than pegged near 100%),
GPT-OSS-20B (~29 pp/~11 gen), Gemma-4-26B-A4B-it (~12 pp/~8.4 gen),
Qwen1.5-MoE-A2.7B (~21 pp/~only 3.7 gen — notably slow, plausibly an older,
less-optimized architecture hitting a weaker SYCL kernel path). llama.cpp
SYCL was the only engine with full 4/4 candidate coverage.

### vLLM-XPU: the Level Zero driver problem got solved, but the reason we wanted it evaporated

`torch.xpu.is_available()` returning `True` on this Lunar Lake iGPU took
four separate, non-obvious fixes stacked together: nixpkgs'
`intel-compute-runtime` already builds the Level Zero backend but stashes
it in a second output (`.drivers`) that isn't wired in by default; the
Level Zero *loader* (`level-zero` package) needs its own
`LD_LIBRARY_PATH` entry separate from the driver; `intel-gmmlib`'s
`libigdgmm.so.12` isn't auto-pulled inside a `steam-run` sandbox; and even
with all three libraries present, device init aborted on an
`UNRECOVERABLE_IF(preemptionSurfaceSize == 0)` check that turned out to be
a hardware-info-population bug specific to this compute-runtime version's
Level Zero path on this device (not a missing-support issue — OpenCL on
the identical runtime works fine) — worked around with
`NEOReadDebugKeys=1 ForcePreemptionMode=1 OverridePreemptionSurfaceSizeInMb=1`.

The actual motivation for vLLM-XPU was a specific checkpoint,
`ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ` — a real, research-lab-grade 2-bit
quantization from a serious quantization group, at ~12.5GB. It turned out
to be a dead end unrelated to any of the driver work: its "Humming" 2-bit
unpacking kernels are CUDA-only by the kernel repo's own description
("supports all NVIDIA GPUs from SM75+... and beyond") — no XPU/SYCL
backend exists, and none of the Level Zero fixes above change that. The
driver setup itself remains a working, reusable capability; this specific
model just can't use it.

### MLC-LLM and IPEX-LLM: not pursued

MLC-LLM's only Intel-iGPU-capable backend on Linux is **Vulkan** — the
exact backend this whole investigation exists to get away from
(`ggml-org/llama.cpp#28590`, the coopmat crash from earlier chapters).
Testing it would mean routing through the same backend class with real
risk of hitting an equivalent driver-level crash in a different codebase,
for uncertain benefit — not pursued further given the poor risk/reward.
IPEX-LLM was archived by Intel in January 2026 (per chapter 10's own
finding) and excluded on that basis alone.

### Candidates ruled out

- **LFM2.5-8B-A1B**: fast (~58 tok/s) and genuinely engages with a
  sufficiently detailed prompt — this is not a speed or laziness problem.
  The disqualifier is reliability under correction: across a 4-turn
  detailed-prompt-then-correct test, it ignored a literal compiler-error
  correction outright, then on the next turn made a no-op edit and
  **confabulated a false success claim** ("the code already uses
  GetString correctly," "the project builds cleanly") directly
  contradicted by the actual unchanged file and real build errors. A
  model that asserts false confidence about a failed fix breaks the trust
  a fast-iteration workflow depends on — worse than being slow, worse
  even than being wrong.
- **Qwen1.5-MoE-A2.7B-Chat**: reproduced twice, never touched a single
  real project file — hallucinated a generic ASP.NET Core MVC
  `NotesController.cs` for a project that is actually a raw
  `HttpListener` console app, answering like a plain chat model rather
  than an agent taking real actions. Root cause is almost certainly age:
  this checkpoint is from Alibaba's Qwen1.5 generation (~February 2024),
  over two years older than every other 2026-era candidate in this round,
  predating the tool-calling/agentic-coding training that's now standard.
  Not a knock against the modern Qwen line.
- **`ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ`**: CUDA-only kernels, covered
  above.

**Lesson (infra, not model):** the SYCL benchmark harness originally
tracked spawned `llama-server` PIDs via a captured `$!` shell variable and
killed them with `kill $SRV_PID` between combinations. It silently missed
once, leaking a 12GB GPT-OSS-20B process at 98% CPU for ~50 minutes —
nearly causing a second OOM when the next (17GB) model tried to load, and
contaminating that round's speed numbers under undetected contention
(measured ~4.5 tok/s vs. a clean ~11 tok/s once found and killed). Fixed
by killing on a pattern match against the model's filename instead of a
captured PID — never trust a `$!` to reliably reference a long-running
backgrounded process across an orchestration script; verify by
pattern/port and checkpoint `free -h` between every heavy model swap.

## 4. Quant ladder and model selection, with Qwen3.6 back on the table

With LFM2.5, GPT-OSS-20B, and Gemma-4-26B-A4B as the survivors after the
engine round, Qwen3.6-35B-A3B was reconsidered too — this time
specifically on llama.cpp SYCL rather than the Vulkan backend chapter 10
tested it on.

### The Qwen3.6 GGUF quant ladder (unsloth/Qwen3.6-35B-A3B-GGUF)

Three quants tested at `-c 65536`, same detailed prompt, one at a time
(disk only had ~26GB free):

| Quant | Size | Wall time | Gen speed | Prefill speed |
|---|---|---|---|---|
| IQ1_M | 10.05GB | 254s | ~17.6-18.6 tok/s | ~130-155 tok/s |
| IQ2_XXS | 10.76GB | 251s | ~17.9-18.0 tok/s | ~130-155 tok/s |
| IQ2_M | 11.52GB | 181s | ~17.2-17.3 tok/s | ~130-155 tok/s |

**Generation speed is essentially quant-independent** — expected for a
MoE model, since only ~3B active parameters actually run per token
regardless of how the routed-expert weights are stored on disk. IQ2_M's
181s looked like a standout; a clean reproducibility rerun gave it **246s**
instead, in line with its siblings. Lesson repeated from chapter 10's own
MXFP4 detour: a single `opencode --auto` wall-clock run is noisy (agent
path variance dominates) and shouldn't be trusted as a per-quant
tie-breaker without reproducing it. All three quants produced working,
building implementations with only minor, correctable edge-case quirks —
no confabulation at any quant level, even the most aggressive IQ1_M.
**Final call: IQ1_M** — smallest file, no demonstrated quality or speed
cost relative to the larger quants once reproduced.

### Gemma-4-26B-A4B's own quant question

A similar story played out on the Gemma-4 side: IQ3_XXS initially looked
like a real win over IQ2_XXS (719s vs. a 900s timeout), but a
reproducibility rerun put IQ3_XXS at **893s** — statistically identical to
IQ2_XXS, not a real advantage. **Final call: IQ2_XXS** (9.92GB) — smaller
file, same effective speed.

### The clean, PATH-bug-fixed final head-to-head

Every earlier timing comparison in this search had an unnoticed
confound: `dotnet` wasn't reliably on `PATH` for both models from the
start, meaning some runs paid an extra discovery cost the others didn't.
Wrapping `opencode` with `direnv exec .` removed it. With the environment
finally leveled:

| Model | Wall time | Decode speed | Build result |
|---|---|---|---|
| Gemma-4-26B-A4B IQ2_XXS | 838s | ~14.68 tok/s | BUILD_OK |
| **Qwen3.6-35B-A3B IQ1_M** | **352s** | **~17.36 tok/s** | **BUILD_OK** |

Qwen3.6 is **~2.4x faster end-to-end**, and this isn't a PATH artifact —
it's real, on a level playing field. A server-side detail from the Gemma
run explains part of the gap: it generated one single, uninterrupted
3647-token turn (steady ~15 tok/s, no tool-call breaks) before finishing —
Gemma-4 appears structurally more verbose and less tool-call-efficient at
this agentic coding task, not just slower per token.

**Winner: Qwen3.6-35B-A3B IQ1_M on llama.cpp SYCL.** This decision is
final. Gemma-4-26B-A4B is dropped from consideration entirely — it lost
decisively, not marginally.

OpenVINO GenAI GPU was separately checked for this specific model/quant
and dropped: the smallest available pre-converted IR
(`OpenVINO/Qwen3.6-35B-A3B-int4-ov`) is ~19.65GB, nearly **2x** the size of
the chosen GGUF, because OpenVINO's GenAI pipeline has no equivalently
aggressive sub-2-bit quantization option — its practical floor is INT4.
Producing a more aggressive quantization ourselves would mean a risky
local full-precision conversion (see the hard rule below) for an engine
already at a structural size disadvantage.

> **Hard rule, violated twice earlier in this project's history and not
> repeated here**: never download or locally convert a model's full
> fp32/fp16 weights on this laptop, not even as a conversion intermediate.
> Always find a pre-quantized artifact first. Doing otherwise OOM-killed
> this machine twice in one earlier session.

## 5. Production cutover

`gpu-server-hard` now runs Qwen3.6-35B-A3B IQ1_M via a custom
llama.cpp-SYCL build, replacing North-Mini/Vulkan entirely. Every flag was
audited against real test data rather than carried over from the old
config:

- `-ngl 99` — kept, present in every real SYCL test run this round.
- No `--parallel` override — North-Mini/Vulkan's reasoning for setting it
  (default 4 slots splits context four ways) doesn't reproduce on SYCL:
  every real test showed `n_slots=4, n_ctx_slot=<full requested -c>` —
  each slot gets the *full* context, not a quarter.
- No `--temp` at this stage (added back in chapter 13 for a real,
  separately-discovered reason).

Both context and quality headroom improved sharply on this cutover: the
finalist model comfortably handles `-c 196608` (~192K tokens) with 9-10GB
RAM still free on this 30GB machine, well past the 120-150K token target
this whole context-ceiling chase has been aiming at since chapter 10.

Batch sizing, sampling presets, the runaway-thinking investigation, Flash
Attention/KV-cache tuning, and three more fine-tuning checks are their own
chapter: [13-qwen36-sycl-fine-tuning-2026-09-11.md](13-qwen36-sycl-fine-tuning-2026-09-11.md).

## 6. Building llama.cpp-SYCL from scratch: the packaging pain was real

Chapter 10's "not packaged, would need a from-scratch overlay" objection
was accurate. The new `overlays/llama-cpp-sycl.nix` needed several
non-obvious fixes because `icx`/`icpx` are prebuilt binaries that assume
an FHS layout and bypass nixpkgs' cc-wrapper entirely — none of the usual
automatic include/rpath injection happens:

1. `--gcc-toolchain=<stdenv.cc.cc>` + `--sysroot=<glibc>` — without these,
   linking fails looking for `Scrt1.o`/`crti.o`/`crtbeginS.o`.
2. `-L<stdenv.cc.cc.lib>/lib` — `libgcc_s.so` lives in a separate `.lib`
   output, not next to the compiler.
3. `-idirafter <glibc.dev>/include`, specifically **not** `-isystem` —
   libstdc++'s own `<cstdlib>` does `#include_next <stdlib.h>`, which
   continues the search from whatever directory came *after* the one
   `<cstdlib>` itself was found in. `-isystem` doesn't reliably sort after
   gcc's own C++ header directories in that search order; `-idirafter`
   guarantees last-resort placement, so `#include_next` actually reaches
   glibc's headers.
4. Manual `-I`/`-L` for `level-zero` and the OpenCL headers — not
   auto-injected from `buildInputs` once the cc-wrapper is bypassed.
5. Explicit `-Wl,-rpath` for every runtime library, `openssl` included —
   `llama-server` links against `libssl` for its HTTPS-capable server, and
   the install-check step failing on a missing `libssl.so.3` was the last
   blocker before a clean build.

Two runtime gotchas surfaced only after the build succeeded: the server
needs `LD_LIBRARY_PATH` covering the oneAPI toolkit's own lib directory
plus `level-zero`, and `OCL_ICD_VENDORS` pointed at the compute runtime's
vendor manifest — and setting `ONEAPI_DEVICE_SELECTOR=level_zero:*` is a
trap, not a helpful pin: this Arc iGPU's SYCL device enumerates via the
OpenCL backend, not Level Zero, and forcing Level Zero selection makes
`ggml_sycl_init` fail, causing llama.cpp to **silently fall back to CPU**
with no obvious error. (This exact trap resurfaced during chapter 13's
`GGML_SYCL_F16` testing when a bare `nohup` launch — without the systemd
service's env vars — hit "no usable GPU found" for a related reason;
worth remembering as a recurring failure class, not a one-off.)

## 7. Q&A: clarifying questions asked during this investigation

**Was chapter 10's SYCL rejection simply wrong?** No — see section 2.
It was correct for the flag combination it tested (paired with
speculative decoding and/or KV-cache quantization). The config that
shipped here doesn't use either of those, which is why most of the cited
bugs don't apply — but the Flash-Attention-correctness question is
inherited, not resolved, and is stated as an open caveat above rather
than papered over.

**Why trust a model that failed once (Qwen3.6, on Vulkan, in an earlier
chapter) after just having distrusted another model (North-Mini) for the
same reason?** Different failure class. North-Mini failed in real use
*after* passing a full benchmark suite, with no clear mechanism found.
Qwen3.6's earlier issues were architecture/backend-specific and are
directly addressed by the backend change (SYCL, not Vulkan) plus a
different, smaller quant (IQ1_M) — this is a different underlying
mechanism being changed, not the same coin flipped again.

**Is this the final model/engine decision?** Yes, barring another real
production failure. It is not to be re-litigated on benchmark numbers
alone — see the opening of this chapter for why that bar exists now.

## Status at time of writing

Qwen3.6-35B-A3B IQ1_M on llama.cpp SYCL is deployed to `gpu-server-hard`
and is the user's confirmed daily driver as of this writing. Extensive
fine-tuning on top of this base — batch sizing, sampling presets, a
runaway-thinking root-cause investigation, and several rejected
optimization attempts — is documented in the next chapter.
