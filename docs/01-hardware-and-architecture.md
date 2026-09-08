# Hardware and Architecture

This guide documents a specific, working local-LLM coding-assistant setup on
a single piece of Intel silicon: one NPU tier for fast, small-model "quick
tasks," and one iGPU tier for a large, capable MoE "hard tasks" model,
switched between at the systemd level so exactly one is ever resident at a
time. This chapter covers the hardware itself, why the split exists at all,
and - the load-bearing part - why the two tiers cannot simply be merged into
"run the big model on the NPU" or "run everything on the GPU all the time."

If you take one thing from this chapter: **Mixture-of-Experts (MoE) model
architectures do not run on the Intel NPU, in any OpenVINO release checked
during this project, for an architectural reason that has nothing to do with
misconfiguration.** Everything else in this document follows from that one
constraint plus a ~30GB RAM budget.

## The hardware

All specs below are what was directly confirmed on the machine this guide
was built and tested on. See `configs/shared/versions.md` for the exact
software versions that pair with these numbers - none of this is
"universally true for all Intel NPU/iGPU hardware," it's what was actually
measured here.

| Component | What it is | Notes |
|---|---|---|
| CPU | Intel Core Ultra 200V ("Lunar Lake") | |
| iGPU | Intel Arc Graphics 130V/140V (Xe2) | No dedicated VRAM - shares system RAM with the CPU |
| NPU | Intel AI Boost NPU | Integrated into this CPU generation |
| RAM | ~30GB total system RAM | Shared by CPU, iGPU, and NPU workloads simultaneously |
| Swap | 0B disk swap | A `zramSwap` safety net was added later - see docs/07-benchmarks-and-methodology.md and the "Safety net: swap" discussion there. Not relied on for routine operation. |
| OS | NixOS (unstable channel) | |

The detail that shapes almost every other decision in this guide is the
iGPU's memory model: it is **not** a discrete GPU with its own VRAM pool. It
draws from the same ~30GB system RAM as everything else, via a
driver-managed heap (Mesa's `ANV_SYS_MEM_LIMIT`, default 75% of system RAM -
see the context-window discussion in the Stage findings referenced from
docs/07 for where this actually became the binding constraint). A "large
model on the GPU" on this hardware is really "a large model sharing the
laptop's one RAM pool with the OS, the editor, the browser, and anything
else running," not a separate memory budget you get for free.

## Why two tiers, not one model

The original idea behind this project long precedes any specific model
choice: pair a **small, fast, always-cheap-to-run model** for quick,
low-stakes edits (rename this, fix this one-line bug, explain this
function) with a **larger, slower, more capable model** for genuinely hard
tasks (multi-file refactors, tracing a real bug across a call chain,
planning a nontrivial change) - and let the developer pick per-task rather
than paying the big model's latency for every trivial request, or accepting
the small model's ceiling for everything.

On this hardware, "cheap and fast" and "large and capable" turned out to map
almost directly onto "the NPU" and "the iGPU" - but not for a simple
performance reason. It's an *architectural* mapping, driven by what kind of
model each accelerator can even execute at all:

- **The NPU tier runs a small, dense model**: `Qwen2.5-Coder-7B-Instruct`,
  served via OpenVINO GenAI's `LLMPipeline` in `"NPU"` mode. Dense models
  compile to a fixed, static computation graph - exactly what the NPU's
  compiler (and its static-shape execution model, see below) requires.
- **The GPU tier runs a large Mixture-of-Experts (MoE) model**:
  `Qwen3.6-35B-A3B` (35.95B total parameters, ~3B active per token, 256
  experts with 8+1 routing), served via `llama.cpp`'s Vulkan backend
  (`pkgs.llama-cpp-vulkan`) as a genuine Q4_K_M GGUF quantization. This is
  the "hard tasks" model - larger, more capable, and considerably slower to
  cold-start.

The two are wired as mutually exclusive systemd services (`Conflicts=` on
each other - see "Two mutually exclusive services" below) rather than as
two things you could run side-by-side by choice. That's not a stylistic
preference; it's the direct consequence of the ~30GB RAM budget, confirmed
empirically, not assumed - see the next section and docs/07's concurrency
test for the actual numbers.

## Why MoE cannot run on the NPU (the architectural wall)

This is the single most important finding behind this whole project's
shape, and it's worth explaining *why*, not just asserting it, because it
determines what's even worth trying on NPU hardware going forward.

**Intel NPU inference (via OpenVINO) compiles a model into a static-shape
execution graph ahead of time.** The NPU's whole value proposition -
extremely low power draw for sustained inference - comes from that
ahead-of-time compilation: the compiler knows exactly what tensor shapes,
what operations, and what memory layout will occur at every step, and
schedules the NPU's execution units accordingly. There is no general-purpose
dynamic dispatch at runtime the way a GPU or CPU kernel launcher has.

**MoE architectures are fundamentally dynamic per-token.** In a MoE layer,
a router computes, for every single token, which subset of experts (e.g. 8
of 256 for `Qwen3.6-35B-A3B`) actually get invoked - and that subset changes
token-to-token, prompt-to-prompt, in a data-dependent way that cannot be
known before the model runs. That's the entire point of MoE: only a small
active-parameter subset does compute per token, keeping inference cheap
relative to the model's total parameter count, at the cost of that subset
being decided dynamically at inference time.

These two things are in direct conflict. A static, precompiled execution
graph has no way to represent "route to whichever experts the input
happens to select this time" - there is no fixed set of operations to
compile ahead of time when the operations themselves depend on runtime
data. This is not a missing feature that a future OpenVINO release patches
around the edges; it is a structural mismatch between what static-shape
compilation can express and what MoE inference requires. Research
performed for this project confirmed this has been true across every
OpenVINO release checked (roughly a year's worth, 2025.3.0 through
2026.3.1) - no dense-vs-MoE distinction has ever been bridged for the NPU
target, and Qwen's own "Coder" model line has in fact gone **MoE-only**
since Qwen3 (`Qwen3-Coder-30B-A3B`, `-480B-A35B`, `-Next` - no dense
Qwen3-Coder model exists at all), which forecloses "just wait for a newer
Coder model" as a path back to NPU for the hard-tasks tier.

This is also why the GPU tier isn't running its MoE model through OpenVINO,
despite OpenVINO being the toolkit used for the NPU tier. OpenVINO's
GPU-plugin MoE support does exist (an `OFFLOAD_RATIO` expert-streaming
feature, added in OpenVINO 2026.3.0 specifically for the memory-paging
problem large MoE models hit on unified-memory hardware), and it was
benchmarked seriously - but it turned out to be broken in a different way
on this hardware: real generations under thinking mode never converged
(0/10 complete on a matched 10-prompt test) and the GPU driver crashed
outright on 4/10 prompts. `llama.cpp`'s Vulkan backend, tested against the
exact same GGUF quantization on the exact same hardware, got 9/10 correct
with no crashes. The full comparison, numbers, and the pivot story are in
docs/03-gpu-tier-setup.md and docs/07-benchmarks-and-methodology.md - the
point for this chapter is narrower: the NPU-vs-MoE wall is architectural
and absolute, while the OpenVINO-vs-llama.cpp choice for the GPU tier was
an empirical, delivery-mechanism finding, not a second architectural limit.

One corollary worth stating plainly: **"put the big MoE model on the NPU
instead of the GPU" is not a config change away from working on this
hardware family** - it is not supported, full stop, and no amount of
`MAX_PROMPT_LEN` or memory tuning changes that. If you're evaluating a model
for the NPU tier, checking "is this architecture MoE?" is the very first
filter to apply, before anything else about size or license.

## The two-tier architecture, end to end

```
                      ┌───────────────────────────┐
                      │   Editor / client layer    │
                      │  nvim (codecompanion.nvim) │
                      │        OpenCode CLI        │
                      └─────────────┬───────────────┘
                                    │  OpenAI-compatible
                                    │  /v1/chat/completions
                                    │  (per-task adapter/model choice)
                     ┌──────────────┴──────────────┐
                     │                              │
                     ▼                              ▼
      ┌───────────────────────────┐   ┌────────────────────────────────┐
      │  npu-server-coder.service │   │   gpu-server-hard.service      │
      │  (systemd --user, on-     │   │   (systemd --user, on-demand)  │
      │   demand, port 8900)      │◄─X─►   port 8901                   │
      │                            │Conflicts=│                        │
      │  server.py (bespoke)      │   │  llama-server (llama.cpp)      │
      │  openvino_genai.LLMPipeline│  │  --spec-type draft-mtp         │
      │  device = "NPU"            │  │  -ngl 99  -ub 4096             │
      │  Qwen2.5-Coder-7B-Instruct │  │  -ctk q8_0 -ctv q4_0 -fa on    │
      │  (dense, OpenVINO IR,      │  │  Qwen3.6-35B-A3B, Q4_K_M GGUF  │
      │   int4/NPU-optimized)      │  │  (MoE: 256 experts, 8+1 active)│
      └─────────────┬───────────────┘   └────────────────┬───────────────┘
                     │                                    │
                     ▼                                    ▼
      ┌───────────────────────────┐   ┌────────────────────────────────┐
      │   Intel AI Boost NPU       │   │   Intel Arc Graphics 130V/140V  │
      │   static-shape compiled    │   │   (Xe2, Vulkan `anv` driver)    │
      │   graph, dense-only        │   │   dynamic per-token MoE routing │
      └────────────────────────────┘   └────────────────────────────────┘

           ~30GB system RAM shared by CPU + this iGPU's driver heap
                (0B disk swap; a zram safety net exists, not relied on)
```

The `Conflicts=` relationship in the middle is enforced by systemd itself,
not just documented as a policy: `npu-server-coder.service` and
`gpu-server-hard.service` each declare `Unit.Conflicts` on the other (see
`configs/npu-tier/npu-server/default.nix` and
`configs/gpu-tier/default.nix`), so starting either one via its nvim/shell
keymap automatically stops the other first. Neither service has
`Install.WantedBy` - both start on demand, not at login, to avoid paying
either model's load cost (roughly 30-70s for the NPU pipeline; single-digit
to low-tens-of-seconds for the GPU tier depending on cold vs. warm) until
you actually ask for it.

Why mutual exclusivity, and not "just run both, let the OS figure it out"?
This was tested empirically, not assumed. With the NPU-hosted 7B warm and
the GPU MoE model generating simultaneously, nothing OOM'd and the NPU
server stayed responsive - but `available` memory bottomed out at roughly
5.8-6.1GB and free RAM hit as low as ~268MB, with the kernel squeezing
`buff/cache` from 20GB down to ~10GB just to make room. With 0B disk swap
configured at the time, there was no graceful-degradation path if anything
else needed a memory burst during that window - the next unlucky allocation
would go straight to an OOM kill. Given that thin a margin, mutual
exclusivity at the systemd level was chosen over "concurrent, but risky"
as the supported mode. The full numbers for this test live in
docs/07-benchmarks-and-methodology.md.

An optional third component, `gpu-guard` (a small C++ reverse proxy that
sits in front of `gpu-server-hard` to retry certain degenerate-stall/
false-refusal failure patterns), was built and verified but is **not
currently wired into this architecture** - it's parked, documented, and
easy to re-enable if you hit the failure pattern it targets on a weaker
model. See docs/09-gpu-guard-optional.md; it doesn't appear in the diagram
above because the default, currently-active path is nvim/OpenCode talking
directly to `gpu-server-hard` on port 8901.

## What each tier is actually for

| | NPU tier (`npu-server-coder`) | GPU tier (`gpu-server-hard`) |
|---|---|---|
| Model | Qwen2.5-Coder-7B-Instruct (dense) | Qwen3.6-35B-A3B (MoE, 3B active/35.95B total) |
| Port | 8900 | 8901 |
| Runtime | Bespoke `server.py` on `openvino_genai.LLMPipeline` | Stock `llama-server` (llama.cpp, Vulkan) |
| Intended use | Quick, low-stakes edits and questions | Multi-file reasoning, hard debugging, planning |
| Cold-start cost | ~30-70s (NPU pipeline load) | Lower load time, but a slower steady-state per-token rate than the NPU tier at small-model scale, and thinking mode (when enabled) adds real wall-clock time before an answer starts |
| Concurrency with the other tier | Not supported (`Conflicts=`) | Not supported (`Conflicts=`) |

This table intentionally doesn't repeat tok/s figures, benchmark
methodology, or the model bake-off that led to picking these two specific
models over their alternatives (Qwen3-30B-A3B, DeepSeek-Coder-V2-Lite,
Qwen3-8B, etc.) - that full story, with real numbers and the seeded-vs-
unseeded methodology lesson learned along the way, is in
docs/07-benchmarks-and-methodology.md. The one-line verdict: Qwen2.5-Coder-7B
beat DeepSeek-R1-Distill-Qwen-7B decisively (9/10 vs 4/10 on a 15-prompt
debugging set) for the NPU tier, and Qwen3.6-35B-A3B via llama.cpp/Vulkan
beat every OpenVINO GPU-MoE path tried for the GPU tier.

## Versions used

Every version pin, model checkpoint, and the exact hardware this guide was
built and tested against is tracked in one place:
**`configs/shared/versions.md`**. If anything in this guide doesn't match
what you observe on your own machine, check version drift there first -
`llama.cpp`/OpenVINO internals and Intel's NPU/GPU driver stack all move
fast, and this guide reflects one specific point-in-time snapshot, not a
guarantee for all future releases.

## What to read next

- **docs/02-npu-tier-setup.md** - the NPU tier in detail: the
  `hardware.cpu.intel.npu.enable` module, the two custom Nix overlays this
  project needed to get a working NPU compiler at all
  (`configs/npu-tier/overlays/intel-npu-compiler.nix` and
  `npu-runtime-libs.nix`), the abandoned third overlay, a walkthrough of
  `server.py`, and the systemd unit in
  `configs/npu-tier/npu-server/default.nix`.
- **docs/03-gpu-tier-setup.md** - the GPU tier in detail: the full model
  selection narrative (why OpenVINO's GPU-MoE path and DeepSeek-Coder-V2-Lite
  were both rejected), Vulkan wiring, the chat-template crash and its fix,
  and every `llama-server` flag in
  `configs/gpu-tier/default.nix`'s `ExecStart` explained.
