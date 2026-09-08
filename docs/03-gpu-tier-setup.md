# GPU Tier Setup

The GPU tier is the "hard tasks" half of the two-tier setup described in
`docs/01-hardware-and-architecture.md`: a large Mixture-of-Experts model
running on the Arc iGPU, invoked on demand for work the NPU tier's small
dense model (`docs/02-npu-tier-setup.md`) isn't strong enough for. This
chapter covers how the model and runtime were chosen, how the iGPU is wired
up without any system-wide NixOS graphics module, a chat-template crash that
will bite almost anyone using this model with a tool-calling client, and a
flag-by-flag reading of the `llama-server` invocation that actually runs in
production.

Everything here reflects the exact stack pinned in
`configs/shared/versions.md` — nixpkgs commit, `llama-cpp-vulkan` version,
OpenVINO version, and the specific GGUF quantization. If you're on
different hardware or newer package versions, treat the *reasoning* as
transferable and the *exact numbers* (context ceiling, `-ub` sweep results,
crash thresholds) as needing re-verification on your own machine.

## 1. Model and runtime selection

### The starting constraint: MoE doesn't run on the NPU at all

Before any GPU-specific research happened, an earlier research pass had
already established a categorical blocker: Mixture-of-Experts architectures
have **never** been supported on Intel's NPU in any OpenVINO release. The
NPU's compiler needs static shapes; MoE's dynamic per-token expert routing
is fundamentally incompatible with that. This isn't a tuning problem or a
missing feature — it rules the NPU out for any MoE candidate entirely,
which is why this tier targets the iGPU instead, and why the NPU tier
(`docs/02-npu-tier-setup.md`) stays on a dense model.

Two named candidates were on the table for the GPU tier: `Qwen/Qwen3.6-35B-A3B`
and `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

### DeepSeek-Coder-V2-Lite — rejected before any benchmark ran

DeepSeek-Coder-V2-Lite-Instruct is a real, legitimately-coder-branded MoE
model (64 routed + 2 shared experts, ~2.4B active / 16B total parameters),
but it lost on paper research alone, before it was worth spending benchmark
time on:

- **No pre-built OpenVINO IR exists for it anywhere.** Using it would have
  meant self-converting via `optimum-intel`, an entirely unvalidated path.
- **OpenVINO's GPU-plugin MoE work has only ever been validated against
  GPT-OSS-20B and Qwen3-30B-A3B.** DeepSeek-Coder-V2's architecture
  (`deepseek_v2`, MLA attention) has zero validation history on this stack.
- **"Lite" is misleading for memory purposes**: all 16B parameters stay
  resident regardless of how few are active per token. Its Q4_K_M size
  (~10.4GB) is comparable to the safer Qwen3-30B-A3B fallback below, but
  with none of that model's validated acceleration path.
- **It's dated.** June 2024, benchmarked against GPT-4-Turbo/Claude 3
  Opus-era competitors (MMLU-Pro 41.57), against Qwen3.6-35B-A3B's
  April 2026 release reporting 73.4% SWE-bench Verified.

Verdict: dropped from the candidate list entirely. Qwen3.6-35B-A3B (or its
safer sibling Qwen3-30B-A3B, see below) won on every axis — validated
conversion path, real on-hardware data once benchmarked, and a current
competitive profile.

### Qwen3.6-35B-A3B on OpenVINO's GPU plugin — the long story of why it was rejected

This is the significant piece of research in this project, so it's worth
walking through in full rather than jumping straight to the conclusion.

**Early desk research (before any hardware testing)** flagged Qwen3.6-35B-A3B
(256 experts, 8+1 active, ~3B active compute, 35.95B total, Apache 2.0,
thinking-on-by-default) as "maybe, with caveats, not a clean go":

- A community benchmark on essentially the same hardware generation
  (Core Ultra 7 258V, Arc 140V, 32GB unified RAM — 2GB *more* than this
  machine) showed the model fully resident on the GPU plugin collapsing to
  **1.4–1.7 tok/s** from paging. Not viable as-is.
- OpenVINO 2026.3.0 had just added a GPU-plugin MoE expert-streaming
  feature (`OFFLOAD_RATIO`) aimed at exactly this problem. At a ratio of
  ~20–30, the same reference hardware reportedly reached **10–21 tok/s** —
  usable, but time-to-first-token rose sharply (2.1s → ~22s) since experts
  stream in cold on every request.
- GPU MoE kernels were reported unfused (an open OpenVINO GitHub issue) —
  roughly half the throughput of llama.cpp's Vulkan backend on identical
  hardware even when everything else worked.
- No confirmed Linux equivalent existed for the Windows "Shared GPU Memory
  override" the reference benchmarks depended on — this system's `xe`/i915
  + compute-runtime stack might expose a smaller usable window, meaning the
  paging collapse could arrive sooner here. Flagged explicitly as something
  that had to be observed directly, not assumed.
- A safer fallback was queued in parallel: `OpenVINO/Qwen3-30B-A3B-int4-ov`
  (16.3GB int4), reported to fit fully resident at 26–30 tok/s on the
  reference hardware with no offload tuning needed at all.

**Stage 0 smoke tests, run directly on this hardware, initially looked
promising for both variants.** Getting OpenVINO's GPU device detection
working at all required a real workaround first: nixpkgs' `openvino`
package is built with `ENABLE_ONEDNN_FOR_GPU=false` (to dodge a C++ ODR
compile error in `intel_gpu`'s graph code), which makes any GPU-side MoE
inference crash immediately with `moe_3gemm_swiglu_opt depends on onednn`.
The fix was to bypass nixpkgs entirely and install the official PyPI wheels
(`openvino`, `openvino-genai`, `openvino-tokenizers`) into a venv, which
ship with oneDNN-for-GPU enabled; `programs.nix-ld.enable = true` plus the
already-cached `intel-compute-runtime`/`ocl-icd`/`intel-gmmlib`/
`intel-graphics-compiler` packages were enough to make the manylinux wheels
work, with `OCL_ICD_VENDORS` pointed at compute-runtime's OpenCL vendor
directory for device detection.

With that in place:

- **`Qwen3-30B-A3B-int4-ov`, fully resident** — passed, and beat the
  reference number: load 31.4s, TTFT 0.99s, 778 tokens in 21.2s →
  **36.75 tok/s**. Coherent, correct output on a DAG/cycle-detection test
  prompt.
- **`Qwen3.6-35B-A3B-int4-ov`, `OFFLOAD_RATIO=20`** — also passed, closely
  matching the uncertain research estimate: load 10.6s, TTFT 23.4s
  (predicted ~22s), 517 tokens in 45.2s → **11.44 tok/s** (predicted range
  10–21 tok/s). This IR turned out to require `VLMPipeline`, not
  `LLMPipeline` — the downloaded model directory ships a full VLM export
  (separate language/text-embedding/vision-embedding/vision-merger
  components), and `LLMPipeline` fails outright with "Port for tensor name
  input_ids was not found."
- **A concurrency smoke test** (NPU-hosted 7B warm, GPU MoE model
  generating simultaneously) passed with no OOM, but on a thin margin:
  `available` memory bottomed at ~5.8–6.1GB, free RAM as low as ~268MB, and
  the kernel had to squeeze `buff/cache` from 20GB down to ~10GB to make
  room. With 0B disk swap configured at the time, there was no graceful
  degradation path left if anything else needed a burst of memory.

**The real problem only surfaced once thinking mode was turned on for a
full 10-prompt benchmark**, which is required for this model's default
behavior. With thinking mode on, **every OpenVINO int4 GPU generation for
both MoE candidates ran out its full token budget without ever closing
`<think>`**, regardless of decoding strategy — tried and ruled out: greedy
decoding (0/10 complete at both 1200 and 4000 tokens) and the model's own
recommended sampling parameters (temp=0.6, top_p=0.95, top_k=20 — still
0/10 complete, spiraling into repetitive self-second-guessing).

Rather than treat this as a dead end for the model, the question was
reframed: is this the model, or the delivery mechanism? Two more avenues
were checked:

- OpenVINO GenAI does have native GGUF-direct-load support, but as of this
  OpenVINO version it only covers SmolLM and Qwen2.5 topologies — loading a
  `qwen3moe`-architecture GGUF crashes immediately
  (`IndexError: unordered_map::at`) on both CPU and GPU. Not usable.
  Confirmed via OpenVINO's own blog post announcing the feature.
- `llama-cpp-vulkan` (already packaged in nixpkgs) with a genuine Q4_K_M
  GGUF, run entirely independently of OpenVINO, worked correctly out of the
  box. Vulkan cleanly detected the Arc iGPU via Mesa's `anv` driver, with no
  NixOS graphics configuration beyond the `mesa`/`vulkan-loader`/
  `vulkan-tools` packages plus `VK_ICD_FILENAMES`. A single-prompt spot
  check on both `Qwen3-30B-A3B` and `Qwen3.6-35B-A3B` produced complete,
  correct, cleanly-terminated answers at ~27 tok/s each — no rambling, no
  crash.

That spot check justified a full, matched 10-prompt head-to-head for
`Qwen3.6-35B-A3B` — same 4000-token budget, same sampling
(temp=0.6/top_p=0.95/top_k=20), thinking mode on in both cases:

| | OpenVINO int4 (GPU, `VLMPipeline`, `OFFLOAD_RATIO=20`) | llama.cpp Q4_K_M (Vulkan, `-ngl 99`) |
|---|---|---|
| Converged to a complete, correct answer | **0 / 10** | **9 / 10** |
| Never closed thinking, ran out the token budget | 6 / 10 | 0 / 10 |
| Crashed outright (`CL_OUT_OF_RESOURCES`) | 4 / 10 | 0 / 10 |

The full methodology and per-prompt detail for this benchmark lives in
`docs/07-benchmarks-and-methodology.md`; the one-line verdict is what
mattered for this decision: **the model is genuinely good on this
hardware — it's OpenVINO's GPU-MoE inference path, both the fully-resident
and `OFFLOAD_RATIO`-streamed variants, that's broken, not
`Qwen3.6-35B-A3B` or `Qwen3-30B-A3B` themselves.** The four outright
crashes are notable on their own: a GPU driver-level stability bug that
worsened across repeated large generations within one process, eventually
killing the whole process on exit.

**Decision: drop OpenVINO as the GPU-tier delivery mechanism entirely.**
`llama-cpp-vulkan` + a Q4_K_M GGUF became the confirmed path for both GPU
candidates, with full GPU residency (`-ngl 99`) chosen deliberately over
any partial/memory-constrained offload — since the NPU and GPU tiers are
kept policy-mutually-exclusive (enforced at the systemd level via
`Unit.Conflicts` in both services' definitions, not just as a stated
intention), there's no concurrent-memory-pressure scenario that would make
llama.cpp's own MoE-expert-offload flags worth exploring.

`Qwen3.6-35B-A3B` became the production model given it passed decisively.
`Qwen3-30B-A3B` — the safer, non-thinking-mode-required fallback — was only
ever spot-checked once, never run through the full 10-prompt suite; it
remains a reasonable candidate to fall back to if you hit a wall with
`Qwen3.6-35B-A3B` on different hardware, but its correctness numbers here
are genuinely unvalidated beyond that single prompt. Model files for both
were pulled as pre-quantized GGUFs — `Qwen/Qwen3-30B-A3B-GGUF`
(`Qwen3-30B-A3B-Q4_K_M.gguf`, ~18GB) and
`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
(`Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`, ~21GB).

## 2. Vulkan wiring: no `hardware.graphics`, no `hardware.opengl`

The single biggest simplification that came out of the pivot away from
OpenVINO: getting the Arc iGPU visible to `llama-server` needs **no
system-wide NixOS graphics module at all**. No `hardware.graphics.enable`,
no `hardware.opengl` (the OpenVINO/OpenCL path explored earlier did need a
real GPU/OpenCL stack — `intel-compute-runtime`, `ocl-icd`,
`intel-gmmlib`, `intel-graphics-compiler` — but none of that carried over,
since it was abandoned along with OpenVINO).

Instead, `configs/gpu-tier/default.nix` wires Vulkan up ad hoc, scoped
entirely to the one systemd service:

```nix
gpuLibraryPath = pkgs.lib.makeLibraryPath [
  pkgs.mesa
  pkgs.vulkan-loader
];
vkIcd = "${pkgs.mesa}/share/vulkan/icd.d/intel_icd.x86_64.json";
```

`VK_ICD_FILENAMES` is set to Mesa's Intel Vulkan ICD JSON directly, and
`LD_LIBRARY_PATH` is extended with `mesa` and `vulkan-loader`, both only
for the `gpu-server-hard` unit's own `Service.Environment` — nothing global
changes on the rest of the system. Vulkan cleanly detects the Arc iGPU
through Mesa's `anv` driver this way with no further kernel or driver
configuration on this hardware.

If Vulkan doesn't detect your iGPU at all with this setup, that points to a
missing kernel driver or a genuinely different graphics stack requirement
on your hardware — outside the scope of what this ad hoc module handles;
see `configs/gpu-tier/README.md` for the same caveat.

## 3. The chat-template crash and fix

**Symptom**: a plain HTTP 500 from `llama-server`, with a message to the
effect of "System message must be at the beginning," the instant a request
contains more than one system-role message anywhere except position 0 in
the conversation.

**Why codecompanion triggers this on essentially every real session**:
Qwen3.6's own embedded chat template (`tokenizer.chat_template` inside the
GGUF) only special-cases `messages[0]` for the system role; any *later*
message with `role: "system"` hits a `raise_exception` guard instead of
being rendered. codecompanion.nvim's `@{agent}` plus any
`<rules>...</rules>` file-context attachment sends exactly that shape — its
own system prompt, followed by a second system-role message carrying
rule/context files — so this isn't an edge case for tool-using sessions,
it's close to guaranteed.

**The fix** is a single line, changing a `raise_exception` into an ordinary
render:

```diff
     {%- if message.role == "system" %}
         {%- if not loop.first %}
-            {{- raise_exception('System message must be at the beginning.') }}
+            {{- '<|im_start|>system\n' + content + '<|im_end|>' + '\n' }}
         {%- endif %}
     {%- elif message.role == "user" %}
```

See `configs/gpu-tier/chat_template.diff` for this exact diff plus the
Python snippet (using the `gguf` package) used to extract the original
template from the GGUF in the first place, and
`configs/gpu-tier/chat_template.patched.jinja` for the complete 153-line
patched template with the fix already applied. It's wired into
`llama-server` via the `--chat-template-file` flag (see the flag table
below) — `configs/gpu-tier/default.nix`'s `chatTemplateFile` argument
points at the patched file.

If you're on a different GGUF, don't reuse the patched template verbatim —
extract your own model's template and look for the same "must be at the
beginning" (or equivalent) guard on subsequent system messages, then apply
the same render-instead-of-raise fix. The exact wording and template logic
varies by model and version.

## 4. `ExecStart` flag rundown

The full command assembled in `configs/gpu-tier/default.nix`'s
`mkGpuService`:

```
llama-server -m <modelPath> -ngl 99 -c 32768 -ub 4096 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ctk q8_0 -ctv q4_0 -fa on --parallel 1 --reasoning off \
  --port <port> --chat-template-file <chatTemplateFile>
```

Every flag below was chosen from a real, on-hardware benchmark, not
guessed. This table gives the one-line "what" and "why" with the headline
number for each; the full sweep tables and methodology (ubatch-size sweep,
speculative-decoding trial, KV-cache-quantization comparison) live in
`docs/07-benchmarks-and-methodology.md` — this section deliberately
doesn't re-derive them.

| Flag | What it does | Why this value |
|---|---|---|
| `-ngl 99` | Offloads all model layers to the GPU (full residency). | The NPU and GPU tiers are kept policy-mutually-exclusive via systemd `Conflicts=`, so there's no concurrent-memory-pressure scenario to optimize for — no reason to explore partial/memory-constrained offload. |
| `-c 32768` | Context window size in tokens. | Raised from an initial 16384 after a real conversation needed 16,923 tokens. Bigger (49152/65536) was tested and rejected — see the context-window section below; this is the largest size confirmed stable under real sustained load. |
| `-ub 4096` | Physical batch size for prompt processing (default 512). | A full sweep of 512/1024/2048/4096/8192 against a real ~30K-token prompt (300s budget) found 4096 the measured ceiling: **102.29 tok/s** full-prompt average, the only value completing in budget; 8192 was no faster and also missed budget. |
| `--spec-type draft-mtp` | Enables self-speculative decoding using this GGUF's own built-in Multi-Token-Prediction ("nextn") head — the `blk.N.nextn.*` tensors otherwise logged as "unused" on every load. | Verified across 4 varied prompts: **+30–45% generation throughput** (35–39 tok/s vs. ~26.6–27 tok/s baseline), 68–84% draft acceptance, no correctness regressions. No separate draft model to manage — it's free against the same GGUF. |
| `--spec-draft-n-max 3` | Caps the speculative draft length at 3 tokens per step. | The value used in the verified benchmark above; mean accepted draft length landed at ~3–3.5 tokens, so this matches the draft head's real effective range. |
| `-ctk q8_0 -ctv q4_0` | Asymmetric KV-cache quantization: K-cache at q8_0, V-cache at q4_0. | 3-fixed-seed retesting found q8_0/q4_0 statistically tied with q8_0/q8_0 on speed and draft acceptance (~78.6–78.7% vs. baseline f16/f16's ~77.3%), but saving **~2GB** of memory (26GB→24GB used at this context size) for no measured cost. |
| `-fa on` | Enables Flash Attention. | Required by llama.cpp for a quantized V-cache (`-ctv`) to work at all — not optional given the KV-cache choice above. |
| `--parallel 1` | Runs a single inference slot instead of llama-server's default of 4. | With the default of 4 slots, the requested context window gets split across them — observed truncating effective context to roughly 1/4 of what was requested. `--parallel 1` keeps the full `-c 32768` on one slot. |
| `--reasoning off` | Sets the server-wide default for thinking mode to disabled. | A dedicated benchmark (`docs/04-thinking-mode-and-preservation.md`) found thinking mode costs nothing in tok/s but hurts task convergence badly (7/11 vs. 11/11 correct across 5 verifiable prompts) with no upside for typical short requests. A per-request `chat_template_kwargs: {"enable_thinking": true}` override still fully re-enables it for one request regardless of this default — see `docs/05-nvim-integration.md` and `docs/06-opencode-integration.md` for how each client uses that override. |
| `--chat-template-file <path>` | Overrides the model's embedded chat template with the patched copy. | Fixes the multi-system-message crash described in section 3 above. Omit this flag entirely if you're not hitting that crash on your own GGUF/client combination. |

## 5. The context-window decision: why 32768, not bigger

`-c 32768` looks like a conservative, round-number choice, but it's the
result of deliberately testing larger windows and hitting a real, confirmed
GPU driver stability wall — not a memory limit, which is the non-obvious
part worth calling out explicitly.

**What was tested**: `-c 49152` and `-c 65536`, on top of the current
production config (`-ub 4096` + `draft-mtp` both enabled), to see whether a
bigger window was viable now that KV-cache quantization had freed some
memory headroom.

**Memory scaled gradually, not as a cliff** — contrary to the first
impression. With MTP enabled in all three cases: 32768 context left
~5.2–5.6GB available, 49152 left ~4.8GB, 65536 left ~4.3GB. The 65536 case
additionally required raising `ANV_SYS_MEM_LIMIT` — the Mesa/ANV
environment variable controlling what fraction of system RAM the Vulkan
driver exposes as its device-local heap, 75% by default — above its
default, because the driver's own heap ceiling, not available system RAM,
was the actual blocker at that size. This was confirmed directly via
`vulkaninfo`: a single ~23GiB heap, hitting `errorOutOfDeviceMemory` well
before system RAM itself ran low. This is the real Linux/Mesa equivalent of
the Windows "Shared GPU Memory Override" flagged as an open question
during the earliest OpenVINO research phase (section 1 above).

**The actual blocker was stability, not memory.** Running a real
~43K-token prefill at `-c 49152` triggered a genuine GPU driver crash —
`vk::DeviceLostError`, confirmed at the kernel level via `journalctl -k`:
`xe ...: Tile0: GT0: Timedout job ... in llama-server`, with a GPU
coredump. The driver's own hang-detection watchdog force-reset the device
under sustained heavy load at this context size. The likely cause is
`-ub 4096`'s large batch dispatches, sustained over a very long prefill,
exceeding some internal timeout threshold — not an out-of-memory
condition.

**KV-cache quantization was tested as a possible fix, and didn't fully
solve it.** After enabling `-ctk q8_0 -ctv q4_0` (section 4 above), the
same ~43K-token prompt was retried at `-c 49152`. It crashed again with the
identical `vk::DeviceLostError`, but noticeably later — past the previous
failure point (~76% through the 32768-token equivalent range) into the
~90%+ range this time. Quantization reduced memory-bandwidth pressure and
bought some margin, but did not fix the underlying instability. This
confirms the crash is a driver/dispatch-duration problem, not primarily a
memory one.

**Decision: stay at `-c 32768`.** It's the only size actually validated
stable under real sustained long-context load in this project's testing.
A safely larger window would need a different lever — most plausibly a
smaller `-ub` specifically at larger `-c`, since ubatch size is what's most
directly tied to per-dispatch GPU job duration — rather than simply raising
the context number. This was not pursued further; treat any larger context
window as unverified on this stack until re-tested with that kind of
incremental approach.

## Installing

See `configs/gpu-tier/README.md` for the concrete install steps (importing
the module, genericizing `modelPath`, starting the service, and the health
check). In short: no `hardware.graphics` setup is required for this tier,
download a Q4_K_M GGUF of your chosen model and point `modelPath` at it,
patch the chat template only if your model's own template has the same
multi-system-message guard, then `systemctl --user start gpu-server-hard`
and confirm with `curl http://127.0.0.1:8901/health`.
