# Fine-Tuning Update — 2026-09-08

This chapter documents a single, self-contained investigation that changed
the GPU tier's production config after this guide's original numbered
chapters (01-09) were written. Where those chapters describe the
architecture as originally built and tuned, this one describes what
happened when real usage kept feeling slow despite healthy isolated
benchmark numbers — and the root cause turned out to be something the
original tuning pass never isolated. Read this chapter alongside
[docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) (the original flag-by-flag
rationale) and
[docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md)
(the original benchmark backbone, including the speculative-decoding and
KV-cache-quantization numbers referenced throughout this update).

Everything below happened on the exact hardware/software stack pinned in
[configs/shared/versions.md](../configs/shared/versions.md) as of this
date — llama.cpp v0.4.0 / build b10809 (tagged 2026-09-04), Mesa 26.2.1,
kernel 7.2.2. Treat every number here the same way the rest of this guide
asks you to: measured on this machine, on this date, not a universal claim.

---

## 1. Root-causing the wall-clock regression

### Trigger: complaints about wait time, not throughput

Despite `gpu-server-hard` (Qwen3.6-35B-A3B, Q4_K_M, port 8901) reporting
healthy tokens/sec in isolated benchmarks, real usage inside nvim/
codecompanion and OpenCode was slow in the metric that actually matters to
a human at a keyboard: seconds elapsed between hitting send and seeing the
first useful output on a real, long-context prompt. Raw generation
throughput was never the complaint — wall-clock time to an answer was.

An earlier same-day test round had walked through a sequence of individual
lever changes against an 18,642-token real prompt (prefill-only,
`max_tokens:10`): fixing a batch-size clamp bug, an asymmetric `-b`/`-ub`
pairing, a llama.cpp bump to v0.4.0/b10809, a thread-count change,
`GGML_VK_DISABLE_COOPMAT=1`, and removal of `--spec-type draft-mtp`. The
results came back in a strictly increasing sequence — 132s → 139s → 142s →
146s → 163s → 168s → 177s — regardless of what each change actually did.

This was rejected as inconclusive on sight. A monotonic climb across
unrelated, independent flag changes is not what a real per-change effect
looks like; it is what session drift (thermal buildup, driver state,
memory fragmentation) looks like. The demand going into this round was
explicit: rebuild the benchmark methodology so tests don't carry residue
from one another, actually find the mechanism behind any drift instead of
asserting it, and only then draw conclusions — with a wider research pass
(SYCL, vLLM, other alternatives) to follow once the baseline itself could
be trusted.

### Same-config repeat test: is the climb even real?

Instead of testing 8 different configs once each, the exact same
production config was run 7 times back-to-back, with a full `systemctl
stop`/`start` of the server between every run — eliminating warm-cache,
warm-driver-state, or any other cross-run residue as a confound. Same
prompt, same `max_tokens:10`, same everything.

| Run | Wall-clock time |
|---|---|
| 1 | 158s |
| 2 | 135s |
| 3 | 136s |
| 4 | 149s |
| 5 | 134s |
| 6 | 132s |
| 7 | 133s |

Range: 132–158s. Noisy, but flat — no climb, no trend, no run-order
correlation. This is the key result: it demonstrates the earlier night's
monotonic-looking sequence was *not* primarily a session-drift artifact
(thermal ramp, driver warm-up, or similar time-ordered effect). That in
turn means the large deltas observed that night are probably real,
config-driven slowdowns — the asymmetric `-b`/`-ub` config at 210s,
`GGML_VK_DISABLE_COOPMAT=1` at 168s, and removing `--spec-type draft-mtp`
at 177s all sit clearly outside the 132–158s noise band. The small deltas
— the v0.4.0/b10809 bump at 140s, the thread-count change at 143s, and the
"fixed" `-ub 4096` config at 146s — all sit inside it, and are therefore
not distinguishable from noise on this evidence.

This 132–158s band is the baseline used for every comparison in the
sections that follow.

### Ruling out hardware throttling

To make sure the 132–158s spread itself wasn't hiding a hardware-level
cause, the repeat test above was run under continuous 1Hz sysfs/hwmon
sampling:

- GPU clock and power state: `/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq`, `/sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq`, `/sys/class/drm/card0/device/tile0/gt0/freq0/power_profile`
- Per-reason throttle flags: `/sys/class/drm/card0/device/tile0/gt0/freq0/throttle/reason_{pl1,pl2,pl4,prochot,ratl,thermal,vr_thermalert,vr_tdc}`
- hwmon power and temperature sensors

`RATL` (Running Average Thermal Limit — a rolling-average-based power
clamp, distinct from instantaneous thermal cutoffs) was the leading
hypothesis going in, since an averaging limiter is exactly the kind of
mechanism that could produce a slow, session-length drift. It never fired.
Neither did `pl1`, `pl2`, `prochot`, `thermal`, `vr_thermalert`, or
`vr_tdc`. Only `pl4` — an instantaneous power-spike guard — tripped, and
only in about 2.5% of samples, briefly. `power_profile` read `"base"` for
the entire run, ruling out `power-profiles-daemon` silently downgrading to
a power-saving profile mid-session as a cause. This was despite a genuine
observed peak of 92°C on the package. Conclusion: sustained thermal or
power throttling is not what produces the run-to-run variance, and is not
the mechanism behind the earlier apparent climb.

### Ruling out GPU misdetection / coopmat loss

Separately, the hardware was checked against a known llama.cpp Vulkan bug
(GitHub issue #20776), where an Arc 140T (Arrow Lake, same Xe2 family as
this Lunar Lake Arc 130V/140V) can fail to be detected as `INTEL_XE2` on
some driver stacks due to a `minSubgroupSize` quirk — silently disabling
cooperative-matrix (coopmat) support and taking a meaningful chunk of
Vulkan prefill throughput with it, with no error or log line to flag the
loss.

`vulkaninfo` was run with `VK_ICD_FILENAMES` and `LD_LIBRARY_PATH` pointed
at the same Mesa store path production actually uses, to make sure the
check reflected the real runtime environment rather than a system-default
driver. Results for this exact chip (device ID `0x64a0`, reported as
"Intel(R) Graphics (LNL)"):

- Correctly identified (not hitting the #20776 misdetection path)
- `VK_KHR_cooperative_matrix` exposed, revision 2
- `subgroupSize=32`, `subgroupSizeControl=true`

The driver/kernel stack itself was also already essentially bleeding-edge
— Mesa 26.2.1 (2026-08-26) and kernel 7.2.2 (2026-08-16) — both postdating
the upstream Xe2/Arc coopmat and compute-dispatch-timing fixes relevant
here. So there was no version-bump lever available on this axis either:
the stack was already current, and coopmat was confirmed active in
production's own environment, not just in a bare `vulkaninfo` default
context.

Between the flat repeat-test result, the absence of any sustained throttle
reason, and confirmed coopmat availability, the 132–158s band is treated
from here on as genuine baseline noise — not a symptom to chase further —
and the investigation moves to config- and architecture-level causes
instead of hardware or driver state.

---

## 2. Alternatives ruled out: SYCL, vLLM, ggml-openvino

Before settling on llama-cpp-vulkan as the serving path for
Qwen3.6-35B-A3B on the Arc 130V/140V iGPU, three other backends were
evaluated as ways to speed up prefill on this 256-expert MoE model. All
three were rejected. None of the rejections are "insufficiently proven" —
each has a concrete, reproducible blocker documented below, with issue
numbers and dates so the status can be re-checked as upstream moves.

### SYCL backend — actively worse than Vulkan, not just unproven

> **Update (2026-09-10):** this verdict was reconfirmed correct *for the
> flag combination evaluated below* — Flash Attention paired with
> speculative decoding and/or quantized KV cache. A later production
> failure of a different model forced a full re-evaluation, and a config
> that needs neither speculative decoding nor KV quantization ended up
> shipping on this exact backend after all — sidestepping most, but not
> all, of what's cited here. Full honest reconciliation, issue by issue:
> [docs/12-sycl-reversal-and-qwen36-migration-2026-09-10.md](12-sycl-reversal-and-qwen36-migration-2026-09-10.md#2-reconciling-chapter-10s-sycl-verdict-honestly).

llama.cpp's SYCL backend on Xe2 iGPUs has open, confirmed bugs hitting all
three flags this project depends on simultaneously: Flash Attention,
quantized KV cache, and MTP speculative decoding.

- **Flash Attention corruption.** Originally tracked as issue #19276, but
  on re-check that issue was closed as `NOT_PLANNED` (2026-03-14) on a
  technicality — the reporter's environment was an unsupported
  third-party IPEX-LLM container, not a fix. The real, current evidence is
  issue **#28193** (opened 2026-09-01, still **OPEN**): llama.cpp's own
  `test-backend-ops -o FLASH_ATTN_EXT` fails on 2x Intel Arc Pro B70
  (Battlemage/Xe2) for the permuted, Q8_0/Q8_0 KV case, reproduced on
  current master, root-caused to the oneDNN/MKL flash-attention path.
  Maintainer `arthw` responded "We will fix it... one by one"
  (2026-09-02) — acknowledgment only, no merged fix as of this check.
- **Quantized KV cache segfaults.** Noted inside the same #19276 report as
  related behavior on the SYCL path on Xe2 iGPUs, never separately
  tracked or fixed.
- **Speculative decoding memory/performance regression.** Issue **#23203**
  documents a memory-growth and performance regression specific to SYCL
  that doesn't occur on Vulkan: "Vulkan does not show the same level of
  memory growth or performance degradation." This was closed only by
  stale-bot after 14 days of inactivity (2026-07-04) — not a real fix. A
  comment posted **after** that stale-close (2026-08-19, different
  reporter, current build) reproduces the bug on nearly this exact
  configuration (Qwen3.8-27B hybrid-GDN, q8_0 KV, `draft-mtp`), describing
  steady RSS growth and performance degradation over hours that clears
  only on restart.
- **New Lunar-Lake-specific crash.** Issue **#27046** (opened 2026-08-14,
  still **OPEN**): SIGSEGV on GPU offload specifically on Lunar Lake iGPU
  (Arc 140V), bisected to a May 2026 oneMKL-routing commit, reproduces
  with `-fa on -ctk q8_0 -ctv q8_0` on the Level-Zero V2 UR adapter. This
  surfaced during the re-check — the SYCL/Xe2 path has picked up a fresh
  crash bug since the original research, not fewer.
- **Packaging.** Not available in nixpkgs at all — no `llama-cpp-sycl`
  attribute, no `syclSupport` flag. Adopting it would require a
  from-scratch overlay build for a backend that is independently broken
  for this exact use case regardless.

**Verdict: conclusively ruled out.** Reconfirmed with fresher (September
2026) evidence than the original research — the bug surface has grown,
not shrunk.

### vLLM — no mature Intel iGPU path exists

The one Intel-iGPU vLLM path that ever existed, `intel/ipex-llm`, was
archived by Intel in January 2026 (confirmed via the GitHub API:
`archived: true`, `pushed_at: 2026-01-28`), with an explicit "known
security issues, no longer supported" notice. This was a substantial
loss, not a niche wrapper going stale — it was the real Intel Arc/iGPU/NPU
inference project, wrapping and optimizing llama.cpp, Ollama, vLLM, and
HuggingFace Transformers via SYCL/oneAPI for exactly this hardware class,
and had even shipped FlashMoE support for large MoE models (DeepSeek
V3/R1 671B, Qwen3MoE 235B) on Arc GPUs as recently as May 2025 — months
before being archived.

The current official path, `vllm-xpu-kernels`, is validated only on
discrete/datacenter Arc Pro B60/B70 cards with dedicated VRAM —
UMA/shared-memory behavior on an iGPU is undocumented territory. Worse,
vLLM's own GGUF-on-XPU path is documented as **stripping the model's MTP
head**, meaning converting to a vLLM-compatible format would sacrifice the
speculative-decoding speedup entirely, on top of requiring an unvalidated
re-quantization (FP8/AWQ/GPTQ), since Q4_K_M-via-GGUF isn't a supported
combination there. No NixOS packaging exists either — nixpkgs' `vllm`
only supports CUDA/ROCm/CPU.

A possible successor, `intel/llm-scaler` (github.com/intel/llm-scaler),
was checked directly: it's real and actively maintained (not archived,
recently pushed), with genuine MoE support for models like Qwen3-30B-A3B
and Qwen3-235B-A22B. But it explicitly targets **datacenter discrete Arc
Pro GPUs (B60/B70)**, not consumer iGPUs like the 130V/140V — no iGPU
support is mentioned anywhere in its README. An open, unanswered issue on
that repo, **intel/llm-scaler#283**, asks the exact same "does this work
for consumer Arc 140V/140T iGPU users" question this project would ask,
with no maintainer response as of this check.

**Verdict: conclusively ruled out.** The one project that filled this gap
is archived, and its only real successor explicitly doesn't target this
hardware class.

### ggml-openvino — the most promising alternative, firsthand tested, crashes on this exact model

This is a real, in-tree llama.cpp backend (`-DGGML_OPENVINO=ON`), distinct
from the OpenVINO GenAI runtime already rejected earlier in this project.
It translates a standard GGUF through OpenVINO's own graph compiler/
runtime targeting CPU/GPU/NPU, and upstream documentation explicitly
validates it on "Intel Core Ultra Series 2 (Lunar Lake)" — this exact
hardware class. A credible upstream report, issue **#25972**, measured a
real 6.5x prefill speedup over Vulkan on an Arc iGPU for a dense Q4_0
model (917 vs 142 tok/s pp512) — a large, hardware-confirmed win for
dense models.

The catch, from the same and one additional open issue: MoE models crash
on this backend's GPU plugin (issue **#27205**, a shape-mismatch bug that
explicitly names "qwen35moe family, e.g. 35B" — essentially this exact
model), and MTP speculative decoding crashes in every configuration tried
on it (also #25972).

This was tested firsthand, not just cited secondhand. The project already
had a head-start overlay (`overlays/llama-cpp-openvino.nix`, pre-existing
from an earlier abandoned NPU attempt, pinned to llama.cpp v0.4.0 with
`-DGGML_OPENVINO=ON`), which built successfully. Device selection for this
backend is via environment variables, discovered from upstream's
`docs/backend/OPENVINO.md` (not a CLI flag): `GGML_OPENVINO_DEVICE=GPU`
(silently defaults to CPU if unset or unavailable) plus
`GGML_OPENVINO_STATEFUL_EXECUTION=1` for the recommended GPU stateful
mode. The first attempt silently fell back to CPU because the
nixpkgs-built binary needed an explicit OpenCL ICD pointed at Intel's
compute-runtime driver — `OCL_ICD_VENDORS` plus adding
`intel-compute-runtime`'s `lib/intel-opencl` to `LD_LIBRARY_PATH`, a
leftover derivation already present in the Nix store from earlier project
history but not wired into system config.

With GPU correctly targeted (confirmed by the absence of the CPU-fallback
warning), the actual test against the Qwen3.6-35B-A3B Q4_K_M GGUF, GPU
device, no MTP (already known-broken on this backend), `-c 4096`: the
model loaded successfully — weights loaded, threadpool initialized — but
**crashed on the very first decode/prefill** with a concrete OpenVINO
graph-validation error:

```
ggml-openvino: dynamic dim value mismatch for VIEW node 'conv_state_last-0'...
ov::Exception: Check 'is_axis_valid(axis, r)' failed... Axis 3 out of the tensor rank range [-3, 2]
```

Specifically, this validates a Concat node combining a reshaped
`conv_states` tensor with a transposed `qkv_mixed` tensor — a rank-3 vs.
rank-4 shape mismatch. This is a real, firsthand-reproduced instance of the
exact bug class named in upstream issue #27205 for this model family: a
graph-translation bug in the backend itself for this MoE architecture's
attention/expert-routing tensor shapes, not a config or flag issue.

**Verdict: confirmed dead end for this specific model, verified
firsthand.** The backend loads the model but cannot execute its
MoE-specific attention computation graph.

---

All three backends above were tested under the same safety protocol used
throughout this whole update: the production `gpu-server-hard` systemd
service was stopped, a temporary instance was run for the test, `/health`
was verified before and after, production was restored immediately after
each test, and no two full model instances were ever run concurrently
(this hardware has no memory headroom for that).

---

## 3. Three more dead ends (and two of them are on us)

### 1. `--n-cpu-moe` / `--cpu-moe` — tested directly, ruled out for prefill

This one is a real, long-available flag (`--cpu-moe` pins every
routed-expert FFN tensor to CPU RAM; `--n-cpu-moe N` pins only the top `N`
layers, counting down from the highest-numbered layer). It shipped in PR
#15077, merged 2025-08-04 by slaren — well before this project's current
llama.cpp pin, so it was never a version-gap problem. It's also almost
universally documented as a VRAM-fitting mechanism ("model doesn't fit in
VRAM, offload some experts to system RAM so it loads at all"), not a
prefill accelerant. We suspected as much going in, for a structural
reason: llama.cpp's op-offload scheduler (`GGML_OP_OFFLOAD_MIN_BATCH`,
default 32 tokens) copies CPU-resident expert weights back to the GPU and
computes there anyway for any batch ≥ 32 tokens. An 18k+ token prefill is
chunked into ubatches that are all comfortably over that threshold, so the
expectation was that `--cpu-moe` would be silently defeated during
prefill — same Vulkan `MUL_MAT_ID` kernel, same GPU, regardless of where
the loader says the tensor "lives."

We didn't leave that as a prediction. We ran `--cpu-moe` together with
`GGML_OP_OFFLOAD_MIN_BATCH=999999`, which forces the scheduler to actually
compute the MoE experts on CPU during prefill instead of shuttling them
back to GPU — i.e., the genuinely-CPU-bound case — against the real
18,642-token prompt.

| Tokens processed | Elapsed | Running rate |
|---|---|---|
| 2048 | 41s | ~50 tok/s |
| 4096 | 85s | ~48 tok/s |
| 6144 | 127s | ~48 tok/s |
| 8192 | 171s | ~48 tok/s |
| 10240 | 217s | ~47 tok/s |
| 12288 | 266s | ~46 tok/s |

Steady state: **46–50 tok/s**, on pace for ~380–400s total — roughly **3x
slower** than the ~120–140 tok/s Vulkan-GPU baseline we get on this same
prompt without any CPU offload.

**Verdict:** conclusively ruled out, and empirically confirmed rather than
just reasoned about. Genuine CPU compute for this model's experts over a
long prompt is dramatically worse than even our "immature" Vulkan GPU
path. `--cpu-moe`/`--n-cpu-moe` remain useful for fitting a model into
limited VRAM; they are not a prefill fix, and forcing the CPU path to
actually engage (instead of being silently overridden by op-offload) makes
things worse, not better.

### 2. Vulkan FA `small_cache` tile tweak — real, but already shipped, and self-documented as not helping MoE

Digging into llama.cpp issue #18808 turned up a genuine,
maintainer-acknowledged Intel-Vulkan flash-attention tweak: a boolean
originally written as `const bool small_cache = nek1 < 1024` in the
Vulkan FA dispatch code. Forcing it to `true` gave one user a real **2.3x**
prompt-processing speedup on Intel Arc Pro B50 (Battlemage, also Xe2) with
FA enabled.

Tracing it forward: this was generalized and merged into master via PR
#19625 ("Vulkan Scalar Flash Attention Refactor", merged 2026-02-24) as
`reduce_block_rows`, and it is no longer a toggle — it's unconditionally
active for every device where `vendor_id == VK_VENDOR_ID_INTEL`. Our build
(v0.4.0 / b10809, tagged 2026-09-04) postdates that merge by over six
months. So there's nothing to change here: this optimization is already
live the moment you pass `-fa on`.

More importantly, the same GitHub thread that measured the 2.3x win on a
dense model (Devstral-Small-2-24B) explicitly noted it did *not*
meaningfully help a MoE model (GPT-OSS), with the reporting user writing:
*"I just don't understand why it didn't improve PPs much for GPT-OSS,
unlike Devstral."*

**Verdict:** already active via `-fa on`; explains none of our remaining
gap. And its own source data independently corroborates the pattern we
keep hitting from every other angle — MoE is the hard case for Intel
Vulkan, dense models are not.

### 3. Citation correction — the "AMD Strix Halo, 971→1276 tok/s" number did not mean what we said it meant

This is the one we need to be straight about: an earlier round of this
investigation cited "971–1276 tok/s prefill on AMD Strix Halo with tuned
coopmat" (same model, Qwen3.6-35B-A3B) as evidence that real, capturable
performance headroom exists — headroom that Intel's Vulkan path is simply
failing to reach. On re-checking the primary source, that citation
doesn't hold up, on four separate counts:

- **It's a Discussion, not an Issue, with zero replies.** GitHub
  Discussion #22598 ("Tweaking tile geometry for MoE on AMD KHR_coopmat")
  has never received a single comment — no maintainer review, no
  reproduction, no pushback, nothing.
- **971 is the baseline, not the "tuned" result.** We misread the table.
  971.44 tok/s is the pre-existing, unmodified AMD number. The author's
  experimental tile-geometry patch adds **+7.6% to +10.5%** on top of that
  baseline across three quant levels (971→1045, 1112→1229, 1159→1277) — a
  modest tuning gain, not the multi-x uplift our original framing implied.
- **The patch was never merged, never even opened as a PR.** It exists as
  a single commit on the author's personal fork, explicitly gated to
  `VK_VENDOR_ID_AMD` with a non-proprietary driver. It has never touched
  Intel hardware.
- **The benchmark's context length is undisclosed.** No `-c`, no `-ub`, no
  prompt-length is given anywhere in the discussion. The shape of the
  numbers strongly resembles a short `llama-bench pp512`-style run, not
  anything comparable to our real ~18,600-token prefill workload.

**Correction:** this citation should not be treated as evidence of
confirmed, actionable headroom on Intel, and we're retracting the
"971→1276, real headroom" framing specifically. The broader point it was
originally in service of — that Intel's Vulkan MoE matmul path is less
mature than AMD's or NVIDIA's — still stands, but on other, better
evidence gathered separately. This specific number just isn't it, and we
shouldn't have used it the way we did.

---

## 4. Root cause: Flash Attention is the bottleneck, not MoE expert routing

### The question that should have been asked earlier

Every prefill test run across this entire investigation — batch-size
sweeps, cooperative-matrix toggling, quant format changes, CPU-offloaded
MoE experts, alternative backends — was run with `-fa on`. This was not a
deliberate control; it was a structural consequence of the production
configuration. The production server runs a quantized KV cache (`-ctk
q8_0 -ctv q4_0`) to save memory, and llama.cpp requires Flash Attention to
be enabled for quantized KV cache to work at all. Every single test,
without exception, inherited `-fa on` from that dependency.

The investigation had spent its entire effort tuning around `MUL_MAT_ID`,
the MoE-routed-expert matmul, following a maintainer comment that appeared
to describe exactly this kind of slowdown. That citation, on
re-verification, turned out to describe Flash Attention's cost on a dense
model — not `MUL_MAT_ID`, not MoE routing at all (see the citation
correction above). Once the citation was corrected, the obvious question
followed: **had Flash Attention's own cost on this hardware ever actually
been measured in isolation?** It had not. Nobody had tried `-fa off`.

### The 3-way isolating test

Same model (Qwen3.6-35B-A3B), same real 18,648–18,649-token prompt, same
hardware (Arc 130V/140V Xe2 iGPU, ~30GB shared RAM), each run following the
project's stop-production / run-temp-instance / verify-health /
restore-production safety protocol.

| Run | Flags | KV cache | Outcome | Prefill throughput | Wall clock |
|---|---|---|---|---|---|
| 1 | `-fa off`, `-c 32768`, `-ub 4096` | f16 (unquantized, mandatory without FA) | Crashed: `ErrorOutOfDeviceMemory` partway through | **337–641 tok/s** (from completed chunks before crash) | incomplete |
| 2 | `-fa on`, `-c 20000` (no `-ctk`/`-ctv`) | f16 (unquantized) | Completed cleanly | **137.7 tok/s** | 136s |
| 3 | `-fa off`, `-c 20000`, `-ub 1024` | f16 (unquantized) | Completed cleanly | **204.6 tok/s** | 91s |

Run 2 is the test that isolates the variable. It holds KV cache format
constant (f16, unquantized) against the production baseline (~127–158
tok/s, ~132–158s) and toggles only Flash Attention, which stays on. The
result — 137.7 tok/s, 136s — is statistically identical to the
quantized-KV production baseline. **KV cache quantization is not the
cause of the slowdown.** Flash Attention is: on or off is the variable
that moves the number, not `-ctk`/`-ctv`.

Run 3, the clean `-fa off` completion, confirms this from the other
direction: 204.6 tok/s and 91s against the same 20000-token-context
baseline is a real, complete, confirmed **35–45% wall-clock reduction**.
Run 1's partial numbers (337–641 tok/s before the OOM crash) show the
effect is even larger before memory pressure at `-c 32768` intervenes.

### What this proves

The bottleneck is not `MUL_MAT_ID` or MoE expert-routing kernel
immaturity. It is Intel's Vulkan Flash Attention implementation itself, on
this hardware, collapsing at depth on long-context prefill. This
independently reproduces the behavior reported in llama.cpp GitHub issue
#18808 (Flash Attention performance collapse at depth on Intel dGPUs), on
a different model architecture (MoE vs. that issue's dense case) and a
different device tier (integrated Xe2 vs. discrete). Every prior
mitigation attempt in this investigation — batch tuning, cooperative-matrix
flags, quant reformatting, expert offloading, backend swaps — was tuning
around a kernel that was never the actual constraint, because none of them
touched the one flag that was on in every single test.

### The trade-off this creates: no free lunch

Disabling Flash Attention is not a strict win — it forces giving up two
things that depend on it:

1. **Quantized KV cache.** `-ctk q8_0 -ctv q4_0` requires FA; going
   without means a full f16 KV cache sized for the entire context,
   allocated upfront at load time.
2. **MTP speculative decoding's draft context**, which does its own
   attention compute and is equally affected by the FA-off memory
   expansion.

A context/batch sweep with `-fa off` and no MTP found:

- `-c 32768` crashes with out-of-device-memory on every attempt,
  regardless of `-ub` (2048 and 4096 both tested) — and crashes at almost
  the same token position (~12288/18649) every time. This points to the
  fixed upfront f16 KV cache allocation at load time as the limiting
  factor, not ubatch/batch size.
- `-c 24576, -ub 1024` is the confirmed safe ceiling: clean, complete,
  repeatable — **283.7 tok/s, 66 seconds** on a clean run.
- `-c 28672` technically completes but leaves the system at 2.7GB free
  memory with swap actively engaged (confirmed via `free -h`) — a
  genuinely unstable operating point, not a usable one.

Testing MTP (`--spec-type draft-mtp`) combined with `-fa off` is worse:
MTP needs its own separate no-FA attention buffer on top of the main
context's, roughly doubling the memory overhead that was already tight.
`-c 24576`, `-c 20000`, and `-c 16000` all failed to even allocate the MTP
draft context. Only `-c 8000, -ub 512` succeeded. At that size MTP itself
worked correctly — a real generation request showed `draft_n=87`,
`draft_n_accepted=69` (79.3% draft-acceptance rate) and 39.1 tok/s
generation, consistent with MTP's normal performance elsewhere in this
project. MTP's benefit is not the problem.

The context ceiling is. An 8,000–16,000-token context is a hard blocker,
not a minor cost: this project had already raised its context window from
16,384 to 32,768 earlier in its history because a real conversation
needed 16,923 tokens — over the old ceiling. Reverting to 8,000–16,000 to
keep MTP with FA off would put the server below a limit it had already
outgrown, and the long, tool-heavy coding sessions this server exists to
serve would hit that wall constantly. Flash Attention off is a viable
choice for maximizing raw prefill throughput within a ~24K-token ceiling
and no speculative decoding; it is not viable as a drop-in replacement for
the current production configuration alongside MTP.

---

## 5. Full round-trip validation

The isolated prefill-only test that motivated this change was, on
reflection, measuring the wrong thing: it asked for a throwaway 10-token
reply, which hides everything that happens during generation. A real
interactive request is not a prefill benchmark — it's prefill *plus*
several hundred tokens of decode, and the earlier 66–91s vs. 132–158s
numbers said nothing about that second half.

To get a number worth trusting, we built a full round-trip test: the same
real 18.6K-token prompt, but this time asking for a genuine, substantive
500-token response (a detailed code review — "point out at least 5 real
issues or improvements, with explanations"). We ran it 3 times per
configuration, and critically, interleaved (A–B–A–B), not as two
back-to-back blocks, specifically to control for session-level drift.

| Run | Old production config (`-fa on`, MTP, quantized KV, `-c 32768`) | New config (`-fa off`, no MTP, `-c 24576`) |
|---|---|---|
| 1 | 200s | 147s |
| 2 | 173s | 91s |
| 3 | 170s | 95s |
| **Average** | **181s** | **111s** |

The result holds up: every single new-config trial beat every single
old-config trial — even the worst new-config run (147s) beat the best
old-config run (170s). Average reduction: **~39% total wall-clock time,
70 seconds saved**, on a realistic full request.

A side-finding worth flagging: on this harder, more open-ended
code-review content, the old config's MTP draft-acceptance rate was only
~47–63% (291/623, 292/618, 326/519 accepted/total across the 3 runs) —
well below the 68–84% MTP was originally benchmarked at on short,
deterministic debugging prompts elsewhere in this project (see
[docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md)).
Speculative decoding pays off on predictable output; it pays off much
less on prose. That means giving up MTP in this trade costs less than it
looks like it should from the original numbers.

Before finalizing, we didn't assume correctness carried over from the
prefill-only test — we checked it directly on the new `-fa off` config: a
real tool-calling request produced a correctly-formatted `tool_calls`
response, and a real multi-system-message request (exercising this
project's existing chat-template patch for the crash bug where the
model's own embedded template hard-crashes on more than one system-role
message, see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md)) also
worked with no regression.

**Verdict: adopt `-fa off`, no MTP, `-c 24576` as the new production
config.** This is a real, multiply-confirmed, robust win that directly
serves the actual complaint — wall-clock time to an answer — not a
synthetic tok/s number.

## 6. The MXFP4_MOE detour: a result that didn't survive scrutiny

Separately, with `-fa off` already in place, we tried swapping
quantization formats: `unsloth/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, a
~21.7GB requant using MXFP4, a 4-bit floating-point "Microscaling" format
standardized by the Open Compute Project, distinct from the K-quant
integer-block format the project had been using. This requant has no
MTP/"nextn" tensors at all, so MTP was never on the table for it
regardless.

A first-pass comparison — 3 trials per config, but run as two sequential
blocks, all three Q4_K_M/`-fa off` trials first, all three MXFP4 trials
after, not interleaved — looked like a clear win for MXFP4:

| Run | Q4_K_M (`-fa off`) | MXFP4_MOE |
|---|---|---|
| 1 | 147s | 92s |
| 2 | 91s | 95s |
| 3 | 95s | 97s |
| **Average** | **111s** | **94.7s** |

MXFP4 looked ~15% faster on average, and its spread was much tighter (5
seconds vs. Q4_K_M's noisy 56-second spread, driven by that one 147-second
outlier).

This is where the project's own process worked as intended: the result
was challenged before being trusted, on two solid grounds — (1) no
output-correctness testing had been done on MXFP4 at all, only format
checks (tool-calling, chat-template), and (2) the comparison wasn't
interleaved, so a time-based confound (thermal drift, system state,
anything else that changes between two sequential blocks run at
different times) could fully explain MXFP4's apparent edge instead of the
quant format itself.

Both concerns were addressed with a proper retest rather than argued away:

- **Properly interleaved re-test** (Q4_K_M-A → MXFP4-A → Q4_K_M-B →
  MXFP4-B, same full-round-trip methodology), run with a continuous 1Hz
  GPU frequency/throttle/thermal trace the whole time specifically to
  verify — not assume — there was no external confound.

| Run | Q4_K_M (`-fa off`) | MXFP4_MOE |
|---|---|---|
| A | 91s | 94s |
| B | 90s | 94s |

The trace confirmed the sequence was thermally clean throughout: no
throttle reason ever fired, GPU temperature stayed in a 37–55°C range,
and `power_profile` in sysfs stayed at `"base"` for the entire run —
ruling out drift as the explanation for this comparison. Breaking the
numbers down further: prefill speed was statistically identical between
the two quants (~290 tokens/sec both); the entire gap came from
generation speed, where Q4_K_M was consistently ~15% faster (about 19
tok/s vs. MXFP4's ~16.5 tok/s, repeatably across all 4 interleaved runs).

**Verdict, reversed: Q4_K_M is marginally faster than MXFP4_MOE once test
order and thermal drift are properly controlled for.** The original
"MXFP4 is 15% faster" result was an artifact of test ordering combined
with one anomalous 147-second Q4_K_M outlier from the earlier,
non-interleaved comparison — not a real property of the quantization
format. That outlier's exact cause couldn't be retroactively diagnosed (no
thermal trace was running during that earlier test), but Q4_K_M
performing identically-or-better once properly controlled is strong
indirect evidence the outlier was anomalous, not architectural.

- **Accuracy check.** Since the first-pass comparison also hadn't tested
  correctness, we ran a 5-prompt verifiable-answer check (greedy/
  deterministic decoding, `-fa off`/`-c 24576`, both quants): an
  arithmetic word problem (a store sells 37% of 143 apples plus 28 more —
  correct answer 62 apples with standard whole-apple rounding), the
  classic three-mislabeled-boxes logic puzzle (correct strategy: pick from
  the box labeled "Mixed"), a memoized-Fibonacci trace (correct answer:
  fib(6) = 8), a day-of-week calculation from a reference date (Jan 1
  2000 = Saturday → correct answer for Jan 1 2024 = Monday), and a
  filter-square-sort list-comprehension task (correct result: `[64, 16,
  4]` from the even numbers 2, 4, 8 in the input list). **Both quants got
  all 5 prompts substantively correct** — no accuracy regression from
  MXFP4, but no accuracy edge for it either; Q4_K_M if anything followed
  the "single list comprehension" instruction slightly more literally.

**Final decision: quantization stays on Q4_K_M — the MXFP4 swap was
reverted.** The genuinely real part of the change (`-fa off`, no MTP, `-c
24576`) was kept exactly as validated above; only the quantization-format
change is undone, since under proper scrutiny it turned out not to be
real. The downloaded MXFP4_MOE file was left on disk, unused, in case it's
worth revisiting under different conditions later.

---

## 7. Final config: before and after

**OLD** (`gpu-server-hard`, before this investigation):

```
llama-server -m <model_path> -ngl 99 -c 32768 -ub 4096 --spec-type draft-mtp --spec-draft-n-max 3 -ctk q8_0 -ctv q4_0 -fa on --parallel 1 --reasoning off --port 8901 --chat-template-file <patched-template>
```

Model file: `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`

**NEW** (`gpu-server-hard`, after this investigation):

```
llama-server -m <model_path> -ngl 99 -c 24576 -ub 1024 -fa off --parallel 1 --reasoning off --port 8901 --chat-template-file <patched-template>
```

Model file: still `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` (unchanged — an
MXFP4_MOE requant was tried and reverted; see section 6 above).

**Flag-by-flag diff:**

- `-c 32768` → `-c 24576`: context window reduced. Once Flash Attention is
  off, the KV cache is mandatorily unquantized f16, which needs more
  memory per token of context than the old quantized cache did. 24576 was
  empirically confirmed as the safe ceiling — 28672 technically loads but
  pushes system memory into active swap use, and 32768 crashes with
  out-of-device-memory partway through any real long prefill.
- `-ub 4096` → `-ub 1024`: batch size reduced. The no-Flash-Attention
  attention-score-matrix compute buffer needs more memory per batch than
  Flash Attention's fused kernel did; 1024 was the batch size that fit
  cleanly at the new context size.
- `--spec-type draft-mtp --spec-draft-n-max 3` **removed entirely**: MTP
  (the model's built-in Multi-Token-Prediction speculative-decoding head)
  requires its own separate attention compute buffer, which — combined
  with Flash Attention already being off — only fits at a context size of
  8000–16000 tokens, well below a size this project's own history already
  proved insufficient (a real conversation once needed 16,923 tokens
  under the old 16384 ceiling, which is why context was raised to 32768
  in the first place).
- `-ctk q8_0 -ctv q4_0` **removed entirely**: quantized KV cache requires
  Flash Attention to be enabled (Flash Attention performs the
  quantized-value math); once Flash Attention is off, the KV cache is
  automatically full 16-bit floating point (f16), unquantized.
- `-fa on` → `-fa off`: the actual root-cause fix. Flash Attention's own
  implementation is what's slow on this Intel Vulkan/Xe2 iGPU hardware for
  this model — not MoE expert-routing kernel immaturity as originally
  suspected (see section 4 above).
- `--reasoning off`, `--parallel 1`, `-ngl 99`, `--chat-template-file`:
  all unchanged from before, not touched by this investigation.

**Result:** real full-round-trip wall-clock time for a genuine ~500-token
response to an 18.6K-token prompt dropped from an average of 181 seconds
(old config, 3 trials: 200s/173s/170s) to an average of 111 seconds (new
config, 3 trials: 147s/91s/95s) — a ~39% reduction, directly measured,
interleaved, and reproduced 3-for-3.

**Trade-off accepted:** MTP speculative decoding's generation-speed
benefit is lost specifically on short, predictable prompts (where it
previously gave 35–39 tok/s vs. a ~26–27 tok/s non-MTP baseline, from
earlier benchmarking elsewhere in this project) — though on realistic,
substantive/unpredictable output (like the code-review test used in this
investigation), MTP's real benefit had already turned out to be much
smaller than that (only ~47–63% draft acceptance, vs. 68–84% on short
debug prompts), meaning the trade costs less in practice than the raw
numbers alone would suggest. This was a deliberate trade favoring total
wall-clock time to an answer (the actual user complaint this whole
investigation was about) over raw tokens/sec on short-prompt scenarios.

**Status at time of writing:** this config change was written and
syntax-validated but **not yet deployed** as of this update — deploying
it requires a `sudo nixos-rebuild switch` which needs to be run
interactively by the machine's owner (no interactive sudo access was
available in the automated session that did this investigation).

---

## 8. Q&A: clarifying questions asked during this investigation

**"What's f16 KV cache?"**

It's the model's per-token memory of past attention keys/values (so it
doesn't have to recompute the whole conversation from scratch every
token), stored in full 16-bit floating point precision — i.e. not
quantized. The project's normal production config quantizes this cache
down to 8-bit/4-bit (q8_0/q4_0) specifically to save memory, roughly 2–4x
smaller than f16, but that quantization only works when Flash Attention is
enabled (Flash Attention is what performs the quantized-value math). So
when Flash Attention is disabled, the KV cache automatically reverts to
full f16 — this isn't an independent choice, it's the only mode available
once Flash Attention is off, and it's exactly why disabling Flash
Attention also forced the context window down (unquantized f16 cache
needs more memory per token of context).

**"Are we faster or slower on generation speed now that MTP is gone,
compared to when we used MTP?"**

It depends entirely on the type of content being generated. On the
realistic substantive content actually tested in this investigation (long
code-review-style responses), generation speed is essentially a wash if
not slightly better without MTP: with MTP it averaged ~17.4 tokens/sec
(16.3, 16.3, 19.6 across 3 trials) because MTP's draft-acceptance rate on
this kind of open-ended, less-predictable output was only 46–63% (much
lower than the 68–84% MTP got on short, deterministic debug prompts
elsewhere in this project) — when acceptance is that low, the overhead of
computing and rejecting draft tokens eats into MTP's benefit. Without MTP,
generation speed was a consistent ~18.9 tokens/sec across all trials. But
on short, predictable debug-style prompts (from an earlier, separate
benchmark done earlier in this project, not retested under the new
no-Flash-Attention config specifically) MTP was a clear win: 35–39
tokens/sec with MTP vs. only ~26.6–27 tokens/sec without it. So the honest
summary: this change is not universally faster or slower on generation —
it trades away MTP's real win on short/predictable output (a genuine loss
on that content type) in exchange for a ~39% total wall-clock win on
prefill-heavy, realistic requests, where MTP wasn't actually helping much
anyway.

**"What's the actual selling point of the MXFP4 quantization format, if
it's the same file size as Q4_K_M and turned out not to be faster on this
hardware either?"**

MXFP4 ("Microscaling FP4") is a 4-bit floating-point format standardized
by the Open Compute Project (backed jointly by NVIDIA, AMD, Intel,
Microsoft, Meta, Arm, and Qualcomm), where each block of weights shares
one scale factor and each individual weight is itself a small
floating-point number, rather than K-quants' integer-plus-scale/minimum
block scheme. Its size is genuinely not the selling point versus Q4_K_M
(both come out to roughly 4 bits/weight on average) — unlike a format
like IQ4_XS, which trades size and accuracy down together, MXFP4 doesn't
meaningfully move either axis versus Q4_K_M on its own. The real selling
point is hardware acceleration on newer silicon: NVIDIA's Blackwell
generation (and a few other very recent accelerators) has genuinely
native FP4 tensor cores — dedicated hardware that computes matrix
multiplication directly in this format with no separate
dequantize-then-multiply step needed. On that hardware, MXFP4 isn't just
"a quant option," it's a fundamentally faster execution path because the
silicon itself was built for it — which is also why it's the native
release format for some models (like GPT-OSS), letting them skip a lossy
requantization step entirely. There's a secondary, softer claim in the
quantization community that floating-point block scaling may handle MoE
experts' varying weight distributions slightly more gracefully than
integer K-quants, but that's a minor effect, not the main pitch. On this
project's hardware (Intel Arc iGPU, Vulkan backend), there is no
dedicated FP4 execution path at all — Vulkan just dequantizes MXFP4 with
a generic compute shader before doing the same matmul it would do for any
other format, so the one thing that makes MXFP4 special (skipping
dequantization, computing natively in FP4) simply isn't available — which
is exactly consistent with the measured result: no speed win, no accuracy
win, because the differentiating hardware this format is designed around
isn't present on this machine.
