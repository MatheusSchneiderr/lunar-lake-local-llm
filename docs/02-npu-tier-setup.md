# NPU Tier Setup

This is the "quick tasks" tier: a dense, coder-tuned 7B model running entirely on the Intel NPU, served over an OpenAI-compatible HTTP endpoint at port 8900. It's meant for fast, cheap completions — not the model you send hard multi-file refactors to (that's the GPU tier, see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md)). This doc covers everything between "fresh NixOS install" and "curl returns a completion": the kernel-level NixOS toggle, the two custom Nix overlays that make OpenVINO's NPU plugin actually work on nixpkgs, the abandoned alternative approach, why this specific model was chosen, how the bespoke Python server compensates for a genuinely weak model's quirks, and the systemd unit that runs it.

Everything here reflects the exact software stack in [configs/shared/versions.md](../configs/shared/versions.md) — treat version-specific numbers and behaviors as a snapshot, not a guarantee.

## 1. The NixOS-level toggle

Before any of the Nix overlay work matters, the kernel needs to actually expose the NPU as a device. That's a single NixOS module option:

```nix
hardware.cpu.intel.npu.enable = true;
```

See [configs/flake-excerpts/nixos-configuration-snippets.nix](../configs/flake-excerpts/nixos-configuration-snippets.nix) for the exact excerpt (it lives in `hardware-configuration.nix` alongside `hardware.cpu.intel.updateMicrocode`).

What this actually does: it pulls in the `intel_vpu` kernel driver module and the associated udev rules that create `/dev/accel/accel0` (the NPU's device node) with the right permissions for a regular user process to open it. Without this option, `/dev/accel/accel0` simply doesn't exist — none of the overlay work below, none of OpenVINO's NPU plugin, none of `server.py`, matters at all, because there's no device for any of it to talk to. This is the one truly hardware/kernel-level dependency in this whole tier; everything downstream is userspace plumbing.

If you're on a different Intel NPU generation (Meteor Lake, Arrow Lake, etc.), this same option should still be the right starting point — it's the standard NixOS module for `intel_vpu`, not something specific to Lunar Lake. What *is* specific to this project's exact driver/OpenVINO version pin is everything in the next section.

## 2. The core problem: nixpkgs' NPU compiler is missing

Once the device node exists, the next layer is OpenVINO's NPU plugin, which needs to *compile* a model graph into a form the NPU's fixed-function hardware can execute — this is a genuinely different code path from CPU/GPU inference, not just a different backend flag. That compiler is normally shipped as part of `intel-npu-driver` (the userspace driver package, distinct from the kernel module above).

nixpkgs' `intel-npu-driver` package builds with the driver-side compiler disabled: `ENABLE_NPU_COMPILER_BUILD` defaults to `OFF` upstream, and building it from source pulls in a full LLVM/MLIR toolchain — expensive enough that nixpkgs doesn't do it by default. The practical effect: on stock nixpkgs, you get a working NPU device node and a working Level Zero backend, but the one component that actually turns a model into something the NPU can run doesn't exist. Any attempt to load a model onto `"NPU"` via OpenVINO GenAI fails at the compiler-lookup step, not at the device level — a confusing failure mode if you don't already know this is the cause, since the device itself checks out fine (`vpu-smi`, `/dev/accel/accel0` present, Level Zero happily enumerates the device).

The fix is two Nix overlays, described in detail below.

## 3. Overlay one: `intel-npu-compiler.nix` — extracting Intel's prebuilt compiler

See [configs/npu-tier/overlays/intel-npu-compiler.nix](../configs/npu-tier/overlays/intel-npu-compiler.nix) for the full file.

Rather than build the compiler from source (the LLVM/MLIR cost mentioned above), this overlay takes the pragmatic route: Intel publishes the driver-side compiler prebuilt, packaged as a `.deb`, inside their GitHub release tarballs for `linux-npu-driver`. This overlay downloads that exact tarball (pinned to `v1.35.0`, matching the driver version nixpkgs builds against), unpacks just the one `.deb` it needs (`intel-driver-compiler-npu_1.35.0.20260722-29947505341~ubuntu24.04_amd64.deb`), and extracts two shared objects from it with `dpkg-deb -x`:

- `libopenvino_intel_npu_compiler.so`
- `libopenvino_intel_npu_compiler_loader.so`

It also separately fetches `npu_driver_compiler.h` directly from the matching tag in Intel's `linux-npu-driver` GitHub repo, because the `.deb` only ships the compiled `.so` files, not the public header the build needs. That header has to land directly in `$out` (not a subdirectory) — `npu_compiler.cmake` in `intel-npu-driver`'s own build adds `$NPU_COMPILER_PACKAGE_DIR` itself (not `$NPU_COMPILER_PACKAGE_DIR/include` or similar) to the include search path, so the derivation's `installPhase` places the header at the top level of `$out` to match what the build expects.

The resulting derivation, `intel-npu-driver-compiler-package`, is then wired into nixpkgs' own `intel-npu-driver` via `overrideAttrs`:

```nix
cmakeFlags = (old.cmakeFlags or [ ]) ++ [
  "-DNPU_COMPILER_PACKAGE_DIR=${final.intel-npu-driver-compiler-package}"
  "-DCMAKE_EXE_LINKER_FLAGS=-ltbb"
];
```

The `-ltbb` linker flag is its own small gotcha: the prebuilt compiler `.so` needs `libtbb` at link time, but `intel-npu-driver`'s own CMake validation-test targets don't declare that link dependency themselves — so without forcing it globally via `CMAKE_EXE_LINKER_FLAGS`, the build fails partway through linking those test binaries, even though the actual driver library links fine. `onetbb`, `zlib`, and `zstd` are added to `buildInputs` for the same reason — runtime dependencies the prebuilt compiler binary needs that aren't otherwise pulled in.

The net result of this overlay: `intel-npu-driver` now builds with a real, working NPU compiler embedded, sourced from Intel's own prebuilt binary rather than compiled from source.

## 4. Overlay two: `npu-runtime-libs.nix` — making OpenVINO's plugin actually find it

See [configs/npu-tier/overlays/npu-runtime-libs.nix](../configs/npu-tier/overlays/npu-runtime-libs.nix) for the full file.

Having a compiler built isn't sufficient on its own — OpenVINO's NPU plugin has to be able to *find* it at runtime, and this is where the second, more subtle problem shows up. OpenVINO's NPU plugin resolves the compiler loader **relative to wherever `libopenvino.so` was actually loaded from at runtime** — it calls `get_ov_lib_path()` internally and looks in an `openvino/` subdirectory next to that path. nixpkgs' `openvino` package doesn't ship the compiler library in that location at all (it doesn't ship it anywhere, prior to overlay one), so even with a working compiler built above, a stock `LD_LIBRARY_PATH` pointing at both `openvino.lib` and the compiler package separately still fails, because the plugin isn't doing a generic library search — it's doing a path-relative lookup keyed off its own load location.

The workaround, rather than rebuilding the entire (large) `openvino` package with a patched `postInstall` step, is to construct a *new* directory that looks like what the plugin expects and load `libopenvino.so` from there instead:

```nix
openvino-npu-runtime = final.runCommand "openvino-npu-runtime" { } ''
  mkdir -p $out/lib/openvino
  for f in ${final.openvino.lib}/lib/*; do
    [ -f "$f" ] && ln -s "$f" "$out/lib/$(basename "$f")"
  done
  for f in ${final.openvino.lib}/lib/openvino/*; do
    [ -f "$f" ] && ln -s "$f" "$out/lib/openvino/$(basename "$f")"
  done
  ln -s ${final.intel-npu-driver-compiler-package}/lib/libopenvino_intel_npu_compiler.so $out/lib/openvino/
  ln -s ${final.intel-npu-driver-compiler-package}/lib/libopenvino_intel_npu_compiler_loader.so $out/lib/openvino/
'';
```

It symlinks in every real OpenVINO library (top-level `lib/*` and the existing `lib/openvino/*` plugin subdirectory) unchanged, then adds symlinks to the two compiler `.so` files from overlay one directly into that same `openvino/` subdirectory. Point `LD_LIBRARY_PATH` at `openvino-npu-runtime` instead of the stock `openvino.lib` output, and `libopenvino.so` gets loaded from *this* directory — so the plugin's path-relative lookup finds the compiler sitting right next to it, exactly where it expected it to be all along.

The overlay then assembles the complete runtime library path a process actually needs to reach the NPU end-to-end, exposed as `npuLibraryPath`:

```nix
npuLibraryPath = final.lib.makeLibraryPath [
  final.openvino-npu-runtime
  final.level-zero
  final.intel-npu-driver
  final.intel-npu-driver-compiler-package
  final.zlib
  final.zstd
  final.onetbb
  final.stdenv.cc.cc.lib
];
```

Each entry closes a distinct gap, and the comments in the source file spell out why all of them are needed simultaneously — worth restating here because it's not obvious from the outside:

| Path entry | Why it's needed |
|---|---|
| `openvino-npu-runtime` | Satisfies the OpenVINO plugin's own path-relative compiler lookup, described above. |
| `level-zero` | The Level Zero loader itself — the NPU backend sits behind Level Zero, not a bespoke driver API. |
| `intel-npu-driver` | Ships `libze_intel_npu.so.1`, the actual Level Zero NPU backend. Its own `zeInit()` call *also* `dlopen`s the compiler by bare filename — a second, independent resolution path from the OpenVINO plugin's directory-relative one above, which is why the compiler needs to be reachable two different ways at once. |
| `intel-npu-driver-compiler-package` | The compiler again, by bare filename this time, for the driver's own internal `dlopen` — plain `LD_LIBRARY_PATH` search, no directory trick involved. |
| `zlib`, `zstd`, `onetbb`, `stdenv.cc.cc.lib` | The compiler binary's own runtime shared-library dependencies. Missing these doesn't fail at model-load time — it fails earlier, at `zeInit()`, with `ZE_RESULT_ERROR_UNINITIALIZED`, because Level Zero's init path probes the compiler before any model is even loaded. |

That last row is worth flagging on its own if you hit it: `ZE_RESULT_ERROR_UNINITIALIZED` at `zeInit()` time, before your code has done anything model-related, points at a missing compiler runtime dependency, not a device or permissions problem.

Both overlays must be registered in your flake's `nixpkgs.overlays`, **compiler overlay first, runtime-libs overlay second** — the second one references `final.intel-npu-driver-compiler-package`, defined by the first.

## 5. The abandoned path: `llama-cpp-openvino.nix.unused`

See [configs/npu-tier/overlays/llama-cpp-openvino.nix.unused](../configs/npu-tier/overlays/llama-cpp-openvino.nix.unused) — kept in the repo for reference, **not used by this guide's actual setup**.

This was an earlier attempt at a different architecture entirely: instead of OpenVINO GenAI's own `LLMPipeline` (what `server.py` actually uses), run `llama.cpp` itself with its OpenVINO backend (`ggml-openvino`, enabled via `-DGGML_OPENVINO=ON`) pointed at the NPU or GPU device. The appeal is obvious — one server binary, one wire format, across every backend, instead of a bespoke OpenVINO-GenAI-based Python server for NPU and a separate `llama-server` invocation for GPU.

It got far enough to need two real patches against upstream `llama.cpp` (pinned to tag `v0.4.0`):

1. `naive_compute()` in `ggml/src/ggml-openvino/utils.cpp` unconditionally sets `ov::hint::execution_mode` (`PERFORMANCE` or `ACCURACY`) on every device — but that property is CPU/GPU-only in OpenVINO's own model, and the NPU plugin rejects it outright, crashing every NPU decode. The patch guards both call sites to skip the property when the target device string contains `"NPU"`.
2. `ggml-openvino-extra.cpp` unconditionally requests the `NPU_COMPILER_DYNAMIC_QUANTIZATION` compiler property, an optional NPU speed/memory optimization — but this nixpkgs-built driver's compiler (see the `ENABLE_NPU_COMPILER_BUILD=OFF` problem above, worked around via the prebuilt `.deb` rather than a from-source build with every optional feature enabled) doesn't recognize it, returning `NOT_FOUND`. The patch just drops the property request; it's a perf knob, not required for correctness.

Both patches are real, targeted fixes for real crashes — this wasn't abandoned because it didn't work at all. It's parked because the two-tier architecture that actually shipped (OpenVINO GenAI's `LLMPipeline` directly for the NPU tier via `server.py`, and separately `llama-cpp-vulkan` for the GPU tier — see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) for why the GPU tier landed on Vulkan instead of OpenVINO's GPU backend) ended up being the path that was actually validated end-to-end and benchmarked. This overlay was never carried through a full smoke test → benchmark cycle on the NPU tier specifically, so treat it as a documented, plausible-looking starting point for someone who wants a single-binary llama.cpp-only architecture across both tiers — not as a proven alternative.

## 6. Model selection: why Qwen2.5-Coder-7B-Instruct

Two separate constraints shaped the choice of NPU model, both architectural rather than incidental:

**MoE architectures are categorically unsupported on the NPU.** This isn't a maturity gap that a future OpenVINO release closes — it's a static-shape-compilation vs. dynamic-per-token-expert-routing mismatch. The NPU compiler needs to compile a fixed computation graph ahead of time; MoE's routing decision (which experts run, chosen per-token at inference time) is exactly the kind of dynamic shape the NPU compilation model can't represent. This ruled out every MoE candidate considered for this tier (`Qwen3.6-35B-A3B`, `DeepSeek-Coder-V2-Lite-Instruct`) outright — they became the GPU tier's candidates instead, run via Vulkan (see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md)), not this one.

**A systematic sweep of every OpenVINO release from 2025.3.0 through 2026.3.1 (roughly a year of releases) found no newer NPU-confirmed dense model worth switching to.** Every candidate found was disqualified for a specific, checkable reason — not a vague "didn't look better":

| Candidate | Why disqualified |
|---|---|
| Qwen3-1.7B / 4B / 8B, Gemma-3-4b-it, SmolLM3-3B | General-purpose, not coding-tuned |
| Qwen2.5-VL-3B, MiniCPM-o-2.6 | Multimodal — irrelevant to a text coding assistant |
| Qwen2.5-Coder-0.5B | Same generation, smaller — a downgrade, not an upgrade |
| LFM2 / LFM2.5-1.2B | Liquid AI's own documentation says not to use it for programming |
| Qwen's newer "Coder" line (Qwen3-Coder-30B-A3B, -480B-A35B, -Next) | MoE-only since Qwen3 — ruled out by the same architectural blocker above. **No dense Qwen3-Coder release exists at all.** |

One near-miss worth naming: `Qwen/Qwen3-8B` is dense, NPU-confirmed, and reportedly stronger than Qwen2.5-dense on general STEM/coding benchmarks — but it's general-purpose, not "Coder"-branded, and was logged only as an optional low-priority smoke-test curiosity, never actually run. If you're replicating this and want to explore beyond what's benchmarked here, that's the one candidate worth a look — but it has zero validated benchmark data in this project.

With no newer contender clearing the bar, the actual decision came down to a head-to-head between the incumbent and a plausible-sounding reasoning-tuned alternative: `Qwen2.5-Coder-7B-Instruct` vs. `DeepSeek-R1-Distill-Qwen-7B`, both NPU-viable dense 7B models. Run across 15 debugging prompts spanning both "classic gotcha" bugs and algorithmic/tracing bugs, `Qwen2.5-Coder-7B-Instruct` won decisively: **9/10 vs. 4/10** on the scored subset. Full methodology and the complete prompt/scoring breakdown live in [docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md) — the one-line verdict here is the whole reason this tier still runs `Qwen2.5-Coder-7B-Instruct` as of this writing: it isn't the newest model that could theoretically run here, it's the one that actually won its benchmark, and nothing newer has displaced it.

## 7. `server.py`: making a weak model usable as a tool-calling assistant

`Qwen2.5-Coder-7B-Instruct` is a genuinely weak model by modern-coding-assistant standards — small enough to run comfortably on the NPU, but correspondingly unreliable at exactly the things a coding assistant needs to be reliable at: emitting tool calls in a consistent format, knowing when to stop generating, and not silently giving up mid-investigation. `server.py` (see [configs/npu-tier/npu-server/server.py](../configs/npu-tier/npu-server/server.py), ~990 lines, FastAPI, wrapping OpenVINO GenAI's `LLMPipeline` behind an OpenAI-compatible `/v1/chat/completions` endpoint) exists almost entirely to compensate for that. It's not re-embedded here — walk through the actual file for exact logic — but the four load-bearing design decisions are worth understanding in depth, because they're the reusable part of this tier if you're adapting it to a different weak model.

### Multi-convention tool-call parsing

The model doesn't reliably stick to one tool-call wrapper format turn to turn. Across real usage, `server.py`'s comments log four distinct conventions observed from the *same* model: the documented `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` tag, a ```` ```-fenced ```` JSON block (sometimes with a `json` language hint, sometimes without), bare unwrapped JSON with no wrapper at all, and — least predictably — an invented XML-ish tag built from the tool's own name wrapped around raw arguments, when the model apparently forgets the documented format entirely.

`extract_tool_calls_and_text()` handles this as an ordered cascade, not a single regex: a delimiter scan for the tagged/fenced conventions first, then a direct JSON-parse attempt on the stripped text, then a delimiter-agnostic scan for any balanced `{"name":...,"arguments":...}` span anywhere in the text (`_find_balanced_json_span`), and finally a fallback keyed off the tool's own known name as an XML tag (`_extract_by_known_tag`). If none of that matches, the raw text is returned as ordinary assistant content rather than the server crashing or silently dropping output — a deliberate choice to fail toward "the user sees something," not toward an opaque 500.

This cascade only ever runs once generation is *complete*, not incrementally per-token — `ToolCallScanner` accumulates the full text buffer and only calls `extract_tool_calls_and_text()` in `finish()`. The reasoning is direct: with four live conventions and no reliable per-turn signal for which one is coming, there's no safe partial delimiter to watch for mid-stream without risking a wrong guess. Live token-by-token streaming is therefore only used when a request carries no `tools` at all — which is the large majority of real usage, so this doesn't cost much in practice, but it's why tool-bearing requests in this tier are inherently non-streaming from the model's perspective even if the outer response is streamed back to the client.

### The forced-file-search heuristic

A separate, narrower failure mode: the model has no actual awareness of the real filesystem, and when asked to edit or create a file without being told its exact path, it tends to *invent* a plausible-looking placeholder path (`path/to/file.ext`) rather than admitting it doesn't know one — which then fails later at tool-execution time with a confusing "file does not exist" error, several turns removed from the actual mistake. A system-prompt reminder alone (`PATH_AWARENESS_REMINDER`, injected whenever any tools are present) wasn't sufficient to reliably prevent this on a model this size.

`maybe_force_file_search()` is the deterministic compensation: when the request has both a `file_search` tool and an edit/create tool available, no tool result yet exists in the conversation, no message already grounds a real path (no `<file>` tag, no absolute-path-looking string matched via `ABS_PATH_RE`), and the last user message contains something that looks like a bare filename (`FILENAME_RE`) — the server **skips the model entirely for that turn** and synthesizes a `file_search` tool call itself, deterministically, before the model ever gets to guess. Once any tool result exists anywhere in the conversation, forcing stops and the model takes over with real information already in front of it. A related piece, `maybe_ask_to_disambiguate()`, watches for a `file_search` result (forced or model-initiated) that returns more than one match and short-circuits straight to asking the user which file was meant, rather than letting the model guess among several real candidates — the entire point of forcing the search in the first place was to replace a guess with certainty, and a guess between multiple genuine matches is still a guess.

### Context compaction: cheapest option first

`MAX_PROMPT_LEN` bounds how much prompt the NPU pipeline will accept (see the systemd unit below for the exact value and why). A long, especially tool-heavy conversation will eventually overflow it. Rather than fail outright the moment that happens, `compact_messages_if_needed()` tries three strategies in strict cheapest-first order, each one only attempted if the previous one wasn't enough:

1. **Deterministic truncation of old tool results** (`truncate_old_tool_results()`) — large tool-result payloads (file contents, diffs, search dumps) above `TOOL_RESULT_TRUNCATE_THRESHOLD` (400 characters) get shortened, except the single most recent tool result (`KEEP_RECENT_TOOL_RESULTS = 1`), which stays at full fidelity since it's most likely still directly relevant. This is a pure string operation — no model call, nothing that can go wrong or produce a wrong answer.
2. **LLM-summarization of a leading run of plain dialogue turns** (`summarize_old_turns()`), only if truncation alone still doesn't fit. This is a real generation call against the same not-fully-reliable model, so it's explicitly a last resort rather than a first move — the code comments are direct about this being a deliberate ordering choice, not an oversight.
3. **A clear failure**, if even summarization doesn't bring the prompt under budget — the user gets a message telling them the conversation hit its limit and to start a new chat, rather than the pipeline crashing opaquely deep inside OpenVINO GenAI.

The ordering itself is the design decision worth internalizing: correctness-risk-free operations are exhausted before anything that asks the (weak) model to do more work on its own behalf, and an honest failure is preferred over letting the underlying pipeline fail in a way that doesn't produce a useful message.

### Degenerate-stall single-retry logic

Even with the `KEEP_INVESTIGATING_REMINDER` system-prompt nudge in place, the model sometimes generates a genuinely empty response after a tool result comes back — no text, no tool call, nothing — instead of either continuing the investigation or answering. `is_degenerate_stall()` detects exactly this shape: no `tool_calls`, empty/whitespace-only content, and the last message in the conversation was a tool result. When detected, the server retries **exactly once**, appending an explicit nudge message (`STALL_NUDGE_MESSAGE`: "Continue investigating... Don't just stop with nothing.") and using whatever comes back from that retry — even if the retry is *also* degenerate, the server gives up and returns that (empty) result rather than retrying again. The bound is hard-coded into the control flow, not a counter that could accidentally be raised — this can never turn into a retry loop.

A related but distinct failure surfaced later in this project's broader testing (documented in the plan history, and relevant context if you're deciding whether this class of mitigation is worth building for your own weak model): instead of silence, the model can *confidently and wrongly* claim it has no file access and ask the user to paste file contents, despite tools genuinely being available and working. That variant was observed on this project's GPU tier, not this NPU tier's `server.py` — see the parked `gpu-guard` component in [docs/09-gpu-guard-optional.md](09-gpu-guard-optional.md) for how (and why, and why it was ultimately not deployed) a similar detect-and-retry mitigation was built for that failure shape specifically.

## 8. The systemd service

See [configs/npu-tier/npu-server/default.nix](../configs/npu-tier/npu-server/default.nix) for the full unit definition, and [configs/npu-tier/README.md](../configs/npu-tier/README.md) for the actual installation steps — they aren't re-derived here, only the reasoning behind the notable choices in the unit itself.

A few things about the unit worth calling out explicitly:

- **Port 8900.** Arbitrary but fixed — every client config in this guide (codecompanion, OpenCode) points at this port for the NPU tier specifically.
- **`Conflicts = [ "gpu-server-hard.service" ]`.** The NPU and GPU tiers are deliberately mutually exclusive at the systemd level, not just by convention. This machine has ~30GB total RAM; running both a resident NPU pipeline and the GPU tier's resident `Qwen3.6-35B-A3B` at once was tested directly (see [docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md)'s concurrency test) and found to leave only a thin, unswapped memory margin — survivable in that one test, but not something to rely on as a steady-state default. `Conflicts=` means starting either service automatically stops the other first, so the two can never both be running by accident.
- **On-demand start, no `Install.WantedBy`.** The unit is deliberately not started at login — it's started on demand via an editor keymap (see [docs/05-nvim-integration.md](05-nvim-integration.md)) precisely to avoid paying the NPU pipeline's cold-start cost (in the tens of seconds) every session regardless of whether the NPU tier is actually used that session.
- **`NPU_MAX_PROMPT_LEN=8192`, up from OpenVINO GenAI's own default of 4096.** The default was found to be too tight in practice — once a real tool schema is included in the prompt (a single tool description, like `insert_edit_into_file`'s, can be substantial on its own), 4096 tokens leaves barely any room for actual conversation content before hitting the pipeline's own hard prompt-length check. 8192 was confirmed to work under real tool-calling usage; `server.py` further subtracts a small safety margin from this value internally (`EFFECTIVE_MAX_PROMPT_LEN`) as the actual threshold its own compaction logic targets, keeping a buffer below the pipeline's hard limit rather than compacting right up against it.
- **`Restart = "on-failure"`.** Ordinary crash-recovery, not specific to any NPU-tier quirk.

Installation itself — registering the overlays, enabling the NixOS option, downloading/converting a model to an OpenVINO IR directory, importing the home-manager module, starting the service — is covered step by step in [configs/npu-tier/README.md](../configs/npu-tier/README.md); follow that rather than reconstructing the sequence from this doc.
