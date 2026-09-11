# GPU tier (SYCL) — current

This directory replaces [`../gpu-tier/`](../gpu-tier/) as the recommended
GPU-tier configuration. `../gpu-tier/` is kept in place as the historical
Vulkan-era reference for [docs/01-09](../../docs/) — it is not deployed by
this configuration and does not need to be deleted or migrated.

Two changes happened at once here, both covered in detail across two
chapters:

- **Engine**: `llama-cpp-vulkan` → a from-scratch `llama-cpp-sycl` build
  (`llama-cpp-sycl-overlay.nix`), escaping a confirmed, unfixed Vulkan
  coopmat crash on Arc GPUs at deep context. Full story, including an
  honest reconciliation with this repo's own earlier "SYCL conclusively
  ruled out" verdict from
  [docs/10-fine-tuning-update-2026-09-08.md](../../docs/10-fine-tuning-update-2026-09-08.md):
  [docs/12-sycl-reversal-and-qwen36-migration-2026-09-10.md](../../docs/12-sycl-reversal-and-qwen36-migration-2026-09-10.md).
- **Model**: `Cohere North-Mini-Code-1.0` → `Qwen3.6-35B-A3B` (IQ1_M GGUF)
  after North-Mini failed in real production use despite passing full
  validation. Full fine-tuning pass on top of this new engine/model pair —
  batch sizing, sampling presets, a runaway-thinking root-cause
  investigation, and several rejected optimization attempts (KV-cache
  quantization, `-fa off`, `--cache-reuse`, `GGML_SYCL_F16`):
  [docs/13-qwen36-sycl-fine-tuning-2026-09-11.md](../../docs/13-qwen36-sycl-fine-tuning-2026-09-11.md).

## Files

- `llama-cpp-sycl-overlay.nix` — wire this into your flake's
  `nixpkgs.overlays` list (alongside the NPU-tier overlays in
  [`../npu-tier/overlays/`](../npu-tier/overlays/)) to build
  `pkgs.llama-cpp-sycl`. Needs `intel-oneapi-toolkit` for the DPC++
  compiler (`icx`/`icpx`) — the open `adaptivecpp`/
  `generic-sycl-components` packages do not satisfy llama.cpp's SYCL
  cmake detection.
- `default.nix` — the `gpu-server-hard` systemd user service, same
  `mkGpuService` pattern as `../gpu-tier/default.nix`. Replace
  `YOUR_USERNAME` with your own (see
  [`../shared/placeholders.md`](../shared/placeholders.md)).

The chat-template patch (`../gpu-tier/chat_template.patched.jinja` +
`.diff`) is reused as-is — the crash it fixes is template-level, not
engine-level, so no new patch was needed for the SYCL build.

Pinned versions for this configuration are in
[`../shared/versions.md`](../shared/versions.md) (SYCL-era section).
