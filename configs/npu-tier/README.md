# NPU tier — file map

See `docs/02-npu-tier-setup.md` for the full explanation. Quick map of what's here:

- `overlays/intel-npu-compiler.nix` — Nix overlay that extracts Intel's official prebuilt NPU driver-side compiler from their GitHub release `.deb` tarball (nixpkgs' `intel-npu-driver` ships with the compiler build disabled by default). Register this in your flake's `nixpkgs.overlays`.
- `overlays/npu-runtime-libs.nix` — a second overlay, depends on the first, that builds the exact `LD_LIBRARY_PATH` (`npuLibraryPath`) needed for OpenVINO's NPU plugin to actually find that compiler at runtime. Also register this one.
- `overlays/llama-cpp-openvino.nix.unused` — **not used, kept for reference only**. An abandoned attempt at running llama.cpp's own OpenVINO backend against the NPU/GPU instead of the approach this guide actually uses. See `docs/02-npu-tier-setup.md`'s "path not taken" note.
- `npu-server/server.py` — a ~990-line FastAPI server wrapping OpenVINO GenAI's `LLMPipeline` in an OpenAI-compatible `/v1/chat/completions` endpoint. This is the bulk of what makes a small/weak local model usable as a real coding-assistant backend: multi-convention tool-call parsing, context compaction, a forced-file-search heuristic, and degenerate-stall retry logic. Copy as-is — it reads all its configuration from environment variables, no paths to edit inside it.
- `npu-server/default.nix` — the systemd user service definition, with the model path genericized (`/home/YOUR_USERNAME/...` — replace with wherever you actually put the model). See `configs/shared/placeholders.md`.

## Installing

1. Add both overlays to your flake's `nixpkgs.overlays` list, in this order (compiler overlay first, runtime-libs overlay second — the latter references the former's output).
2. Add `hardware.cpu.intel.npu.enable = true;` to your NixOS configuration (see `configs/flake-excerpts/`).
3. Download/convert your model to an OpenVINO IR directory (see `docs/02-npu-tier-setup.md` for how this project did it) and point `modelPath` in `default.nix` at it.
4. Import this module (`../../configs/npu-tier` in this repo's layout, or wherever you place it in yours) into your home-manager config.
5. `systemctl --user start npu-server-coder`, then `curl http://127.0.0.1:8900/health`.
