# GPU tier — file map

See `docs/03-gpu-tier-setup.md` for the full explanation. Quick map of what's here:

- `default.nix` — the systemd user service definition for `llama-server` (llama.cpp, Vulkan backend), including the full flag rundown with inline benchmark justification for every one of them. Genericize `modelPath` to wherever you put your GGUF.
- `chat_template.patched.jinja` — the model's own embedded chat template, patched to fix a real crash (see below). Wired in via `--chat-template-file`.
- `chat_template.diff` — the actual one-line fix as a diff, plus instructions for extracting and patching your own model's template if you're not on the exact same GGUF.

## Installing

1. No system-wide NixOS graphics module is needed for this - Vulkan access is wired ad hoc here via `pkgs.mesa`/`pkgs.vulkan-loader` and `VK_ICD_FILENAMES`. If Vulkan doesn't detect your iGPU at all, that's a `hardware.graphics`/kernel-driver problem outside this module's scope.
2. Download a GGUF quantization of your model (this guide used `Qwen3.6-35B-A3B`, Q4_K_M, ~21GB) and point `modelPath` at it.
3. If your model's chat template has the same "system message must be at the beginning" (or similar) crash on multi-system-message requests, extract and patch it per `chat_template.diff`. If not, drop the `chatTemplateFile` argument entirely and let llama-server use the model's built-in template as-is.
4. Import this module into your home-manager config, alongside the NPU tier if you're running both (see `configs/flake-excerpts/` for the systemd `Conflicts=` pattern that keeps them mutually exclusive).
5. `systemctl --user start gpu-server-hard`, then `curl http://127.0.0.1:8901/health`.

Every flag in `default.nix`'s `ExecStart` was chosen from a real benchmark, not guessed - see `docs/07-benchmarks-and-methodology.md` for the full sweep tables (ubatch size, speculative decoding, KV cache quantization) if you're tuning this for different hardware.
