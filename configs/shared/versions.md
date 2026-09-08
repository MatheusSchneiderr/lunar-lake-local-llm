# Versions used

All benchmarks, configs, and behaviors described in this guide reflect this
specific point-in-time software stack. Driver behavior, chat templates, and
llama.cpp/OpenVINO internals change fast - if something in this guide
doesn't match what you observe, check version drift here first before
assuming your setup is wrong.

| Component | Version / pin |
|---|---|
| nixpkgs | `nixpkgs-unstable`, commit `e8be7818e19ada32105a8af937a6a473b38167ca` |
| `llama-cpp` (Vulkan backend, `pkgs.llama-cpp-vulkan`) | 0.2.0 (server reports `system_fingerprint: b10566-bb4caa7`) |
| `openvino` / `openvino-genai` | 2026.3.0 |
| `opencode` | 1.18.21 |
| `codecompanion.nvim` | commit `c8bd2d0f` (check `handlers/init.lua`'s `uses_new_handlers()` if you're on a different commit - the old-vs-new handler format distinction this guide relies on is commit-sensitive) |
| GPU model | `Qwen3.6-35B-A3B`, GGUF Q4_K_M quantization (`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`) |
| NPU model | `Qwen2.5-Coder-7B-Instruct`, OpenVINO IR (int4/NPU-optimized) |

## Hardware this was built and tested on

- **CPU**: Intel Core Ultra 200V ("Lunar Lake")
- **iGPU**: Intel Arc Graphics 130V/140V (Xe2, no dedicated VRAM - shares system RAM)
- **NPU**: Intel AI Boost NPU (integrated, this CPU generation)
- **RAM**: ~30GB total system RAM, 0B disk swap (zram added as a safety net - see `docs/07-benchmarks-and-methodology.md`)
- **OS**: NixOS (unstable channel)

If you're on different-generation Intel hardware (Meteor Lake, Arrow Lake,
etc.) the NPU driver overlay work in `configs/npu-tier/overlays/` should
still apply in spirit (the `ENABLE_NPU_COMPILER_BUILD=OFF` problem is a
nixpkgs packaging choice, not hardware-specific), but exact tuning numbers
(context window ceiling, `-ub` sweep results, memory headroom) are specific
to this iGPU/RAM configuration and will differ.
