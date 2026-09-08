{ pkgs, lib, ... }:
let
  # OpenVINO's GPU-plugin MoE inference path (both fully-resident and the
  # OFFLOAD_RATIO expert-streaming variant) was benchmarked and rejected:
  # under a matched 10-prompt/4000-token/thinking-mode-on test it never once
  # produced a complete answer (endless unclosed <think> blocks) and
  # destabilized the GPU driver into outright CL_OUT_OF_RESOURCES crashes on
  # 4/10 prompts. llama.cpp's Vulkan backend, on the exact same hardware and
  # the exact same Q4_K_M quantization, got 9/10 fully correct. See
  # docs/07-benchmarks-and-methodology.md for the full comparison.
  gpuLibraryPath = pkgs.lib.makeLibraryPath [
    pkgs.mesa
    pkgs.vulkan-loader
  ];
  vkIcd = "${pkgs.mesa}/share/vulkan/icd.d/intel_icd.x86_64.json";

  mkGpuService = { port, modelPath, description, chatTemplateFile ? null }: {
    Unit = {
      Description = description;
      # Never run alongside npu-server-coder - see the matching Conflicts=
      # on that service for why (memory margin, not just policy).
      Conflicts = [ "npu-server-coder.service" ];
    };
    Service = {
      Environment = [
        "VK_ICD_FILENAMES=${vkIcd}"
        "LD_LIBRARY_PATH=${gpuLibraryPath}"
      ];
      # -ngl 99: full GPU residency, deliberately not the memory-constrained
      # OFFLOAD_RATIO-style partial offload - the NPU and GPU tiers are kept
      # policy-mutually-exclusive instead, so there's no concurrent-memory-
      # pressure scenario to optimize for here.
      # --parallel 1: keep the full context window on a single slot rather
      # than splitting it across llama-server's default of 4 (which was
      # observed truncating context to ~1/4 the requested size).
      # -ub 4096 (default 512): prompt-processing (prefill) on this Vulkan
      # iGPU backend degrades badly as context grows, and long tool-heavy
      # codecompanion conversations frequently need a near-full reprocess
      # (imperfect KV-cache reuse between turns). Full sweep of 512/1024/
      # 2048/4096/8192 against a real ~30K-token prompt: only 4096 actually
      # completed within a 300s budget (102.29 tok/s full-prompt average);
      # 512/1024/2048 all needed longer, and 8192 was no faster than 4096
      # (also didn't finish in the same budget) - 4096 is the measured
      # ceiling on this hardware, going higher buys nothing. See
      # docs/07-benchmarks-and-methodology.md for the full sweep table.
      # --spec-type draft-mtp: this GGUF ships its own built-in Multi-Token-
      # Prediction head (the "blk.N.nextn.*" tensors logged as "unused" on
      # every load without this flag) - llama.cpp can use it as a free,
      # self-speculative draft source, no separate draft model needed.
      # Verified across 4 varied prompts (3 debugging tasks + a tool-calling
      # request): +30-45% generation throughput (35-39 tok/s vs. ~26.6-27
      # tok/s baseline), 68-84% draft acceptance, no correctness regressions
      # (tool-call format stayed clean). Costs ~2GB extra memory for the
      # draft context - still fits comfortably at this context size.
      # -ctk q8_0 -ctv q4_0 -fa on: asymmetric KV cache quantization (Flash
      # Attention required for a quantized V-cache). A same-day retest with
      # 3 fixed seeds per config (to rule out sampling noise, not just a
      # single run) found q8_0/q8_0 and q8_0/q4_0 statistically tied on
      # speed and draft acceptance (~78.6-78.7% vs. baseline f16/f16's
      # ~77.3%) - q8_0/q4_0 wins only on memory, saving ~2GB (26GB->24GB
      # used at this context size) for no measured quality/speed cost. See
      # docs/07-benchmarks-and-methodology.md for the methodology story
      # behind this (a first, invalid, unseeded comparison overstated the
      # win dramatically). Also confirmed correct on a real needle-in-
      # haystack retrieval test at ~18K tokens.
      # --reasoning off: server-wide default. A real benchmark (see
      # docs/04-thinking-mode-and-preservation.md) found thinking mode costs
      # nothing in tok/s but hurts task convergence badly (7/11 vs 11/11
      # correct across 5 verifiable prompts) with no upside for typical
      # short requests, so "off by default, opt in per-request" is the
      # safer server default. This also matters if you plan to use a client
      # (e.g. OpenCode's @ai-sdk/openai-compatible provider) that has no
      # client-side way to control thinking mode at all - it silently
      # inherits whatever the server defaults to. A per-request
      # chat_template_kwargs: {"enable_thinking": true} override still
      # re-enables reasoning for that one request regardless of this flag -
      # see docs/05-nvim-integration.md and docs/06-opencode-integration.md.
      ExecStart = "${pkgs.llama-cpp-vulkan}/bin/llama-server -m ${modelPath} -ngl 99 -c 32768 -ub 4096 --spec-type draft-mtp --spec-draft-n-max 3 -ctk q8_0 -ctv q4_0 -fa on --parallel 1 --reasoning off --port ${toString port}"
        + lib.optionalString (chatTemplateFile != null) " --chat-template-file ${chatTemplateFile}";
      Restart = "on-failure";
    };
    # Deliberately no Install.WantedBy: started on demand via an nvim keymap
    # (see configs/nvim/), not at login - same pattern as npu-server-coder.
  };
in
{
  systemd.user.services.gpu-server-hard = mkGpuService {
    port = 8901;
    modelPath = "/home/YOUR_USERNAME/models/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf";
    description = "llama.cpp GPU LLM server (Qwen3.6-35B-A3B, Vulkan)";
    # The model's own embedded chat template hard-crashes (500, "System
    # message must be at the beginning") the moment more than one
    # system-role message appears anywhere but position 0 - and
    # codecompanion's rules/context-file attachment (@{agent} plus any
    # <rules>...</rules> file references) sends exactly that shape. Patched
    # copy just renders a later system message as an ordinary system turn
    # instead of raising - see chat_template.diff for the single line
    # changed against the model's original tokenizer.chat_template
    # (extracted via the `gguf` Python package).
    chatTemplateFile = ./chat_template.patched.jinja;
  };
}
