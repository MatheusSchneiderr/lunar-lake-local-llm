{ pkgs, lib, ... }:
let
  # llama.cpp SYCL backend (see llama-cpp-sycl-overlay.nix, wire it into your
  # flake's overlays list) replaced the Vulkan backend entirely. Root cause:
  # a confirmed, unfixed upstream llama.cpp/ggml-vulkan coopmat SPIR-V defect
  # on Arc GPUs (ggml-org/llama.cpp#28590) causing vk::DeviceLostError crashes
  # under sustained deep-context use - every Vulkan config tested crashed
  # somewhere between ~31K and ~50K tokens, none survived to 65536. SYCL
  # actually uses this hardware's OpenCL backend under the hood (confirmed
  # via server startup log: "SYCL GPU device 0 does not use Level Zero
  # backend, disabling Level Zero memory API" - NOT a bug, this is expected
  # on this driver stack; forcing ONEAPI_DEVICE_SELECTOR=level_zero:* instead
  # makes SYCL init fail and llama.cpp SILENTLY falls back to CPU). See
  # docs/12-sycl-reversal-and-qwen36-migration-2026-09-10.md for the full
  # engine-search story and the honest reconciliation with this repo's own
  # earlier "SYCL conclusively ruled out" verdict.
  ocl = pkgs.intel-compute-runtime;
  oneapi = pkgs.intel-oneapi-toolkit;
  syclLibraryPath = pkgs.lib.makeLibraryPath [
    pkgs.level-zero
    pkgs.intel-gmmlib
  ];

  mkGpuService = { port, modelPath, description, chatTemplateFile ? null }: {
    Unit = {
      Description = description;
      # Never run alongside npu-server-coder - see the matching Conflicts=
      # on that service for why (memory margin, not just policy).
      Conflicts = [ "npu-server-coder.service" ];
    };
    Service = {
      Environment = [
        "OCL_ICD_VENDORS=${ocl}/etc/OpenCL/vendors"
        "LD_LIBRARY_PATH=${oneapi}/2026.0/lib:${oneapi}/lib:${syclLibraryPath}"
      ];
      # -ngl 99: full GPU residency - the NPU and GPU tiers are kept
      # policy-mutually-exclusive instead of partial-offloading, so there's
      # no concurrent-memory-pressure scenario to optimize for here.
      # No --parallel override: checked directly against server logs before
      # shipping - every SYCL test run (default 4 slots) reports
      # "n_slots = 4, n_ctx_slot = <full requested -c>, kv_unified = 'true'" -
      # each slot gets the FULL requested context, not a quarter of it (the
      # old Vulkan config's --parallel 1 workaround doesn't apply here).
      # -c 131072: comfortably covers a 120-150K token target with 9-10GB
      # RAM still free on a 30GB machine.
      # -b 4096 -ub 2048: a fresh, independent -b/-ub sweep on a real
      # agentic coding task found this combo ~29% faster wall-clock and
      # ~1.7x prefill throughput vs the untouched default (b2048/ub512),
      # with decode speed within noise (~2% lower). -ub is the actual GPU
      # compute chunk size; -b only needs to satisfy b>=ub for logical
      # buffering - they are NOT meant to be set equal. Full sweep table:
      # docs/13-qwen36-sycl-fine-tuning-2026-09-11.md section 1.
      # --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0: with no CLI/request
      # override, llama-server falls through to the GGUF's own embedded
      # generation_config (temp=1.0 for this model) - NOT llama-server's own
      # binary --help default (0.80). These values are Qwen's documented
      # "precise/coding" THINKING-mode preset, set here as the server-wide
      # safety-net default for any client that sends no sampling params of
      # its own (e.g. codecompanion). A client needing the non-thinking
      # preset (temp 0.7/top_p 0.8/top_k 20/min_p 0/presence_penalty 1.5)
      # must send it per-request - see configs/opencode/opencode.json.
      # Deliberately absent: -fa (default `auto`, confirmed effectively
      # mandatory-on - `-fa off` crashes/near-OOMs at any real context size
      # on this 30GB machine), -ctk/-ctv (KV cache quantization, tested and
      # rejected as slower with no benefit), --cache-reuse (architecturally
      # incompatible with this model's hybrid recurrent-attention layers),
      # and GGML_SYCL_F16 (a build flag, tested and rejected - see the
      # overlay file). Full investigation for each:
      # docs/13-qwen36-sycl-fine-tuning-2026-09-11.md.
      ExecStart = "${pkgs.llama-cpp-sycl}/bin/llama-server -m ${modelPath} -ngl 99 -c 131072 -b 4096 -ub 2048 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --port ${toString port}"
        + lib.optionalString (chatTemplateFile != null) " --chat-template-file ${chatTemplateFile}";
      Restart = "on-failure";
    };
    # Deliberately no Install.WantedBy: started on demand via an nvim keymap
    # (see ../nvim/), not at login - same pattern as npu-server-coder.
  };
in
{
  systemd.user.services.gpu-server-hard = mkGpuService {
    port = 8901;
    modelPath = "/home/YOUR_USERNAME/models/gguf/Qwen3.6-35B-A3B-UD-IQ1_M.gguf";
    description = "llama.cpp GPU LLM server (Qwen3.6-35B-A3B IQ1_M, SYCL)";
    # Qwen3.6's default chat template crashes on multi-system-message
    # payloads (codecompanion's rules/context-file attachment shape sends
    # exactly that) - this override renders a later system message as an
    # ordinary system turn instead of raising. This is the SAME patched
    # template file used in the original Qwen3.6/Vulkan attempt
    # (../gpu-tier/chat_template.patched.jinja, ../gpu-tier/chat_template.diff
    # for the single-line diff against the model's original template) - the
    # crash is template-level, not engine-level, so no new patch was needed.
    chatTemplateFile = ../gpu-tier/chat_template.patched.jinja;
  };
}
