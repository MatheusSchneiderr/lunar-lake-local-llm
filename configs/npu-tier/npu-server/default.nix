{ pkgs, lib, ... }:
let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    openvino-genai
    openvino-tokenizers
    fastapi
    uvicorn
  ]);

  mkNpuService = { port, modelPath, modelName }: {
    Unit = {
      Description = "OpenVINO NPU LLM server (${modelName})";
      # Never run alongside gpu-server-hard: both models loaded at once on
      # a ~30GB machine leaves only a thin, unswapped memory margin (see
      # docs/07-benchmarks-and-methodology.md's concurrency test) - starting
      # either one now stops the other instead of risking that.
      Conflicts = [ "gpu-server-hard.service" ];
    };
    Service = {
      Environment = [
        "LD_LIBRARY_PATH=${pkgs.npuLibraryPath}"
        "NPU_MODEL_PATH=${modelPath}"
        "NPU_MODEL_NAME=${modelName}"
        "NPU_SERVER_PORT=${toString port}"
        # Default (4096) leaves barely any room once a tool schema (e.g.
        # insert_edit_into_file's description alone) is included in the
        # prompt - confirmed by hitting the pipeline's hard MAX_PROMPT_LEN
        # check while wiring up tool-calling. 8192 was confirmed to work.
        "NPU_MAX_PROMPT_LEN=8192"
      ];
      ExecStart = "${pythonEnv}/bin/python3 ${./server.py}";
      Restart = "on-failure";
    };
    # Deliberately no Install.WantedBy: started on demand via an nvim keymap
    # (see configs/nvim/), not at login - avoids paying the ~30-70s NPU
    # pipeline load cost until you actually want it.
  };
in
{
  # Benchmarked Qwen2.5-Coder-7B-Instruct against DeepSeek-R1-Distill-Qwen-7B
  # across 15 debugging prompts (both "classic gotcha" and algorithmic/
  # tracing bugs) - Qwen2.5-Coder won decisively on both speed and
  # correctness (9/10 vs 4/10). See docs/07-benchmarks-and-methodology.md.
  systemd.user.services.npu-server-coder = mkNpuService {
    port = 8900;
    modelPath = "/home/YOUR_USERNAME/models/Qwen2.5-Coder-7B-Instruct-npu-ov";
    modelName = "qwen2.5-coder-7b-npu";
  };
}
