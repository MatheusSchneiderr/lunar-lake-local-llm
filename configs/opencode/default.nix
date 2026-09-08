{ pkgs, ... }:
{
  home.packages = [ pkgs.opencode ];

  # Points at the same llama.cpp/gpu-server-hard and OpenVINO/npu-server-coder
  # servers used from Neovim - pure JSON config, no server-side changes
  # needed, since both already speak the OpenAI-compatible wire format via
  # the @ai-sdk/openai-compatible provider adapter. OpenCode doesn't
  # auto-discover models, hence the explicit `models` map per provider.
  xdg.configFile."opencode/opencode.json".source = ./opencode.json;
}
