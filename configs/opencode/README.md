# OpenCode integration — file map

See `docs/06-opencode-integration.md` for the full explanation. Quick map:

- `opencode.json` — the actual provider/model config. Two provider entries
  (`local-gpu`, `local-npu`) pointing at the same two servers this repo's
  `configs/gpu-tier/` and `configs/npu-tier/` set up, plus a **third model**
  entry (`qwen3.6-35b-a3b-gpu-thinking`) that demonstrates a real discovery:
  a model's `options` object in OpenCode's config is a raw passthrough
  merged directly into the outgoing request body, any key, no whitelist -
  confirmed by proxying OpenCode's real traffic. That's what lets
  `chat_template_kwargs` reach the server per-model here, even though
  OpenCode's own `@ai-sdk/openai-compatible` provider never sends any
  reasoning-control field on its own.
- `default.nix` — installs the `opencode` package and manages the config
  file declaratively via home-manager's `xdg.configFile`.

## Installing

1. Genericize nothing here - there are no personal paths in this file, only
   `127.0.0.1` URLs and model names. Just adjust the ports/model names to
   match your own `gpu-tier`/`npu-tier` setup if they differ.
2. If you only run one tier, delete the other provider block and drop
   `small_model` (or point it at the same model as `model` - see
   `docs/06-opencode-integration.md` for why pointing `small_model` at a
   tier that's `systemd Conflicts=`-mutually-exclusive with your default
   model is a real, easy-to-hit bug).
3. To use the `-thinking` model variant: `opencode run "..." -m
   local-gpu/qwen3.6-35b-a3b-gpu-thinking --thinking` (the `--thinking`
   flag controls *display* of reasoning, not whether it's generated - both
   are needed to actually see it).
