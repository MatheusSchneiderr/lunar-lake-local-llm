# nvim / codecompanion.nvim integration — file map

See `docs/05-nvim-integration.md` for the full explanation. This is an
**excerpt** (via [nvf](https://github.com/NotAShelf/nvf), a Nix flake for
Neovim configuration) — not a full nvim config. Splice the pieces you need
into your own setup:

- `assistant.codecompanion-nvim.setupOpts.adapters` — the `gpu_hard`
  adapter definition. If you're not on nvf, the equivalent in a plain
  Lua `codecompanion.setup({...})` call is the same shape, just without
  the Nix string-wrapping.
- `enable_thinking` schema field + `parse_message_meta` handler — the two
  most reusable pieces if you're wiring up *any* `llama-server`-backed
  adapter with a Qwen3-family thinking model, independent of the rest of
  this repo's specifics.
- `display.chat.token_count` + `display.chat.show_settings` — the
  context-window-percentage display and the setting that makes
  `enable_thinking` (and any other schema field) actually visible/
  editable in the chat buffer.
- `luaConfigRC.gpuServerWaitHelper` + the two `<leader>ac`/`<leader>ai`
  keymap pairs — the "start the systemd service on demand, wait for
  /health, then open the chat" pattern.

## Prerequisites

- [codecompanion.nvim](https://github.com/olimorris/codecompanion.nvim)
  recent enough to have the `openai_compatible` base adapter and the
  `display.chat.token_count`/`show_settings` options (check
  `lua/codecompanion/config.lua` in your installed version if unsure).
- The GPU tier (`../gpu-tier/`) running and reachable at the URL in
  `env.url`.
