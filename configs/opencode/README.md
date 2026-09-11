# OpenCode integration — file map

See `docs/06-opencode-integration.md` for the original wiring, and
`docs/13-qwen36-sycl-fine-tuning-2026-09-11.md` for the current sampling
presets and the `small_model` fix below. Quick map:

- `opencode.json` — the actual provider/model config. Two provider entries
  (`local-gpu`, `local-npu`) pointing at the same two servers this repo's
  `configs/gpu-tier-sycl/` and `configs/npu-tier/` set up, each GPU model
  entry carrying a **full validated sampling preset** in its `options`
  object — this is a raw passthrough merged directly into the outgoing
  request body, any key, no whitelist (confirmed by proxying OpenCode's
  real traffic), which is what lets `temperature`/`top_p`/`top_k`/`min_p`/
  `presence_penalty`/`chat_template_kwargs` all reach the server
  per-model even though OpenCode's own `@ai-sdk/openai-compatible`
  provider sends none of these on its own.
  - `qwen3.6-35b-a3b-gpu`: thinking ON, `temperature=0.6, top_p=0.95,
    top_k=20, min_p=0, presence_penalty=1.0` — Qwen's own "precise/coding"
    thinking preset plus an empirically-tuned `presence_penalty` against a
    real, documented runaway-reasoning failure mode (see doc 13).
  - `qwen3.6-35b-a3b-gpu-no-think`: thinking OFF via
    `chat_template_kwargs.enable_thinking: false`, plus Qwen's own
    documented non-thinking preset (`temperature=0.7, top_p=0.8,
    presence_penalty=1.5`) — the fastest configuration found in this
    entire project.
  - A third provider, `no-titlegen`, points `small_model` at a
    deliberately unreachable port. OpenCode fires a session-title-
    generation call on every new session via `small_model`, concurrently
    with the real task's first message — pointing it at the same backend
    as `model` means the two compete for one GPU on every fresh session
    (confirmed via a logging proxy, decode speed measurably drops during
    the overlap). Pointing it somewhere unreachable makes that call fail
    fast and silently instead.
- `default.nix` — installs the `opencode` package and manages the config
  file declaratively via home-manager's `xdg.configFile`.

## Installing

1. Genericize nothing here - there are no personal paths in this file, only
   `127.0.0.1` URLs and model names. Just adjust the ports/model names to
   match your own `gpu-tier-sycl`/`npu-tier` setup if they differ.
2. If you only run one tier, delete the other provider block. Keep the
   `no-titlegen` provider and `small_model` pointed at it regardless of
   how many tiers you run — the contention bug above is unrelated to
   which tiers exist, only to `small_model` sharing a backend with `model`.
3. To use the thinking-off variant: `opencode run "..." -m
   local-gpu/qwen3.6-35b-a3b-gpu-no-think`. Both variants stream
   `reasoning_content` when the model produces any (the `interleaved`
   field above) - thinking-off simply asks the model not to produce any
   in the first place.
