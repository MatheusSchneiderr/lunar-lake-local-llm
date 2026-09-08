# OpenCode Integration

This chapter covers wiring [OpenCode](https://opencode.ai) to both local
tiers as separate providers, the `small_model` mutual-exclusivity bug it's
easy to hit if you follow the naive setup, and a real config-passthrough
discovery — model-level `options` in OpenCode's config are a raw,
unvalidated passthrough into the outgoing request body — that was verified
by capturing OpenCode's actual network traffic, not by reading
documentation or guessing. The full config is in
[configs/opencode/opencode.json](../configs/opencode/opencode.json) and
[configs/opencode/default.nix](../configs/opencode/default.nix); this doc
explains the reasoning behind each non-obvious piece of it.

## Provider / model config shape

OpenCode does not auto-discover models from a running OpenAI-compatible
server — every model has to be declared explicitly. This was confirmed by
reading OpenCode's own bundled JSON schema (`share/config.json` inside the
Nix package) rather than relying on external docs, since the schema is the
actual source of truth for what fields are accepted:

- `provider.<id>` takes `npm: "@ai-sdk/openai-compatible"` (the AI SDK
  provider package OpenCode uses for any generic OpenAI-wire-format
  backend), an `options` object with at least `baseURL` and `apiKey`, and a
  `models` map — there is no discovery endpoint OpenCode queries on its
  own.
- Each entry under `models` can declare capability flags (`tool_call`,
  `reasoning`, `temperature`) and, notably, `interleaved:
  "reasoning_content"` — a bonus discovery made while reading the schema
  rather than something initially planned: this tells OpenCode to natively
  surface `llama-server`'s non-standard `reasoning_content` field as
  interleaved reasoning output, the same underlying idea as the
  `parse_message_meta` fix on the codecompanion side (see
  [docs/05-nvim-integration.md](05-nvim-integration.md)), just via a
  built-in config flag instead of custom extraction code.

This repo's [configs/opencode/opencode.json](../configs/opencode/opencode.json)
wires both existing servers as separate providers, mapping directly onto
the project's original two-tier idea via OpenCode's own top-level `model`
and `small_model` fields (`small_model` is OpenCode's own concept for
lightweight background tasks, e.g. session-title generation):

| Provider | Model | Backend | Port | Role |
|---|---|---|---|---|
| `local-gpu` | `qwen3.6-35b-a3b-gpu` | `gpu-server-hard` (llama.cpp/Vulkan) | 8901 | default `model` |
| `local-gpu` | `qwen3.6-35b-a3b-gpu-thinking` | same, `enable_thinking: true` | 8901 | explicitly-selected, see below |
| `local-npu` | `qwen2.5-coder-7b-npu` | `npu-server-coder` (OpenVINO) | 8900 | defined, selectable, not wired as a default dependency |

Both providers point at `127.0.0.1`-only URLs with `apiKey: "not-needed"` —
no secret material involved, since neither backend does auth.

Validating this config doesn't require a full NixOS rebuild: pointing a
scratch `$HOME` at the config directory and running `opencode debug
config` confirms it resolves with no schema errors, and `opencode models`
lists the resolved `provider/model` identifiers
(`local-gpu/qwen3.6-35b-a3b-gpu`, `local-npu/qwen2.5-coder-7b-npu`, etc.) —
a fast way to catch a typo or schema mismatch before touching the live
system.

## The `small_model` mutual-exclusivity bug

The first version of this config set `small_model` to
`local-npu/qwen2.5-coder-7b-npu` — the NPU tier, on the reasoning that
lightweight background tasks (like generating a session title) should use
the cheaper/smaller model rather than the big GPU one. This is a real,
easy mistake to make, and it's broken by construction:

`npu-server-coder` and `gpu-server-hard` declare `Conflicts=` on each other
at the systemd unit level (see
[docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) for why — they were
found to be unsafe to run concurrently on this machine's ~30GB RAM budget).
Starting either service automatically stops the other. That means: with
`model` defaulted to `local-gpu` and `small_model` defaulted to
`local-npu`, any session using the default GPU model would have every
`small_model` call — including the background title-generation request
issued automatically at session start — permanently fail, because the NPU
server can never be up at the same moment the GPU server is (which it is,
by definition, for the entire duration of a `local-gpu` session).

This was caught by direct review, not by observing the failure first —
worth calling out as the general shape of the bug: **any OpenCode field
that names a model from a different provider than your default should be
checked against your systemd `Conflicts=`/mutual-exclusivity graph before
assuming it'll be reachable.** A model that's merely "defined and
selectable" is fine to leave pointed at a conflicting tier; a model wired
as an *always-on background dependency* of your default session is not.

**Fix**: set `small_model` to the same value as `model`
(`local-gpu/qwen3.6-35b-a3b-gpu`). `local-npu` stays defined in the config
and fully selectable on its own (e.g. for a manual `-m
local-npu/qwen2.5-coder-7b-npu` run when you specifically want the NPU
tier and don't mind stopping the GPU server first) — it's just no longer
wired as a dependency that has to be alive in the background of a
GPU-default session.

If you only run one tier at all, the simpler fix is to delete the other
provider block entirely and drop `small_model` (or point it explicitly at
your one model) — see
[configs/opencode/README.md](../configs/opencode/README.md).

## The model-level `options` raw-passthrough discovery

This was investigated because of a legitimate follow-up concern after a
separate change: `gpu-server-hard`'s `ExecStart` was given `--reasoning
off` as its new default (see
[docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md)
for the benchmark that motivated defaulting thinking off). On the nvim
side, the `enable_thinking` schema toggle (see
[docs/05-nvim-integration.md](05-nvim-integration.md)) still lets you
override that per request. OpenCode had no equivalent — which raised the
question of whether OpenCode sessions would now *always* run
thinking-off, with no way to get real deliberation for planning/design work
where it genuinely helps.

### How this was actually verified: proxying OpenCode's real traffic

Rather than infer this from documentation or CLI `--help` text, OpenCode's
outgoing HTTP traffic was captured directly with a `socat` TCP relay
sitting between OpenCode and `gpu-server-hard` (i.e. OpenCode was pointed
at the relay's local port instead of 8901, and the relay forwarded to the
real server while a copy of the traffic was inspected). This is the same
proxy-based verification methodology used elsewhere in this project (see
[docs/07-benchmarks-and-methodology.md](07-benchmarks-and-methodology.md)
for the general "verify against the real wire format, don't guess" theme)
and it's what actually settled every claim in this section — these are
observed facts from a raw request body, not assumptions about how
`@ai-sdk/openai-compatible` or OpenCode's CLI flags "probably" work.

The capture showed, concretely:

- **OpenCode's `@ai-sdk/openai-compatible` provider never sends any
  reasoning-control field on its own** — not `chat_template_kwargs`, not
  `reasoning_effort`, not anything else — for either the main model or the
  `small_model` background call. The background title-generation request
  was independently confirmed to run its own full thinking deliberation
  over a one-line title, unnecessarily, before this was fixed.
- Setting arbitrary keys inside a model's `options` object in
  `opencode.json` — `extraBody`, `chat_template_kwargs`, `body`, and
  `reasoningEffort` were all tried at once, as a deliberate probe — showed
  **all four keys verbatim in the real outgoing HTTP request body**,
  including the one that actually matters: `chat_template_kwargs`. There
  is no field whitelist at this layer; whatever you put in a model's
  `options` object is merged directly into the request JSON sent to the
  backend.

### The fix this enabled: `qwen3.6-35b-a3b-gpu-thinking`

Given that a model's `options` is a genuine raw passthrough, a second,
explicitly-selectable model entry was added under the `local-gpu` provider
in [configs/opencode/opencode.json](../configs/opencode/opencode.json):
identical to the default model in every capability flag, but with

```json
"options": { "chat_template_kwargs": { "enable_thinking": true } }
```

This was validated end to end, not just configured and assumed to work:

1. `opencode debug config` / `opencode models` list it with no schema
   errors.
2. A live request against it produced real `reasoning_content` in the raw
   server response (re-confirmed via the same proxy capture) — proving
   thinking is genuinely being triggered server-side, not just a
   no-op config field.
3. `opencode run "..." -m local-gpu/qwen3.6-35b-a3b-gpu-thinking
   --thinking` genuinely shows the deliberation before the final answer in
   the terminal, matching the nvim experience.

Net effect: the default `local-gpu` model stays fast and thinking-off for
everyday quick tasks (consistent with the benchmark data in
[docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md)),
and switching to `local-gpu/qwen3.6-35b-a3b-gpu-thinking` (plus the
`--thinking` flag — see below) gives real, visible deliberation for
planning/design-style work, without needing any server-side change or a
second running instance of the model.

## `--thinking` is display-only; `--variant` is a no-op here

Two more findings confirmed by the same proxy-capture methodology, worth
stating precisely because both are easy to misread from the flag names
alone:

| Flag | What it actually does | What it does NOT do |
|---|---|---|
| `--thinking` | Controls whether OpenCode's terminal UI **prints** reasoning blocks it receives | Does not affect generation in any way — the model reasons or doesn't reason regardless of this flag; if the request itself doesn't ask for thinking (no `enable_thinking: true` reaching the server), there's nothing for `--thinking` to display |
| `--variant` | Selects among a model's **catalog-defined** reasoning-effort variants, for models that have any | A true no-op for a custom model like `local-gpu/qwen3.6-35b-a3b-gpu-thinking` — this repo defines it directly in `opencode.json` with no catalog entry and no declared variants, so there's nothing for `--variant` to select between |

Concretely: `opencode run "..." -m local-gpu/qwen3.6-35b-a3b-gpu-thinking`
**without** `--thinking` still sends the request that triggers real
server-side reasoning (because the model's `options.chat_template_kwargs`
passthrough is unconditional, not gated by the CLI flag) — you just won't
see the reasoning text in the terminal output. You need **both**: the
`-thinking` model variant to make the request actually ask the server to
reason, and the `--thinking` flag to make OpenCode print what comes back.
Also worth noting from the same investigation: `--format json` mode
doesn't currently surface reasoning as a distinct event and doesn't count
it in `tokens.reasoning` — only the default/plain output format renders it,
as `Thinking: ...` followed by the final answer.

The model's `reasoning: true` config field (set on all three model entries
in `opencode.json`) is a separate, unrelated thing again — it's a
capability flag advertised to OpenCode's UI/routing logic, not a switch
that turns reasoning on or off by itself.

## Applying and verifying

This config was validated without a full `nixos-rebuild switch` by
pointing a scratch `$HOME` at it and running `opencode debug config` /
`opencode models`, but a real end-to-end generation against `local-gpu`
does need the actual home-manager activation (`xdg.configFile` in
[configs/opencode/default.nix](../configs/opencode/default.nix) writes the
real config into place). Once activated, a real interactive session was
confirmed working end to end — correctly editing code, catching and
correcting its own errors, and building freely — closing out the last open
verification item for this integration. The `small_model` fix (title
generation no longer silently failing against a conflicting tier) should
also be spot-checked after activation if you're replicating this setup,
since it's the kind of failure that's easy to not notice (a failed
background call doesn't interrupt your foreground session).
