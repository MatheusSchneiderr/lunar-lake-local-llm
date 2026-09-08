# Neovim / codecompanion.nvim Integration

This chapter covers wiring [codecompanion.nvim](https://github.com/olimorris/codecompanion.nvim)
to the GPU tier (`gpu-server-hard`, see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md))
as a local, OpenAI-compatible chat/inline-edit assistant inside Neovim. The
full adapter is in [configs/nvim/nvim-config.nix](../configs/nvim/nvim-config.nix)
(an excerpt from a real [nvf](https://github.com/NotAShelf/nvf) configuration);
this doc explains *why* every non-obvious piece of it exists, because each one
was a real bug or a real gap discovered while building this, not a stylistic
choice.

Only the GPU tier is wired into nvim. The NPU tier's `npu-server-coder`
systemd unit still exists (see [docs/02-npu-tier-setup.md](02-npu-tier-setup.md))
but isn't given its own codecompanion adapter — the GPU model's throughput
made a fallback-to-NPU adapter unnecessary in day-to-day editor use. If you
want both tiers reachable from nvim, add a second `http` adapter following
the same shape as `gpu_hard` below, pointed at port 8900 instead of 8901.

## Prerequisites

- codecompanion.nvim recent enough to have the `openai_compatible` base
  adapter, and `display.chat.token_count` / `display.chat.show_settings`.
  This guide was built and tested against the commit pinned in
  [configs/shared/versions.md](../configs/shared/versions.md) — the
  old-vs-new handler format section below is genuinely commit-sensitive, so
  check `lua/codecompanion/adapters/http/init.lua` in your installed
  version if anything here doesn't match.
- `gpu-server-hard` running and reachable at `http://127.0.0.1:8901` (or
  whatever URL you set in `env.url`).

## The `gpu_hard` adapter

codecompanion organizes adapters by transport type since the version this
guide targets — `adapters.http.*` / `adapters.acp.*`, not a flat
`adapters.*` table as in older versions. The adapter itself is built with
`require("codecompanion.adapters").extend("openai_compatible", { ... })` —
extending the generic OpenAI-compatible base rather than writing one from
scratch, since both `gpu-server-hard` (llama.cpp's `llama-server`) and
`npu-server-coder` already speak `/v1/chat/completions` in the standard
shape.

Two setup details worth calling out explicitly because they're easy to get
wrong and fail in confusing ways:

- **`env.api_key` must be a non-empty string**, even though the local
  server has no auth. codecompanion's `get_schema()` splits this string on
  `"."` to walk a schema path; an empty string produces zero path segments,
  so the lookup loop never executes and the function degenerately returns
  the *entire adapter table* instead of `nil`. That table then gets
  substituted into a request string later, which crashes with `invalid
  replacement value (a table)`. The fix is trivial — any non-empty
  placeholder string (`"not-needed"` in this config) — but the failure mode
  gives no hint that the api_key field is the cause.
- **The model schema needs a `choices` entry with `meta.context_window`
  set**, not just a `default` model name string. This isn't for display
  purposes on its own — it's what the token-count percentage feature (below)
  reads. See `configs/nvim/nvim-config.nix`'s `schema.model.choices` block.

### `enable_thinking`: a per-chat toggle backed by real benchmark data

The adapter exposes an `enable_thinking` boolean in its schema, defaulted
to `false`. This is not a cosmetic setting — it's wired to
`mapping = "body.chat_template_kwargs"`, which merges straight into the
top-level outgoing request JSON via codecompanion's own
`Client.merge_body` (the same mechanism the built-in `gemini.lua` adapter
uses for its `thinkingLevel` field). On the wire, toggling it on sends
`{"chat_template_kwargs": {"enable_thinking": true}}` in the request body,
which `llama-server`'s patched chat template reads directly.

The default of `false` is not arbitrary. A real benchmark (5
verifiable-answer prompts, 2–3 fixed seeds each, 1536-token cap, direct
against the live endpoint) found thinking-mode-on failed to converge on 2
of 5 prompt types entirely — running out the full token budget spiraling
in reasoning without ever producing an answer — while thinking-off got
11/11 correct, using a fraction of the tokens in easy cases. Generation
throughput itself barely moved (~39 vs ~41 tok/s); the real cost of
thinking mode is wall-clock time from extra tokens before an answer, not a
per-token speed penalty. Full numbers and methodology are in
[docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md) —
this doc only asserts the conclusion the adapter default encodes.

Because `display.chat.show_settings = true` is set (explained below), this
toggle is directly editable per chat buffer: flip `enable_thinking: false`
to `true` in the settings YAML block before submitting a message, and only
that turn is affected. A practical rule of thumb that came out of using
this day to day: thinking-under-a-tool-chain was never actually
benchmarked (only plain single-turn prompts were), and tool-calling chains
are exactly where this model's degenerate-stall / false-tool-refusal
failure mode shows up (see
[docs/08-troubleshooting-and-incidents.md](08-troubleshooting-and-incidents.md)).
So the safer workflow for anything involving tools is: leave thinking off,
let the model gather information and draft a result, then — only if the
result looks shallow or misses something — flip `enable_thinking` on for
one follow-up message asking it to double-check its own draft.

## The OLD-vs-NEW handler format gotcha (reasoning extraction)

This is the single most subtle piece of the adapter, and worth explaining
precisely rather than just pointing at the code.

codecompanion's chat UI already renders model reasoning by default
(`show_reasoning = true` is the built-in default) — the UI side of "visible
thinking" needed no work. The gap was entirely on the adapter side: nothing
in a generic `openai_compatible`-based adapter extracts `reasoning_content`
out of the raw HTTP response and puts it somewhere the UI looks.

codecompanion has gone through a handler-format transition, and which
format an adapter uses determines *which hook name* you register the
reasoning-extraction function under. Reading
`adapters/http/init.lua`'s `get_handler()` and `uses_new_handlers()`
directly (rather than assuming) is what resolved this:

- `uses_new_handlers()` checks whether the adapter's `handlers` table has
  the *new*, nested shape — specifically whether it defines
  `handlers.lifecycle`, `handlers.request`, or `handlers.response` as
  sub-tables. If so, codecompanion treats it as a **new-format** adapter,
  and reasoning-extraction is a `handlers.response.parse_meta` function.
  This is the shape the built-in `deepseek.lua` adapter uses.
- If none of those nested tables are present, codecompanion falls back to
  the **old, flat handler format** — a single `handlers` table with
  top-level function keys like `chat_output`, `tools`, `parse_message_meta`,
  etc., no nested `lifecycle`/`request`/`response` grouping. `get_handler()`
  resolves the "extract metadata from a chunk" hook by name here as
  `handlers.parse_message_meta`, not `handlers.response.parse_meta`.

The `gpu_hard` adapter here is built by extending `openai_compatible`,
which is an old-flat-format adapter — it has no `handlers.lifecycle` /
`handlers.request` / `handlers.response` sub-tables. So even though the
*intent* ("pull `reasoning_content` out of the response and expose it as
model reasoning") is identical to what `deepseek.lua` does, the hook must
be registered as **`handlers.parse_message_meta`**, not
`handlers.response.parse_meta`. Registering it under the new-style name on
an old-format adapter silently does nothing — codecompanion never calls a
`response.parse_meta` function it doesn't know to look for on an adapter it
has classified as old-format, and there's no error, just reasoning that
never renders.

The actual extraction logic mirrors `deepseek.lua`'s exactly, only the
registration point differs:

```lua
parse_message_meta = function(self, data)
  local reasoning_content = data.extra and data.extra.reasoning_content
  if reasoning_content then
    data.output.reasoning = { content = reasoning_content }
    if data.output.content == "" then
      data.output.content = nil
    end
  end
  return data
end,
```

`llama-server` already sends `reasoning_content` as a non-standard
top-level field on each streamed chunk; `openai_compatible`'s own generic
`find_extra_fields()` already collects unrecognized fields like this into
`data.extra` for you — the only missing piece was moving
`data.extra.reasoning_content` into `data.output.reasoning`, which is the
field codecompanion's chat renderer actually looks at to draw the
"Thinking..." block.

**If you're on a different codecompanion commit**, re-check this before
assuming the flat-format hook name is still correct — this is exactly the
kind of implementation detail that a handler-format refactor would change
without necessarily changing the observable adapter API. See
[configs/shared/versions.md](../configs/shared/versions.md) for the exact
commit this was verified against.

## Token-count / context-window visual feedback

Two related but separate pieces of visual feedback were added, both
validated by building the config and grepping the generated
`nvf-init.lua` for the rendered functions (not yet exercised in a long
live session at time of writing — treat as validated-by-construction, not
battle-tested over hours of real use).

### Context-window-usage percentage

`display.chat.token_count` is a user-overridable formatter function that
codecompanion already calls with two arguments: the live token count for
the current chat, and the full adapter object. The override here computes
`tokens / context_window * 100` using codecompanion's own
`adapters.shared.context_window(adapter)` helper, which internally reads
`schema.model.choices[model].meta.context_window` off the adapter — which
is exactly why the `gpu_hard` adapter's schema needs that `choices` entry
populated (see above). The value is hardcoded to `32768` in the adapter
config and must be kept in sync by hand with whatever `-c` value
`gpu-server-hard`'s `ExecStart` actually uses (see
[docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) for why 32768 specifically
was chosen and why larger values were tested and rejected). There's no
mechanism that reads the server's actual configured context size at
runtime — if you change `-c` on the server, you must also update this
number in `nvim-config.nix`, or the percentage will silently be wrong.

The formatter falls back to a plain `"(N tokens)"` string if
`context_window()` isn't available or returns 0, so it degrades gracefully
on an adapter that hasn't set up the `choices`/`meta` schema.

### Visible thinking

Handled entirely by the handler-format fix above — once
`data.output.reasoning` is populated correctly, codecompanion's own
default `show_reasoning = true` behavior renders it in the chat buffer
with no further configuration needed.

### Why `display.chat.show_settings = true` is required

This is easy to miss and worth stating plainly: **codecompanion hides the
per-chat settings block by default, and there is no per-chat keymap to
reveal it** — only this one global `display.chat.show_settings` option.
Without setting it to `true`, the `enable_thinking` schema field (and the
model `choices` selector, and any other schema field you add later) exists
in the adapter but is never rendered anywhere in the UI, and there is no
command or keybinding to surface it on demand. It has to be turned on
globally, ahead of time, in setup.

With it enabled, every chat buffer gets a YAML-like settings block at the
top showing the adapter's editable schema fields (including
`enable_thinking: false`). Edit that block directly, in the buffer, before
submitting your message — the edited value applies to that turn's request
only, it's not a persistent config change.

## `GpuStartAndOpen`: start-and-health-check-then-open

Because `gpu-server-hard` is deliberately **not** kept always-on (no
`WantedBy=` in its systemd unit — see
[docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md) for the mutual-exclusivity
rationale with the NPU tier), opening a codecompanion chat needs to first
make sure the backend is actually up. Cold-loading a ~21GB Q4_K_M GGUF onto
the iGPU over Vulkan takes roughly 10–20 seconds — long enough that just
opening an empty chat buffer immediately and hoping the first request
lands after the server's ready is a bad user experience (the first message
would either hang or fail outright against a not-yet-listening port).

`luaConfigRC.gpuServerWaitHelper` in
[configs/nvim/nvim-config.nix](../configs/nvim/nvim-config.nix) defines a
single global helper, `GpuStartAndOpen(open_fn)`, that:

1. Fires `systemctl --user start gpu-server-hard` via `vim.fn.jobstart`
   with `detach = true` — non-blocking, doesn't stall the editor while the
   model loads.
2. Immediately issues a 1-second-timeout `curl` against
   `http://127.0.0.1:8901/health` via `vim.system` (also non-blocking,
   async callback).
3. If the health check returns HTTP `200` within that 1 second window,
   calls `open_fn()` — the actual chat/inline-edit command the keymap
   wanted to run.
4. Otherwise, notifies via `vim.notify` (WARN level) that the server isn't
   up yet and to try again shortly, rather than opening a chat buffer that
   would immediately fail.

This is intentionally a **cold-start nudge, not a real wait-loop** — it
does one fast health probe and either proceeds or tells you to retry, it
does not poll repeatedly or block Neovim for the full 10–20s load time.
In practice this means: on a cold start, the first keymap press starts the
service and (almost always) tells you to try again a few seconds later;
pressing the same keymap again once the model has finished loading opens
the chat immediately, since the systemd `start` command is a no-op against
an already-running unit and the health check now succeeds on the first try.

All four keymaps that open codecompanion route through this helper:

| Keymap | Mode | Command | Purpose |
|---|---|---|---|
| `<leader>ac` | normal | `CodeCompanionChat Toggle` | Toggle the chat panel |
| `<leader>ac` | visual | `'<,'>CodeCompanionChat Add` | Add the visual selection to chat |
| `<leader>ai` | normal | `CodeCompanion` | Inline Assistant at cursor |
| `<leader>ai` | visual | `'<,'>CodeCompanion` | Inline Assistant on selection |

The bare `CodeCompanion` command (as opposed to `CodeCompanionChat`) is
codecompanion's Inline Assistant — it applies the model's response
directly to the buffer as a diff you accept (`ga`) or reject (`gr`) via
plain text completion, no tool-calling involved, distinct from the
chat-panel workflow the `ac` keymaps drive.

### A leftover autocmd bug this pattern exposed

Worth mentioning here since it directly involves this same wiring: an
older, NPU-era autocmd (`luaConfigRC.npuServerAutostart`) had been wired to
codecompanion's generic `CodeCompanionRequestStarted` event to
unconditionally start `npu-server-coder` on *every* message, regardless of
which adapter was actually in use. Because `npu-server-coder` and
`gpu-server-hard` declare `Conflicts=` on each other at the systemd level
(see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md)), this leftover
autocmd silently killed `gpu-server-hard` on every single chat message
sent — a real, live-breaking bug, not a hypothetical one. It was missed
during the initial "remove NPU wiring from nvim" pass because it lived in
a separate `luaConfigRC` block from the adapter definition. The fix was to
delete the autocmd entirely. If you're adapting this config and still keep
an NPU-tier adapter around, make sure nothing analogous auto-starts a
`Conflicts=`-declared service on a generic request-lifecycle event —
scope any such autostart logic to the specific adapter it belongs to, not
a global event.

## Not yet verified

At the time of writing, the context-window-percentage display and the
visible-thinking rendering were validated by building the Nix config and
grepping the generated Lua for the expected functions — they had not yet
been exercised in a long, real, multi-hour nvim session. If you hit a
rendering issue that doesn't match this description, check first whether
your codecompanion version's chat-rendering internals differ from the
commit pinned in [configs/shared/versions.md](../configs/shared/versions.md).
