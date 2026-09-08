# gpu-guard (parked, not wired in)

See `docs/09-gpu-guard-optional.md` for the full write-up. This component
is **not currently deployed** in this guide's own setup - it's included
because it might be useful to someone on a weaker model where the failure
pattern it targets is still real.

A thin C++ reverse proxy for a `llama-server`-backed GPU tier (see
`main.cpp`) that catches two specific failure patterns seen under long
tool-calling chains with early, weak local models:

1. **Silent stall**: the model ends a turn with no tool call and no
   content at all, right after a tool result comes back.
2. **False refusal**: the model confidently (and wrongly) claims it has no
   file/tool access and asks the user to paste something, despite tools
   being available.

Both are detected only on requests that actually carry a non-empty `tools`
array - everything else gets pure byte-for-byte streaming passthrough, no
buffering, no added latency. On a match, it retries **once** with a nudge
message appended, then uses whatever comes back either way. This mirrors
the retry-once logic already proven out in `configs/npu-tier/npu-server/server.py`,
ported to C++ since this needs to be an always-resident process with
negligible overhead (a few hundred KB RSS, no model pipeline) rather than
a heavy Python service.

## Why it's parked, not active

Built and fully verified (13 self-test assertions in `main.cpp`, run via
`gpu-guard --selftest` and wired into the Nix package's `checkPhase`; also
live-tested against a real `llama-server` GPU tier - passthrough,
streaming, and tool-calling all confirmed working). But the stall/false-
refusal pattern it targets was observed against noticeably weaker/smaller
models and configurations than the one this guide's GPU tier ended up on -
by the time this was built, the GPU tier had already moved to a much
larger model (`Qwen3.6-35B-A3B`) with `--reasoning off` as the default
(see `docs/04-thinking-mode-and-preservation.md`), which on its own likely
already removes most of what was causing the original stalls. So it's
saved here as source, not wired into anything.

## When you might want it

If you're running a smaller/weaker model than this guide's `Qwen3.6-35B-A3B`
and see the silent-stall or false-refusal pattern in real tool-calling
usage, this is a ready-made fix:

1. Add this directory to your home-manager `imports`.
2. Point your nvim adapter and/or OpenCode provider `baseURL` at
   `http://127.0.0.1:8899` instead of your GPU tier's real port (e.g.
   `8901`) directly - `gpu-guard`'s systemd unit `Requires=`/`After=` your
   real backend service, so starting `gpu-guard` alone brings it up
   underneath it.
3. `systemctl --user start gpu-guard`.

The code itself is model-agnostic (it only inspects the OpenAI-compatible
wire format, not anything model-specific) - it should work unmodified
against any `llama-server` GPU tier. If you hit a *new* failure pattern
instead of these two, extend `is_degenerate_stall`/`is_false_refusal` in
`main.cpp` (the refusal-phrasing regex list in particular is meant to be
easy to grow) rather than starting over.
