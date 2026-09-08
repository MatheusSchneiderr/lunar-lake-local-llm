# 9. gpu-guard (Optional, Not Deployed)

**Status: parked.** `gpu-guard` is a small C++ reverse proxy that was designed, built, and fully verified against this project's GPU tier — and then deliberately *not* wired into the running system. This chapter documents it in full because the code is real, tested, and sitting in the repo (`configs/gpu-guard/`), but it is **not part of this guide's own deployed setup**. If you clone this repo and follow docs 01–08 end to end, you will never start `gpu-guard`, and nothing in those chapters depends on it.

Read this chapter if you want to understand a real, verified mitigation for a specific local-model failure pattern — either because you're hitting that pattern yourself on a weaker model, or because you just want to see how a bounded-retry proxy like this is built in C++ against an OpenAI-compatible streaming API. Skip it if you only care about the setup that's actually running; docs 01–08 are self-contained without it.

## Why this got built

Two related failure symptoms surfaced during real interactive use of `gpu-server-hard` (logged in full in [docs/08-troubleshooting-and-incidents.md](08-troubleshooting-and-incidents.md), items 3 and its "related variant"), both under long tool-calling chains in codecompanion:

1. **Degenerate stall**: the model ends its turn with no `tool_calls` and essentially empty content, immediately after a tool result comes back. The conversation just goes silent — no error, no answer, nothing to react to.
2. **False refusal**: the model confidently (and wrongly) claims it has no file/tool access and asks the user to paste the file contents — despite tools genuinely being available and working (confirmed live: the identical prompt with a bigger token budget correctly called `file_search` three times). This reads as run-to-run sampling inconsistency in the model's tool-use decision-making under its tendency toward long, indecisive internal deliberation, not a systemic break.

This is not a new problem in this project. The NPU tier's `server.py` (see [docs/02-npu-tier-setup.md](02-npu-tier-setup.md)) already needed its own bespoke stall-detection-and-retry logic to handle the same failure class on that tier, and that logic had already proven itself in production use there. When the identical symptom showed up on `gpu-server-hard` — which runs plain `llama-server` (llama.cpp/Vulkan) with no custom Python layer in front of it — there was nothing equivalent protecting it. Prompting alone did not fix it: even codecompanion's own built-in agent-group system prompt (whose entire purpose is to tell the model to keep calling tools until the task is done) did not prevent either symptom from recurring.

That gap, plus a track record showing the retry-once pattern actually works, is what motivated building a GPU-tier equivalent rather than trying to prompt-engineer the problem away a second time.

### Why C++, not Python, and why a strict size target

This was an explicit requirement going in, not a stylistic preference discovered afterward: this proxy has to sit in front of **every single chat request** to the GPU tier and stay always-resident the whole time `gpu-server-hard` is up. The NPU tier's `server.py` can afford to be a heavier Python process because it *is* the model-serving process — it already pays the cost of loading and running a model pipeline, so a Python interpreter's overhead on top of that is noise. `gpu-guard` is different: it does not load a model, does not touch the GPU, and exists purely to forward and occasionally inspect text between a client and `gpu-server-hard`. Any overhead it adds is overhead the *entire system* pays on every request, forever, so it needed to be as close to free as possible.

The concrete target set for this was **under 5MB** for the compiled binary. The actual result, compiled locally with `g++ -std=c++17 -O2` against the fetched `httplib`/`nlohmann_json` headers (the same command the Nix build runs, see `configs/gpu-guard/default.nix`), came in at **~860KB unstripped** — comfortably inside that target, with headroom to spare. No model pipeline, no interpreter startup, no bytecode compilation step: just a static binary that opens a listening socket and moves bytes.

## Architecture

`gpu-guard` (`configs/gpu-guard/main.cpp`, ~450 lines) is an HTTP reverse proxy built on two header-only libraries already packaged in nixpkgs — `httplib` (`pkgs.httplib`) for the server/client HTTP plumbing, and `nlohmann::json` (`pkgs.nlohmann_json`) for JSON parsing. It listens on `127.0.0.1:8899` by default and forwards to `gpu-server-hard` on `127.0.0.1:8901` (both configurable via `GPU_GUARD_PORT`/`GPU_UPSTREAM_HOST`/`GPU_UPSTREAM_PORT` environment variables — see the `Environment` block in `configs/gpu-guard/default.nix`). Deliberately no TLS, no compression, no auth: it only ever talks to `127.0.0.1`, matching `gpu-server-hard`'s own posture (see [docs/03-gpu-tier-setup.md](03-gpu-tier-setup.md)).

The entire design pivots on one fact: only a small minority of requests actually need inspection. A request either carries a non-empty `tools` array or it doesn't, and that single check at the top of the `/v1/chat/completions` handler decides which of two completely different code paths a request takes.

### Path 1: requests with no tools — real streaming passthrough

The overwhelming majority of chat traffic (plain questions, non-agentic chat, anything without a `tools` array or with an empty one) gets **real byte-for-byte streaming passthrough** — no buffering, no added latency, live token-by-token SSE exactly as if the client were talking to `gpu-server-hard` directly.

This is implemented as a background-thread-plus-thread-safe-queue bridge (`StreamBridge` in `main.cpp`), because the two ends of the pipe run on different threads that don't naturally talk to each other:

- A detached background thread opens its own `httplib::Client` to the upstream and issues the request with a `content_receiver` callback. `httplib`'s streaming client calls this callback incrementally as SSE chunks arrive off the wire; the callback's only job is to push each chunk into `StreamBridge`'s internal `std::deque<std::string>` (guarded by a `std::mutex`) and notify a `std::condition_variable`.
- The server side of the same request is being driven by one of `httplib::Server`'s own worker threads, via `res.set_chunked_content_provider(...)`. Each time that provider is invoked, it calls `StreamBridge::next()`, which blocks on the same condition variable until either a chunk is available or the background thread has signaled completion (`finish()`), then writes whatever chunk it got straight to the client's `DataSink`.

The result is two independent threads — one pulling from `gpu-server-hard`, one pushing to the actual client — synchronized only through this small queue, with no point where a full response is ever assembled in memory. This is the mechanism that makes "no added latency" true rather than aspirational: the first byte gpu-guard receives from upstream is (modulo queue/mutex overhead measured in microseconds) the same byte it hands to the client.

### Path 2: tool-bearing requests — buffered detect-and-retry-once

Requests that do carry a non-empty `tools` array take a fundamentally different path, implemented in `handle_tools_request()`. Regardless of what the client asked for, gpu-guard forces `stream: false` against the upstream (and strips `stream_options`) for these requests — the same tradeoff the NPU tier's `ToolCallScanner` already makes in `server.py`, for the identical reason: detecting a degenerate stall or a false refusal requires looking at the *complete* final message (its `tool_calls` array, its full `content` string), and there is no way to make that decision from a partial stream of tokens. If the client itself asked for `stream: true`, gpu-guard reconstructs a synthesized SSE stream from the single complete response it already has once it's done deciding — the client still gets an SSE response shape, it just isn't token-by-token for this request type. Both tool-calling and non-tool-calling requests were confirmed live against a real `file_search` call to work correctly end to end, in both streaming and non-streaming client modes.

The buffered flow is:

1. Send the full request upstream (once), non-streaming.
2. Run the response's message through two detection functions (below).
3. If either fires, build one retry request — the original message list with a nudge message appended, `stream` forced to `false` — and send it upstream exactly once more.
4. Use whatever came back, retried or not. If the retry call itself fails to come back (upstream error), fall through and use the original (possibly stalled/refusing) response rather than leave the client hanging — never more than one retry, matching the NPU tier's bounded design.

### The two detection functions

Both are ported directly from `npu-server/server.py`'s already-proven logic, not reinvented from scratch:

- **`is_degenerate_stall(message, request_messages)`**: returns `false` immediately if the message has a non-empty `tool_calls` array (the model did act — this check only cares about turns where it produced neither a tool call nor real content). Otherwise it trims the message's `content` for whitespace; if anything non-whitespace remains, it's real content and this returns `false`. Only when content is empty or whitespace-only does it check the *request's* last message role — if the last message the model was replying to was itself a `role: "tool"` message (i.e., the model just received a tool result and then said nothing), this returns `true`.
- **`is_false_refusal(message)`**: also returns `false` immediately on a non-empty `tool_calls` array — the model calling a tool is proof it isn't refusing, even if its accompanying text happens to match a refusal phrase. Otherwise it runs the message's `content` against a small fixed list of `std::regex` patterns (case-insensitive), such as `don't have (direct )?access to`, `please paste (the|your)`, `i (can't|cannot) access`, and a few close variants. This list is deliberately not meant to be exhaustive — it's built to be easy to extend once more real refusal phrasings are observed in the wild, and the code comment above it says so explicitly.

Both functions operate purely on the OpenAI-compatible wire format (message roles, `content`, `tool_calls`) — nothing about them is specific to `Qwen3.6-35B-A3B` or to this project's exact chat template, which is what makes gpu-guard reusable against any `llama-server`-backed GPU tier without modification.

### The 13-assertion selftest

`gpu-guard --selftest` runs **exactly 13 deterministic assertions** against fabricated inputs for `is_degenerate_stall`, `is_false_refusal`, and the retry-body builder (`build_retry_body`). This exists because the live model's actual failure modes cannot be reliably triggered on demand for an integration test — you cannot write a test that reproduces sampling-dependent model misbehavior deterministically, so the only correctness check available is a direct unit-level check of the detection logic itself against hand-built message shapes covering the boundary cases (empty content after a tool result, whitespace-only content, real content, a message with `tool_calls` present regardless of content, a refusal phrase paired with actual tool calls, and so on — see the full list in `configs/gpu-guard/main.cpp`'s `run_selftest()`).

This selftest is wired directly into the Nix package's `checkPhase` (`configs/gpu-guard/default.nix`): `doCheck = true` and `checkPhase = "./gpu-guard --selftest"`, so a regression in the detection logic fails the Nix build itself, not just a separate CI step someone has to remember to run. All 13 assertions were confirmed passing both from a local ad hoc compile (fast iteration outside the full Nix rebuild loop) and from the final `nix build`-produced package binary, including the `checkPhase` run as part of that build.

### Nix packaging and the systemd unit

`configs/gpu-guard/default.nix` builds the binary via a plain `stdenv.mkDerivation` — no build system beyond a direct `g++` invocation, since this is a single translation unit (`main.cpp` in, `gpu-guard` binary out; see the derivation's `buildPhase`). It also defines a `systemd.user.services.gpu-guard` unit: listens on port 8899, forwards to `gpu-server-hard` on port 8901, and declares both `Unit.Requires` and `Unit.After` on `gpu-server-hard.service`. That dependency declaration is the whole point of fronting the real backend with this proxy at all — starting `gpu-guard` alone is enough to bring the real `gpu-server-hard` up underneath it, so in a deployment that uses gpu-guard, it becomes the *only* service nvim, OpenCode, or a `gpu-start` shell alias needs to touch; nothing else needs to know port 8901 exists. Like `gpu-server-hard` itself, the unit has no `Install.WantedBy` — it's meant to be started on demand (e.g. from an nvim keymap), not at login.

## Verification performed

This was not shipped on faith. Beyond the 13-assertion selftest itself, it was live-tested against the real, running `gpu-server-hard`:

- `/health` and `/v1/models` passthrough (the two plain `GET` routes, proxied transparently via `proxy_transparent()`) both work.
- Non-tool streaming passthrough produces identical real-time SSE output to talking to `gpu-server-hard` directly — confirmed by direct comparison, not just "it returns something."
- Tool-bearing requests, in both streaming and non-streaming client modes, correctly pass through a real successful `file_search` tool call unchanged when no stall or refusal occurs.
- The final Nix-built package binary (`nix build`, not just the local `g++` compile used for fast iteration) was checked directly, including its `checkPhase` selftest run as part of that build — so the artifact that would actually get deployed was the one verified, not a stand-in.

One additional retest is worth calling out because it's a **non-finding**, and the plan behind this guide flagged it *before* running the test rather than treating it as a surprise afterward: re-running the two prompts that spiraled hardest in the thinking-mode benchmark (the Fibonacci trace and the list-comprehension task — see [docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md)) forced back through gpu-guard with thinking re-enabled per-request, both still ran out their full 1536-token budget with no answer, on both seeds, completely unchanged from the original benchmark. This is expected, not a gap in gpu-guard: those two prompts carry no `tools` array at all, so gpu-guard is pure streaming passthrough for them by construction — it only ever inspects requests that actually carry a non-empty `tools` array, because that's the literal scope of the bug it targets. A no-tools stall is simply outside what this proxy was ever built to catch.

## Why it was parked rather than deployed

After building and fully verifying gpu-guard, the decision was to **not** wire it into the live system — not because it doesn't work, but because a fresh look at the timeline made the underlying bug's continued relevance doubtful:

- The original silent-stall and false-refusal observations were made against noticeably weaker/smaller models and configurations earlier in this project.
- By the time gpu-guard was finished, `gpu-server-hard` had already moved on to two changes that plausibly remove most of what was causing the original stalls in the first place: the much larger `Qwen3.6-35B-A3B` model, and `--reasoning off` as the server's default (see [docs/04-thinking-mode-and-preservation.md](04-thinking-mode-and-preservation.md) — that same investigation found thinking-mode-on itself produced the identical failure shape, an unclosed reasoning block eating the entire token budget with no answer, on a plain single-turn prompt with no tools at all).

Put together: the bug class this proxy targets probably isn't a live problem anymore at the model and configuration this guide's GPU tier actually ships, so adding an always-resident proxy in front of every single request wasn't judged worth the extra moving part — one more service to keep running, one more thing that can fail or need a restart, for a failure mode that may no longer occur in practice. The wiring that would have activated it was reverted in full: `nvim-config.nix`'s adapter URL and its `GpuStartAndOpen` helper, `opencode.json`'s `baseURL`, and the `gpu-start`/`model-stop` shell aliases were all pointed back at `gpu-server-hard`'s port (8901) directly, and gpu-guard's own home-manager import was removed. No systemd service is defined in the deployed configuration described by docs 01–08, and nothing in that configuration points at port 8899.

The code was kept in the repo rather than deleted specifically so it doesn't have to be rebuilt from scratch if the pattern resurfaces — see `configs/gpu-guard/README.md` for the same rationale from the config side.

**Be clear-eyed about what this is and isn't.** This is a "built, verified, not needed here, might be needed by you" component — not a required part of this stack, and not something the rest of this guide assumes you have running. Don't treat its existence in this repo as a signal that the deployed setup is somehow incomplete without it; docs 01–08 describe a system that was tested end to end with gpu-guard absent.

## When you might actually want to deploy it

Two situations where reactivating gpu-guard is a reasonable choice, not just a hypothetical:

| Situation | Why gpu-guard helps |
|---|---|
| You're running a smaller/weaker model than this guide's `Qwen3.6-35B-A3B` | The original stall/refusal pattern was observed on weaker models in this exact project's history — if you're on a lighter model for memory or speed reasons, you're closer to the conditions that produced it |
| You leave thinking mode on (`--reasoning auto`/`on`, or per-request `enable_thinking: true`) rather than this guide's `--reasoning off` default | This project's own benchmark found thinking-mode-on reproduces the same "runs out the budget with no answer" shape even outside tool-calling — you're keeping the ingredient most linked to the failure |
| You see the silent-stall or false-refusal pattern in real tool-calling usage, on any model | This is the direct, concrete symptom gpu-guard exists to catch — if you're seeing it, this is a ready-made fix, not something you'd need to build yourself |

If none of those apply — you're on this guide's exact model and config, with thinking off by default — there's no evidence from this project that you need it, and adding it would be adding an unnecessary moving part for a bug that likely isn't occurring.

### How to actually turn it on

The steps are the same ones spelled out in `configs/gpu-guard/README.md` (kept there so the reactivation instructions travel with the code, independent of this doc):

1. Add `configs/gpu-guard` to your home-manager `imports`.
2. Point your nvim codecompanion adapter's URL and/or your OpenCode provider's `baseURL` at `http://127.0.0.1:8899` instead of your GPU tier's real port (`8901` in this guide's numbering) directly. Because gpu-guard's systemd unit declares `Requires=`/`After=` on the real backend service, starting `gpu-guard` alone brings that backend up underneath it — you don't need to separately manage both services.
3. `systemctl --user start gpu-guard`.

The code is model-agnostic by construction — it only ever inspects the OpenAI-compatible wire format (`tools`, `tool_calls`, `content`, message roles), never anything tied to a specific model's tokens or template — so it should work unmodified against any `llama-server`-backed GPU tier, not just the one this guide builds. If you hit a genuinely *new* failure pattern rather than one of the two this proxy already catches, the intended path is to extend `is_degenerate_stall`/`is_false_refusal` in `configs/gpu-guard/main.cpp` (the refusal-phrasing regex list in particular was written to be easy to grow) rather than starting a replacement from scratch — and remember to add selftest assertions for whatever you add, since that's the only correctness net this component has.

## What was tested and what wasn't

To be precise about the actual boundaries of verification here, since this guide commits to not overselling anything untested (see [configs/shared/versions.md](../configs/shared/versions.md) for the point-in-time stack this all reflects):

- **Tested**: the 13 selftest assertions against fabricated inputs; live passthrough (streaming and non-streaming, tools and no-tools) against a real running `gpu-server-hard`; a real successful `file_search` tool call passing through unchanged; the final `nix build`-produced binary including its `checkPhase`.
- **Not tested**: gpu-guard has never been exercised against a real, naturally-occurring stall or false-refusal from actual usage — because by the time it was ready to test that way, the model/config change (`Qwen3.6-35B-A3B` plus `--reasoning off`) had already plausibly reduced how often those symptoms occur, and the parking decision was made before extended real-world exposure could confirm one way or the other. If you deploy this yourself and it catches (or fails to catch) a real occurrence, that's genuinely new information this project doesn't have.
