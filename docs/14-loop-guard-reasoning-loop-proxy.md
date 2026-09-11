# loop-guard: Catching a Reasoning Loop Without a Budget

[Chapter 13](13-qwen36-sycl-fine-tuning-2026-09-11.md) left Qwen3.6-35B-A3B
tuned and confirmed as the daily driver. Shortly after, it did something
the earlier fine-tuning pass hadn't accounted for: mid real agentic-coding
task, in Plan mode, it got stuck inside its own `<think>` block — not
hanging, not repeating literally, but restating the same dead-end
hypothesis about an F#-format-string bug in different words, over and
over, never converging on an answer. It only stopped after a manual
interrupt. This chapter covers the fix that shipped for it:
[loop-guard](https://github.com/MatheusSchneiderr/loop-guard), a small
reverse proxy that detects this specific failure mode by content, not by
length or time, and interrupts it automatically.

## 1. Why a token budget or timeout doesn't work here

The obvious first fix is a hard cap: limit reasoning tokens, or time out
after N seconds. Both were rejected before writing any code, for the same
reason: **a legitimately hard task and a stuck loop both need "however
long it takes."** A budget that's generous enough to let a hard problem
finish thinking is also generous enough to let a stuck loop run the same
distance before getting cut off — and a budget tight enough to catch the
loop early would also cut off real, productive reasoning on a hard task.
Budgets aren't dynamic to context; they can't tell "still making progress"
apart from "still restating the same idea."

The failure mode itself compounds this: it's not literal token-for-token
repetition (which a naive dedup check would catch instantly) — it's
**paraphrastic self-repetition**, a documented, model-family-wide
Qwen3/3.6 issue where semantically reworded restatement of the same
conclusion suppresses the model's own probability of emitting the
`</think>` closing tag. Whatever catches this has to look at *meaning*,
not token count or wall-clock time.

## 2. Detection: content-triggered, not length-triggered

loop-guard segments the model's streaming reasoning text into "steps" at
its own restart markers — words like "wait", "actually", "hold on", "hmm"
that the model itself uses to mark a new attempt at the problem. Each new
step is compared against every earlier step in the same trace using a
hand-rolled bag-of-words cosine similarity, with in-session/online IDF
reweighting so common connector words don't dominate the score. When a
new step is similar enough to a prior one, that's the loop signal —
regardless of how many tokens it took to get there. Full implementation:
[`src/tracker.rs`](https://github.com/MatheusSchneiderr/loop-guard/blob/main/src/tracker.rs)
in the source repo.

Once detected, the intervention is "budget forcing" (the technique from
the s1 reasoning-scaling paper, also used by vLLM's
`ThinkingTokenBudgetLogitsProcessor`): rather than truncating the response
outright (which returns an empty answer), it injects the model's own
closing tag plus a small nudge, then lets generation continue for just the
final answer.

## 3. Why Rust

This sits in the hot path of a daily-driver coding assistant, and the
proxy logic is AI-authored — memory safety mattered more than it would for
a one-off script. A first C++ prototype
([`main.cpp`](https://github.com/MatheusSchneiderr/loop-guard) kept in the
source repo for comparison, not deployed) had a real, hand-rolled
slot-cache-targeting optimization (matching a continuation request to the
right `llama-server` KV-cache slot via `/slots`) that was racy and was
confirmed broken live, twice, before being rewritten. The Rust version
queries `/slots` once, at the exact moment of detection, instead of
diffing a before/after snapshot.

## 4. Production wiring

loop-guard runs as a reverse proxy in front of the GPU-tier `llama-server`:
it listens on `8901` (the port editors/tools should point at) and forwards
to `127.0.0.1:8902` (where the GPU-tier server itself now listens). The
systemd unit declares `Requires=`/`After=` on the backend service, so a
single `systemctl --user start loop-guard` brings both up. Config:
[`../configs/loop-guard/`](../configs/loop-guard/).

It's built from source (`buildRustPackage`, pinned to a commit) rather
than fetched as a prebuilt binary — a Nix-machine-built ELF binary
legitimately contains its own `/nix/store` paths (its dynamic linker
interpreter path), which a plain binary fetch's fixed-output-derivation
purity check rejects outright. Building from source avoids that class of
problem entirely and is simpler besides.

## 5. Q&A

**Does this replace careful prompting or a better system prompt?** No —
it's a safety net for a failure mode that's a property of the model
family under certain conditions, not something a system prompt reliably
prevents. It's meant to catch the rare case, not to be relied on routinely.

**Could this false-positive on a long, legitimately winding reasoning
trace?** It's possible in principle — the whole point of a content-based
check is to tolerate a task that genuinely needs a lot of distinct
reasoning steps, since each step only needs to be *dissimilar enough* from
prior ones, not short. The similarity threshold was tuned against a real
captured transcript of the stuck loop plus known-good, genuinely-distinct
traces, but it hasn't been stress-tested against every possible long,
legitimate reasoning shape.

**Does this need revisiting if the model changes?** The restart-marker
segmentation is tuned to how Qwen3.6 marks its own restarts in English -
a different model or a different reasoning style might need different
markers or a re-tuned similarity threshold.

## Status at time of writing

Deployed to production, fronting the GPU-tier server on every real
request. It has caught at least one genuine, reproduced production loop
live.
