# loop-guard

A reasoning-loop-detecting reverse proxy that sits in front of the GPU-tier
`llama-server` ([`../gpu-tier-sycl/`](../gpu-tier-sycl/)). It catches a real
failure mode this guide's model has hit in production: getting stuck
restating the same dead-end hypothesis in different words inside its own
`<think>` block, never converging. Full explanation, including why a fixed
reasoning-token budget can't fix this and how the detection actually works:
[docs/14-loop-guard-reasoning-loop-proxy.md](../../docs/14-loop-guard-reasoning-loop-proxy.md).

loop-guard is a separate, standalone tool (not tied to this hardware/guide
the way `llama-cpp-sycl` is), so it lives in its own repo rather than being
vendored here:
[github.com/MatheusSchneiderr/loop-guard](https://github.com/MatheusSchneiderr/loop-guard) -
read that repo's README and `src/tracker.rs` for the full detection
mechanism and rationale.

## Files

- `default.nix` — a NixOS/home-manager module (`services.loop-guard.enable`)
  that builds loop-guard from source (`buildRustPackage`, pinned to a
  commit) and wires it up as a systemd user service: listens on `8901`,
  forwards to `127.0.0.1:8902` (where the GPU-tier server should listen -
  see [`../gpu-tier-sycl/default.nix`](../gpu-tier-sycl/default.nix)),
  `Requires=`/`After=` so starting loop-guard also starts the backend.
- `loop-guard-Cargo.lock` — copy of the source repo's lockfile, needed for
  `cargoLock.lockFile` to resolve during the build.

Point whatever you currently point at the GPU-tier server's port (editor
keymap, shell alias, OpenCode config) at `8901` instead, and point the
GPU-tier server itself at `8902` so loop-guard can front it.

Pinned commit is in [`../shared/versions.md`](../shared/versions.md).
