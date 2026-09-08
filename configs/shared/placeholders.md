# Placeholder conventions used across this repo

Every config file in `configs/` has been genericized from a real, working
personal setup. Wherever you see one of these tokens, replace it with your
own value before using the file:

| Placeholder | Replace with | Appears in |
|---|---|---|
| `YOUR_USERNAME` | your actual Linux username | `configs/npu-tier/npu-server/default.nix`, `configs/gpu-tier/default.nix`, `configs/flake-excerpts/flake.nix.excerpt` |
| `YOUR_HOSTNAME` | your NixOS host name (whatever you'd put in `networking.hostName`) | `configs/flake-excerpts/flake.nix.excerpt`, `configs/flake-excerpts/nixos-configuration-snippets.nix` |
| `/home/YOUR_USERNAME/models/...` | wherever you actually download/convert your models to | `configs/npu-tier/npu-server/default.nix`, `configs/gpu-tier/default.nix` |

Everything else - ports (`8900`, `8901`, `8899`), service names
(`npu-server-coder`, `gpu-server-hard`, `gpu-guard`), and model names
(`qwen3.6-35b-a3b-gpu`, etc.) - is arbitrary and safe to keep or rename
freely; nothing in this guide's logic depends on those exact strings, they
just need to stay internally consistent across the files you actually use
(e.g. if you rename `gpu-server-hard` to something else, update the
`Conflicts=`/`Requires=` references in the sibling NPU-tier/gpu-guard units
too).

No config file in this repo needs your real IP address, real hostname for
networking purposes, or any secret/API key - every server here is
`127.0.0.1`-only, no auth. `YOUR_HOSTNAME` only matters for NixOS's own
`nixosConfigurations.<name>` attribute naming, not networking.
