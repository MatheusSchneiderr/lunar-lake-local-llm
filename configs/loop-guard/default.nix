{ config, pkgs, lib, ... }:
let
  cfg = config.services.loop-guard;

  # Built from source (buildRustPackage), not fetched as a prebuilt
  # GitHub Release binary - a Nix-machine-built ELF binary legitimately
  # contains literal /nix/store/... paths (its own dynamic linker
  # interpreter path), which a plain `fetchurl` fixed-output derivation's
  # purity check rejects outright. Building from source sidesteps that
  # category of problem entirely. Full rationale and mechanism:
  # docs/14-loop-guard-reasoning-loop-proxy.md and the source repo itself:
  # github.com/MatheusSchneiderr/loop-guard (src/tracker.rs has the
  # detection logic). To update: bump `rev`, `src.sha256`, and
  # loop-guard-Cargo.lock together.
  loop-guard-bin = pkgs.rustPlatform.buildRustPackage {
    pname = "loop-guard";
    version = "0.1.0";
    src = pkgs.fetchFromGitHub {
      owner = "MatheusSchneiderr";
      repo = "loop-guard";
      rev = "0a6cb8bba35b0c596ca3894bfbb8303378ffe546";
      sha256 = "16nychgq8vb88npcbsghkaqx08gbc5r0pig8rqnwnqjzb9xcn7g2";
    };
    cargoLock = {
      lockFile = ./loop-guard-Cargo.lock;
    };
  };
in
{
  options.services.loop-guard = {
    enable = lib.mkEnableOption "loop-guard, the reasoning-loop-detecting reverse proxy in front of the GPU-tier server";
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.loop-guard = {
      Unit = {
        Description = "Reverse proxy in front of the GPU-tier server: detects and interrupts real reasoning loops";
        # Starting loop-guard should always bring up the backend it
        # depends on - point nvim/OpenCode/your start alias at loop-guard
        # instead of the GPU-tier service directly, so a single
        # `systemctl --user start loop-guard` is enough.
        Requires = [ "gpu-server-hard.service" ];
        After = [ "gpu-server-hard.service" ];
      };
      Service = {
        Environment = [
          "LOOP_GUARD_PORT=8901"
          "LOOP_UPSTREAM_HOST=127.0.0.1"
          "LOOP_UPSTREAM_PORT=8902"
        ];
        ExecStart = "${loop-guard-bin}/bin/loop-guard";
        Restart = "on-failure";
      };
      # Deliberately no Install.WantedBy: started on demand via a shell
      # alias / editor keymap, not at login - same pattern as the GPU/NPU
      # tier services themselves (see ../gpu-tier-sycl/, ../npu-tier/).
    };
  };
}
