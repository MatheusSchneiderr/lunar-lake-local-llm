{ pkgs, lib, ... }:
let
  gpu-guard = pkgs.stdenv.mkDerivation {
    pname = "gpu-guard";
    version = "0.1.0";
    src = ./.;
    dontConfigure = true;
    buildPhase = ''
      runHook preBuild
      $CXX -std=c++17 -O2 -DNDEBUG \
        -I${pkgs.httplib}/include -I${pkgs.nlohmann_json}/include \
        main.cpp -o gpu-guard -lpthread
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin
      cp gpu-guard $out/bin/gpu-guard
      runHook postInstall
    '';
    # Self-test doubles as the build's correctness check - a real assertion
    # on the detection logic (see main.cpp's run_selftest), not just "it
    # compiled". Fails the build if any check regresses.
    doCheck = true;
    checkPhase = ''
      ./gpu-guard --selftest
    '';
  };
in
{
  # A thin reverse proxy in front of gpu-server-hard (see main.cpp for the
  # full rationale) - written in C++ rather than Python specifically to
  # keep this always-resident process's footprint tiny (a few MB RSS, no
  # model pipeline, unlike npu-server's server.py) since it just forwards
  # bytes for the overwhelming majority of requests and only buffers/
  # inspects the rare tool-bearing ones.
  systemd.user.services.gpu-guard = {
    Unit = {
      Description = "Reverse proxy in front of gpu-server-hard: retries degenerate stalls/false tool-refusals";
      # Starting gpu-guard should always bring up the backend it depends on -
      # this is the service nvim/OpenCode now point at instead of
      # gpu-server-hard directly, so a single `systemctl --user start
      # gpu-guard` needs to be enough.
      Requires = [ "gpu-server-hard.service" ];
      After = [ "gpu-server-hard.service" ];
    };
    Service = {
      Environment = [
        "GPU_GUARD_PORT=8899"
        "GPU_UPSTREAM_HOST=127.0.0.1"
        "GPU_UPSTREAM_PORT=8901"
      ];
      ExecStart = "${gpu-guard}/bin/gpu-guard";
      Restart = "on-failure";
    };
    # Sem Install.WantedBy de propósito - mesmo padrão do gpu-server-hard:
    # start sob demanda via keymap no nvim, não no login.
  };
}
