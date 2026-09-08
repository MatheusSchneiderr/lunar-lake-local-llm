# Excerpts from hosts/YOUR_HOSTNAME/configuration.nix and
# hardware-configuration.nix - only the lines actually relevant to this
# setup. A real configuration.nix has plenty of unrelated desktop/hardware
# config too; this is not a complete file.

{ config, lib, pkgs, ... }:

{
  # --- from hardware-configuration.nix ---

  # The single NixOS module toggle that pulls in the intel_vpu kernel
  # driver and udev rules for /dev/accel/accel0. Without this, none of the
  # NPU-tier overlays/config matter - the device won't exist at all.
  hardware.cpu.intel.npu.enable = true;
  hardware.cpu.intel.updateMicrocode = lib.mkDefault config.hardware.enableRedistributableFirmware;

  # --- from configuration.nix ---

  # This machine has no swap at all otherwise - confirmed during local LLM
  # GPU/NPU concurrency testing that free RAM can bottom out under 300MB
  # under normal (non-pathological) load. zram gives the OOM killer a
  # graceful fallback instead of an outright kill; not meant to be relied
  # on for routine operation, just a safety net. See
  # docs/07-benchmarks-and-methodology.md and docs/08-troubleshooting-and-incidents.md
  # for two real incidents where even this didn't fully save a severe
  # enough memory spike - it's a safety net for a surprise, not insurance
  # against a known-bad pattern (e.g. running two model instances at once).
  zramSwap.enable = true;

  # OpenVINO/openvino-genai and several NPU-adjacent packages are unfree.
  nixpkgs.config.allowUnfree = true;

  # Note: NO hardware.graphics/hardware.opengl option is needed anywhere
  # for the Arc iGPU's Vulkan backend to work - that's wired ad hoc at the
  # home-manager/service level instead (see configs/gpu-tier/default.nix's
  # gpuLibraryPath/vkIcd, which reference pkgs.mesa/pkgs.vulkan-loader
  # directly). If Vulkan genuinely can't see your iGPU at all (check with
  # `vulkaninfo`), that's a kernel/firmware-level problem outside this
  # guide's scope, not something this repo's config toggles.
}
