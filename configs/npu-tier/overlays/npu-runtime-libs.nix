final: prev: {
  # OpenVINO's NPU plugin resolves its compiler loader relative to wherever
  # libopenvino.so was actually loaded from at runtime (get_ov_lib_path() +
  # an "openvino/" subdir) - nixpkgs' openvino package doesn't ship the
  # compiler there. Rather than rebuild the whole (huge) openvino package
  # with a patched postInstall, build a directory that symlinks in the real
  # openvino libs and adds the compiler alongside them, then point
  # LD_LIBRARY_PATH at THIS directory instead - libopenvino.so gets "loaded
  # from here" and the sibling-directory lookup finds the compiler.
  openvino-npu-runtime = final.runCommand "openvino-npu-runtime" { } ''
    mkdir -p $out/lib/openvino
    for f in ${final.openvino.lib}/lib/*; do
      [ -f "$f" ] && ln -s "$f" "$out/lib/$(basename "$f")"
    done
    for f in ${final.openvino.lib}/lib/openvino/*; do
      [ -f "$f" ] && ln -s "$f" "$out/lib/openvino/$(basename "$f")"
    done
    ln -s ${final.intel-npu-driver-compiler-package}/lib/libopenvino_intel_npu_compiler.so $out/lib/openvino/
    ln -s ${final.intel-npu-driver-compiler-package}/lib/libopenvino_intel_npu_compiler_loader.so $out/lib/openvino/
  '';

  # Everything a process needs on LD_LIBRARY_PATH to actually reach the NPU:
  #  - openvino-npu-runtime: libopenvino.so + the compiler discoverable next to it
  #    (satisfies the OpenVINO plugin's own compiler lookup)
  #  - level-zero: the Level Zero loader
  #  - intel-npu-driver: libze_intel_npu.so.1 (the actual Level Zero NPU backend -
  #    its own zeInit() ALSO dlopens the compiler by bare filename, separately
  #    from the OpenVINO plugin's lookup above, hence needing it listed twice
  #    via two different resolution mechanisms)
  #  - intel-npu-driver-compiler-package: the compiler by bare filename, for the
  #    driver's own internal dlopen (plain LD_LIBRARY_PATH search, no directory trick)
  #  - zlib/zstd/onetbb/stdenv.cc.cc.lib: the compiler binary's own runtime deps -
  #    without these, zeInit() itself fails (ZE_RESULT_ERROR_UNINITIALIZED) because
  #    it probes the compiler at init time, before any model is even loaded
  npuLibraryPath = final.lib.makeLibraryPath [
    final.openvino-npu-runtime
    final.level-zero
    final.intel-npu-driver
    final.intel-npu-driver-compiler-package
    final.zlib
    final.zstd
    final.onetbb
    final.stdenv.cc.cc.lib
  ];
}
