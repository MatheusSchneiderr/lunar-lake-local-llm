final: prev: {
  # llama.cpp has no packaged SYCL variant in nixpkgs (only -vulkan/-cuda/-rocm).
  # The SYCL backend needs Intel's proprietary DPC++ compiler (icpx), which
  # only ships in intel-oneapi-toolkit - not in the open generic-sycl-components
  # / adaptivecpp packages. This overlay builds llama.cpp with GGML_SYCL=ON
  # using that compiler. See docs/12-sycl-reversal-and-qwen36-migration-2026-09-10.md
  # section 6 for the full story behind every flag below.
  llama-cpp-sycl = prev.llama-cpp.overrideAttrs (old: {
    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.intel-oneapi-toolkit ];
    buildInputs = (old.buildInputs or [ ]) ++ [
      final.level-zero
      final.ocl-icd
      final.opencl-headers
      final.opencl-clhpp
    ];
    # icx/icpx are prebuilt binaries that assume an FHS layout for the C
    # runtime (Scrt1.o/crti.o/crtbeginS.o) - unlike nixpkgs' stdenv.cc, they
    # don't know where Nix's glibc/gcc actually live, so linking fails with
    # "cannot find Scrt1.o" unless explicitly pointed at both via
    # --gcc-toolchain (crtbegin*.o, from gcc) and --sysroot (Scrt1.o/crti.o,
    # from glibc; libgcc_s.so itself lives in the separate stdenv.cc.cc.lib
    # output, not alongside the compiler, so it needs its own -L; and glibc's
    # headers (stdio.h etc.) live in the separate glibc.dev output at
    # <out>/include rather than <sysroot>/usr/include, which --sysroot alone
    # doesn't cover. Must be added via -idirafter, not -isystem: libstdc++'s
    # own <cstdlib> does `#include_next <stdlib.h>`, which continues the
    # search from the directory AFTER wherever <cstdlib> itself was found
    # (inside gcc's C++ headers) - -isystem doesn't reliably sort after that
    # point, so #include_next never reaches glibc.dev; -idirafter guarantees
    # last-resort placement, after all other search paths including gcc's own.
    cmakeFlags =
      let
        # Bypassing nixpkgs' cc-wrapper (icx/icpx are invoked directly, not
        # through it) means buildInputs' include/lib dirs are never
        # auto-injected via NIX_CFLAGS_COMPILE/NIX_LDFLAGS the way a normal
        # nixpkgs derivation gets them - level-zero and the OpenCL headers
        # need to be added by hand here too, same as the glibc/gcc paths.
        includeFlags = "-idirafter ${final.glibc.dev}/include -I${final.level-zero}/include -I${final.opencl-headers}/include -I${final.opencl-clhpp}/include";
        # -Wl,-rpath is needed alongside -L: bypassing the cc-wrapper means
        # none of these lib dirs get auto-embedded into the binaries' runtime
        # search path the way a normal nixpkgs derivation gets for free, so
        # e.g. llama-server linked fine against libssl but couldn't find it
        # at runtime (--completion-bash install-check step failed on it).
        rpathDirs = [
          "${final.stdenv.cc.cc.lib}/lib"
          "${final.level-zero}/lib"
          "${final.ocl-icd}/lib"
          "${final.openssl.out}/lib"
        ];
        linkFlags = "-L${final.stdenv.cc.cc.lib}/lib -L${final.level-zero}/lib -L${final.ocl-icd}/lib -L${final.openssl.out}/lib "
          + (final.lib.concatMapStringsSep " " (d: "-Wl,-rpath,${d}") rpathDirs);
        toolchainFlags = "--gcc-toolchain=${final.stdenv.cc.cc} --sysroot=${final.glibc} ${includeFlags} ${linkFlags}";
      in
      (builtins.filter
        (f: !(final.lib.hasInfix "GGML_VULKAN" f) && !(final.lib.hasInfix "CMAKE_CXX_COMPILER" f) && !(final.lib.hasInfix "CMAKE_C_COMPILER" f))
        old.cmakeFlags
      ) ++ [
        (final.lib.cmakeBool "GGML_SYCL" true)
        (final.lib.cmakeFeature "GGML_SYCL_TARGET" "INTEL")
        # GGML_SYCL_F16 (dequantize-before-matmul intermediate type fp32->fp16)
        # was tested and REJECTED - see docs/13-qwen36-sycl-fine-tuning-2026-09-11.md
        # section 5. Real benchmarks showing a prefill win are dense-model-only
        # (Llama/Qwen2.5) and don't transfer to this MoE model: it made BOTH
        # sampling presets slower end-to-end on a real agentic coding task,
        # decode included - thinking-on 172s->264s (+53%), thinking-off
        # 57s->97s (+70%). Left off (the default). Do not re-enable without a
        # fresh benchmark if the model or SYCL backend changes materially.
        (final.lib.cmakeFeature "CMAKE_C_COMPILER" "${final.intel-oneapi-toolkit}/2026.0/bin/icx")
        (final.lib.cmakeFeature "CMAKE_CXX_COMPILER" "${final.intel-oneapi-toolkit}/2026.0/bin/icpx")
        (final.lib.cmakeFeature "CMAKE_C_FLAGS" toolchainFlags)
        (final.lib.cmakeFeature "CMAKE_CXX_FLAGS" toolchainFlags)
        (final.lib.cmakeFeature "CMAKE_EXE_LINKER_FLAGS" toolchainFlags)
        (final.lib.cmakeFeature "CMAKE_SHARED_LINKER_FLAGS" toolchainFlags)
      ];
  });
}
