final: prev: {
  # nixpkgs' intel-npu-driver is built with the driver-side NPU compiler
  # disabled (ENABLE_NPU_COMPILER_BUILD defaults to OFF upstream, and building
  # it from source pulls in a full LLVM/MLIR build). Intel ships this exact
  # component prebuilt as a .deb inside their GitHub release tarball, matching
  # our driver version - extract just the two .so files it needs.
  intel-npu-driver-compiler-package = final.stdenv.mkDerivation {
    pname = "intel-npu-driver-compiler-package";
    version = "1.35.0";

    src = final.fetchurl {
      url = "https://github.com/intel/linux-npu-driver/releases/download/v1.35.0/linux-npu-driver-v1.35.0.20260722-29947505341-ubuntu2404.tar.gz";
      hash = "sha256-OYND5T/axgI60IVu+Iu2ARseEkR6ESvlXoXifvf5bGY=";
    };

    nativeBuildInputs = [ final.dpkg ];

    unpackPhase = ''
      tar -xzf $src "./intel-driver-compiler-npu_1.35.0.20260722-29947505341~ubuntu24.04_amd64.deb"
    '';

    # The .deb only ships the compiler .so files, not its public header -
    # npu_compiler.cmake adds $NPU_COMPILER_PACKAGE_DIR itself (not a subdir) to
    # the include path, so npu_driver_compiler.h must sit directly in $out.
    # Same version (1.35.0) as the driver source we build against.
    npuDriverCompilerHeader = final.fetchurl {
      url = "https://raw.githubusercontent.com/intel/linux-npu-driver/v1.35.0/compiler/include/npu_driver_compiler.h";
      hash = "sha256-mz9qEF/qt+QqwSKRTkObPKwDYjWcBJrZzn9gD7o3RNE=";
    };

    installPhase = ''
      dpkg-deb -x intel-driver-compiler-npu_1.35.0.20260722-29947505341~ubuntu24.04_amd64.deb extracted
      mkdir -p $out/lib
      cp extracted/usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler.so $out/lib/
      cp extracted/usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler_loader.so $out/lib/
      cp $npuDriverCompilerHeader $out/npu_driver_compiler.h
    '';
  };

  intel-npu-driver = prev.intel-npu-driver.overrideAttrs (old: {
    cmakeFlags = (old.cmakeFlags or [ ]) ++ [
      "-DNPU_COMPILER_PACKAGE_DIR=${final.intel-npu-driver-compiler-package}"
      # The prebuilt compiler .so needs libtbb, but its own validation test
      # targets don't declare that link dependency in CMake - force it globally.
      "-DCMAKE_EXE_LINKER_FLAGS=-ltbb"
    ];
    buildInputs = (old.buildInputs or [ ]) ++ [
      final.zlib
      final.zstd
      final.onetbb
    ];
  });
}
