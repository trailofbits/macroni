# Macroni

Macroni is an MLIR dialect that adds macro expansions to VAST's tower of IRS.

## Table of Contents

- [Macroni](#macroni)
  - [Table of Contents](#table-of-contents)
  - [Setting up](#setting-up)
  - [Running Macroni](#running-macroni)
  - [Testing](#testing)
  - [License](#license)

## Setting up

The following instructions assume an Ubuntu 22.04.4 LTS x86_64 operating system:

- Download and build `gap`:

  ```bash
  git clone https://github.com/lifting-bits/gap.git gap/
  cd gap/ && git checkout ad8fefaf7235a9cd6670e272ca4487807ed81f8a && cd ../
  mkdir -p build/gap/Debug/
  cmake -S gap/ -B build/gap/Debug/ -G "Ninja Multi-Config" \
    -DCMAKE_CONFIGURATION_TYPES="Debug" \
    -DCMAKE_INSTALL_PREFIX="`realpath -s .`" \
    -DGAP_ENABLE_EXAMPLES="FALSE" \
    -DGAP_ENABLE_TESTING="FALSE" \
    -DGAP_ENABLE_VCPKG="FALSE" \
    -DGAP_WARNINGS_AS_ERRORS="FALSE"
  cmake --build build/gap/Debug/ --config Debug --target install
  ```

- Download a patched version of LLVM/Clang:

  ```bash
  wget -O llvm-pasta-beeda8d.tar.xz https://github.com/trail-of-forks/llvm-project/releases/download/beeda8d/llvm-pasta-beeda8d.tar.xz
  mkdir -p llvm-pasta-beeda8d
  tar -xvf llvm-pasta-beeda8d.tar.xz --directory llvm-pasta-beeda8d
  ```

- Download and build `PASTA`:

  ```bash
  git clone https://github.com/trailofbits/pasta.git pasta/
  cd pasta/ && git checkout c84abe593f1d1640859a548ba84f5863523a90ce && cd ../
  mkdir -p build/pasta/Release/
  cmake -S pasta/ -B build/pasta/Release/ -G "Ninja" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="`realpath -s .`" \
    -DPASTA_USE_VENDORED_CLANG=OFF \
    -DClang_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/clang`" \
    -DLLVM_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/llvm`" \
    -DCMAKE_C_COMPILER="`realpath -s ./llvm-pasta-beeda8d/bin/clang`" \
    -DCMAKE_CXX_COMPILER="`realpath -s ./llvm-pasta-beeda8d/bin/clang++`" \
    -DPASTA_BOOTSTRAP_MACROS=OFF \
    -DPASTA_BOOTSTRAP_TYPES=OFF \
    -DPASTA_ENABLE_TESTING=OFF \
    -DPASTA_ENABLE_PY_BINDINGS=OFF \
    -DPASTA_ENABLE_INSTALL=ON
  cmake --build build/pasta/Release/ --config Release --target install -j4
  ```

- Download and build `VAST`:

  ```bash
  git clone --depth=1 https://github.com/PappasBrent/vast.git vast/
  mkdir -p build/vast/Debug/
  cmake -S vast/ -B build/vast/Debug/ -G "Ninja Multi-Config" \
    -DCMAKE_INSTALL_PREFIX=. \
    -DCMAKE_CONFIGURATION_TYPES="Debug" \
    -DCMAKE_C_COMPILER="`realpath -s ./llvm-pasta-beeda8d/bin/clang`" \
    -DCMAKE_CXX_COMPILER="`realpath -s ./llvm-pasta-beeda8d/bin/clang++`" \
    -DLLVM_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/llvm`" \
    -DClang_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/clang`" \
    -DMLIR_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/mlir`" \
    -Dgap_DIR="`realpath -s ./lib/cmake/gap`" \
    -DLLVM_USE_LINKER="mold" \
    -DLLVM_ENABLE_RTTI=ON \
    -DVAST_ENABLE_GAP_SUBMODULE=OFF \
    -DVAST_WARNINGS_AS_ERRORS=OFF \
    -DVAST_ENABLE_TESTING=OFF
    cmake --build build/vast/Debug/ --config Debug --target install -j4
  ```

- And finally download and build macroni:

  ```bash
  git clone https://github.com/trailofbits/macroni macroni/
  cmake -S macroni/ -B build/macroni/Debug/ -G "Ninja Multi-Config" \
    -DCMAKE_INSTALL_PREFIX=build/macroni/Debug/ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=true \
    -DCMAKE_BUILD_TYPE="Debug" \
    -DCMAKE_C_COMPILER="`realpath -s ./llvm-pasta-beeda8d/bin/clang`" \
    -DCMAKE_CXX_COMPILER="`realpath -s ./llvm-pasta-beeda8d/bin/clang++`" \
    -DLLVM_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/llvm`" \
    -DClang_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/clang`" \
    -DMLIR_DIR="`realpath -s ./llvm-pasta-beeda8d/lib/cmake/mlir`" \
    -DLLVM_ENABLE_RTTI=true \
    -DMACRONI_WARNINGS_AS_ERRORS=false \
    -DMACRONI_USE_VENDORED_GAP=false \
    -DMACRONI_USE_VENDORED_PASTA=false \
    -DMACRONI_USE_VENDORED_VAST=false \
    -Dgap_DIR="`realpath -s ./lib/cmake/gap/`" \
    -Dpasta_DIR="`realpath -s ./lib/cmake/pasta/`" \
    -DVAST_DIR="`realpath -s ./lib/cmake/VAST/`"
  cmake --build build/macroni/Debug/ --config Debug --target kernelize -j4
  ```

## Running Macroni

Once the `macronify` binary has been built, you can run it on a C source file.
Assuming `macronify` has been installed and its location added to your path:

```bash
macronify -xc some_file.c
```

## Testing

TODO: Update this section to reflect latest changes to Macroni

## License

Macroni is licensed according to the Apache 2.0 license. Macroni links against
and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with
[LLVM](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT)
exceptions.
