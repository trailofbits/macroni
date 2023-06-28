# Macroni
Macroni is an MLIR dialect that adds macro expansions to VAST's tower of IRS.

## Table of Contents
- [Macroni](#macroni)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Setting up](#setting-up)
  - [Running Macroni](#running-macroni)
  - [Testing](#testing)
  - [License](#license)

<!-- See https://github.com/trailofbits/vast for template -->

## Requirements

The [latest release](https://github.com/lifting-bits/cxx-common/releases/) of
Trail of Bits' cxx-common repository. Choose the release has `pasta` in the name
and matches your platform. This provides `llvm`, `mlir`, and a version of
`clang` with patches to track macro expansion information; Macroni requires all
these dependencies.

## Setting up

Clone Macroni and download its required submodules:

```bash
git clone https://github.com/trailofbits/macroni.git
git submodule init
git submodule update
```

Set the following environment variables:

| Variable name         | Definition                                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `VCPKG_DIR`           | The path to an extracted release of Trail of Bits' cxx-common repository. This provides a path to all of Macroni's dependencies. |
| `LLVM_LIT`            | The path to your install of `llvm-lit`. This enables automated testing of Macroni.                                               |
| `MACRONI_INSTALL_DIR` | The directory to install Macroni to.                                                                                             |
| `MACRONI_SRC`         | The directory macroni has been downloaded to.                                                                                    |
| `MACRONI_BUILD_DIR`   | The directory to build Macroni in.                                                                                               |


Configure Macroni:

```bash
cmake --no-warn-unused-cli \
      -DCMAKE_BUILD_TYPE:STRING=Debug \
      -DCMAKE_TOOLCHAIN_FILE:STRING="${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake" \
      -DLLVM_ENABLE_RTTI:STRING=ON \
      -DLLVM_SYMBOLIZER_PATH:STRING="${VCPKG_DIR}/installed/x64-linux-rel/bin/llvm-symbolizer" \
      -DCMAKE_INSTALL_PREFIX:STRING="${INSTALL_DIR}" \
      -DLLVM_EXTERNAL_LIT:STRING="${LLVM_LIT}" \
      -DVCPKG_TARGET_TRIPLET:STRING=x64-linux \
      -DVCPKG_HOST_TRIPLET:STRING=x64-linux-rel \
      -DCMAKE_PREFIX_PATH:STRING="${VCPKG_DIR}/installed/x64-linux-rel" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
      -S${MACRONI_SRC} \
      -B${MACRONI_BUILD_DIR} \
      -G Ninja
```


Build the `macronify` binary for translating C code to a mix of `VAST` and
`Macroni` MLIR code:
```bash
cmake --build ${MACRONI_BUILD_DIR} --config Debug --target macronify -j 8 --
```

This will build the `macronify` binary and place it in the directory
`${MACRONI_BUILD_DIR}/Macronify`.

## Running Macroni
Once the `macronify` binary has been built, you can run it on a C source file.
Assuming `MACRONIFY` and `INPUT` are environment variables set to the paths to
the `macronify` binary and a C source file, respectively, this command would
look like:

```bash
${MACRONIFY} -xc ${INPUT}
```

## Testing

Macroni uses [llvm-lit](https://llvm.org/docs/CommandGuide/lit.html) to automate
testing. To set up Macroni's test suite, configure Macroni as normal, with the
additional option `-DMACRONI_ENABLE_TESTING=ON`. Then run the following command
to run Macroni's test suite:

```bash
cmake --build ${MACRONI_BUILD_DIR} --config Debug --target check-macroni -j 8 --
```

## License
Macroni is licensed according to the Apache 2.0 license. Macroni links against
and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with
[LLVM](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT)
exceptions.
